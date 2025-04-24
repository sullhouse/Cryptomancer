from google.cloud import bigquery
from datetime import datetime, timezone # Import timezone
from typing import Dict, Optional, Tuple, Any
from wallet import Wallet, Transaction

class Exchange:
    def __init__(self, fees: Dict[str, Dict[str, float]], slippage: float = 0.0):
        """
        :param fees: dict of pair_to_swap (e.g. 'ETH_USDC') to dict with 'flat_fee' and 'perc_fee'
        :param slippage: decimal between 0 and 1 representing slippage
        """
        self.fees = fees
        self.slippage = slippage

        # Pairs where the table name matches the pair string format (primary-secondary)
        self.base_pairs = [
            "ETH_USDC",
            "BTC_USDC",
            "ETH_BTC",
            "ETH_MATIC",
            "MATIC_USDC"
            # Note: BTC_ETH needs special handling if ETH_BTC table is used
        ]
        # All pairs the exchange conceptually supports (including inverted ones)
        self.supported_pairs = self.base_pairs + ["BTC_ETH", "USDC_ETH", "USDC_BTC", "USDC_MATIC"] # Add inverted pairs

        # Cache attributes
        self.rate_cache: Dict[datetime, Dict[str, float]] = {} # Cache structure: {timestamp: {pair: rate}}
        self.cache_start_time: Optional[datetime] = None
        self.cache_end_time: Optional[datetime] = None

    def cache_rates(self, start_time: datetime, end_time: datetime):
        """
        Queries BigQuery for exchange rates for all base pairs in the given range
        and stores them in memory. Ensures start/end times are treated as UTC.
        :param start_time: Start of the time range (inclusive). Assumed UTC if naive.
        :param end_time: End of the time range (exclusive). Assumed UTC if naive.
        """
        # --- Standardize start_time to UTC ---
        utc_start_time = start_time
        if utc_start_time.tzinfo is None:
            utc_start_time = utc_start_time.replace(tzinfo=timezone.utc)
        else:
            utc_start_time = utc_start_time.astimezone(timezone.utc)
        # --- End Standardization ---

        # --- Standardize end_time to UTC ---
        utc_end_time = end_time
        if utc_end_time.tzinfo is None:
            utc_end_time = utc_end_time.replace(tzinfo=timezone.utc)
        else:
            utc_end_time = utc_end_time.astimezone(timezone.utc)
        # --- End Standardization ---

        print(f"Caching exchange rates from {utc_start_time} to {utc_end_time}...")
        client = bigquery.Client()
        self.rate_cache = {} # Clear previous cache

        for pair in self.base_pairs:
            primary, secondary = self._parse_pair(pair)
            table_name = f"{primary.lower()}-{secondary.lower()}"
            print(f"  Querying {table_name}...")
            query = f"""
                SELECT TIMESTAMP, CLOSE
                FROM `cryptomancer-456619.Coindesk.{table_name}`
                WHERE TIMESTAMP >= @start_time AND TIMESTAMP < @end_time
                ORDER BY TIMESTAMP
            """
            job_config = bigquery.QueryJobConfig(
                query_parameters=[
                    bigquery.ScalarQueryParameter("start_time", "TIMESTAMP", utc_start_time),
                    bigquery.ScalarQueryParameter("end_time", "TIMESTAMP", utc_end_time),
                ]
            )
            try:
                query_job = client.query(query, job_config=job_config)
                count = 0
                for row in query_job.result():
                    ts = row.TIMESTAMP # Already UTC from BigQuery
                    rate = row.CLOSE
                    if ts not in self.rate_cache:
                        self.rate_cache[ts] = {}
                    self.rate_cache[ts][pair] = rate
                    count += 1
                print(f"    Cached {count} rates for {pair}")
            except Exception as e:
                print(f"    Error caching rates for {pair}: {e}")


        # Store the standardized UTC times
        self.cache_start_time = utc_start_time
        self.cache_end_time = utc_end_time
        print(f"Finished caching rates for {len(self.rate_cache)} timestamps.")


    def get_exchange_rate(self, pair_to_swap: str, minute: datetime) -> Optional[float]:
        """
        Get the exchange rate for the given pair and minute. Checks cache first.
        If the primary currency is USDC (e.g. USDC_ETH), invert the rate from the secondary-primary table.
        """
        if pair_to_swap not in self.supported_pairs:
             print(f"Warning: Pair {pair_to_swap} not explicitly supported.")
             # Decide if you want to proceed or return None here
             # return None

        # --- Standardize minute to UTC ---
        lookup_minute = minute
        if lookup_minute.tzinfo is None:
            lookup_minute = lookup_minute.replace(tzinfo=timezone.utc)
        else:
            lookup_minute = lookup_minute.astimezone(timezone.utc)
        # --- End Standardization ---

        primary, secondary = self._parse_pair(pair_to_swap)
        base_pair = pair_to_swap
        invert = False

        # Determine the base pair and if inversion is needed
        if primary == "USDC":
            base_pair = f"{secondary}_{primary}" # e.g., ETH_USDC
            invert = True
        elif pair_to_swap == "BTC_ETH": # Special case if using ETH_BTC table
             base_pair = "ETH_BTC"
             invert = True
        # Add other inversion cases if necessary (e.g., MATIC_ETH based on ETH_MATIC)

        # Check cache first
        cached_rate = None
        if self.rate_cache and self.cache_start_time and self.cache_end_time:
            if self.cache_start_time <= lookup_minute < self.cache_end_time:
                timestamp_cache = self.rate_cache.get(lookup_minute)
                if timestamp_cache:
                    cached_rate = timestamp_cache.get(base_pair)
                    # print(f"Cache hit for {base_pair} at {lookup_minute}") # Debug

        rate_to_use = None
        if cached_rate is not None:
            rate_to_use = cached_rate
        else:
            # If not in cache or outside range, query BigQuery for the specific minute
            # print(f"Cache miss for {base_pair} at {lookup_minute}, querying BQ...") # Debug
            client = bigquery.Client()
            # Use the table name corresponding to the base_pair
            bp_primary, bp_secondary = self._parse_pair(base_pair)
            table_name = f"{bp_primary.lower()}-{bp_secondary.lower()}"

            query = f"""
                SELECT CLOSE
                FROM `cryptomancer-456619.Coindesk.{table_name}`
                WHERE TIMESTAMP = @minute
                LIMIT 1
            """
            job_config = bigquery.QueryJobConfig(
                query_parameters=[
                    # Use the original 'minute' for BQ query if it expects that,
                    # or lookup_minute if BQ expects UTC. Assuming BQ expects UTC here.
                    bigquery.ScalarQueryParameter("minute", "TIMESTAMP", lookup_minute)
                ]
            )
            try:
                query_job = client.query(query, job_config=job_config)
                result = query_job.result()
                row = next(result, None)
                if row and row.CLOSE is not None:
                    rate_to_use = row.CLOSE
                    # Optionally add this single fetched rate to cache?
                    # if lookup_minute not in self.rate_cache: self.rate_cache[lookup_minute] = {}
                    # self.rate_cache[lookup_minute][base_pair] = rate_to_use
            except Exception as e:
                 print(f"Error querying BQ for {base_pair} at {lookup_minute}: {e}")
                 return None # Failed to get rate

        # Apply inversion if needed and rate was found
        if rate_to_use is not None:
            if invert:
                if rate_to_use == 0: return None # Avoid division by zero
                return 1 / rate_to_use
            else:
                return rate_to_use
        else:
            # print(f"Rate not found for {base_pair} at {lookup_minute} in cache or BQ.") # Debug
            return None # Rate not found

    def _parse_pair(self, pair_to_swap: str) -> Tuple[str, str]:
        """Helper to split pair string like 'ETH_USDC' into ('ETH', 'USDC')"""
        return tuple(pair_to_swap.split('_'))

    def submit_trade(self, timestamp: datetime, pair_to_swap: str, wallet: Wallet, amount: float) -> Dict[str, Any]:
        """
        Attempt to execute a trade of 'amount' in the primary currency of pair_to_swap.
        Returns dict with 'success', 'transactions', and 'reason' if failed.
        """
        primary, secondary = self._parse_pair(pair_to_swap)
        rate = self.get_exchange_rate(pair_to_swap, timestamp)
        if rate is None:
            return {"success": False, "reason": "No exchange rate available", "transactions": []}

        # Apply slippage
        effective_rate = rate * (1 - self.slippage)

        # Fees
        fee_conf = self.fees.get(pair_to_swap, {})
        flat_fee = fee_conf.get('flat_fee', 0)
        perc_fee = fee_conf.get('perc_fee', 0)

        # Calculate fee in primary currency
        fee_amount = flat_fee + (amount * perc_fee)
        total_amount = amount + fee_amount

        # Check wallet balance
        if wallet.holdings.get(primary, 0) < total_amount:
            return {"success": False, "reason": "Insufficient funds", "transactions": []}

        # Fee transaction
        fee_tx = Transaction(
            timestamp=timestamp,
            changes={primary: -fee_amount},
            note=f"Fee for swapping {amount} {primary} to {secondary}",
        )
        fee_tx.type = "EXCHANGE FEE"
        wallet.transactions.append(fee_tx)
        wallet.holdings[primary] -= fee_amount
        wallet.history[timestamp] = wallet.holdings.copy()

        # Swap transaction
        received = amount * effective_rate
        swap_tx = Transaction(
            timestamp=timestamp,
            changes={primary: -amount, secondary: received},
            note=f"Swapped {amount} {primary} to {received} {secondary} at rate {effective_rate}",
        )
        swap_tx.type = "SWAP"
        wallet.transactions.append(swap_tx)
        wallet.holdings[primary] -= amount
        wallet.holdings[secondary] = wallet.holdings.get(secondary, 0) + received
        wallet.history[timestamp] = wallet.holdings.copy()

        return {
            "success": True,
            "transactions": [fee_tx, swap_tx],
            "details": {
                "rate": rate,
                "effective_rate": effective_rate,
                "fee_amount": fee_amount,
                "swapped_amount": amount,
                "received_amount": received
            }
        }

    def submit_max_trade(self, timestamp: datetime, pair_to_swap: str, wallet: Wallet) -> Dict[str, Any]:
        """
        Attempt to swap the maximum allowed of the primary currency in pair_to_swap.
        Returns dict with 'success', 'transactions', and 'reason' if failed.
        Explicitly sets the source currency balance to 0 after a successful trade.
        """
        primary, secondary = self._parse_pair(pair_to_swap)
        balance = wallet.holdings.get(primary, 0)
        fee_conf = self.fees.get(pair_to_swap, {})
        flat_fee = fee_conf.get('flat_fee', 0)
        perc_fee = fee_conf.get('perc_fee', 0)

        # Solve for max amount: balance = amount + flat_fee + amount * perc_fee
        # => amount = (balance - flat_fee) / (1 + perc_fee)
        denom = 1 + perc_fee
        if denom == 0 or balance <= flat_fee: # Check if balance can cover flat fee
            return {"success": False, "reason": "Insufficient funds for flat fee", "transactions": []}

        max_amount = (balance - flat_fee) / denom
        if max_amount <= 0:
            return {"success": False, "reason": "Insufficient funds after fees", "transactions": []}

        # Execute the trade using the calculated max amount
        result = self.submit_trade(timestamp, pair_to_swap, wallet, max_amount)

        # If the trade was successful, explicitly zero out the balance
        # to avoid floating point residuals.
        if result.get("success"):
            wallet.holdings[primary] = 0.0
            # Update history again to reflect the explicit zeroing
            wallet.history[timestamp] = wallet.holdings.copy()

        return result

    def wallet_value_in_usdc(self, wallet: Wallet, timestamp: datetime) -> float:
        """
        Returns the total value of the wallet converted to USDC at the given timestamp.
        """
        total_usdc = wallet.holdings.get("USDC", 0)
        for currency, amount in wallet.holdings.items():
            if currency == "USDC" or amount <= 0:
                continue
            pair = f"{currency}_USDC"
            rate = self.get_exchange_rate(pair, timestamp)
            if rate is not None:
                total_usdc += amount * rate
        return total_usdc