from datetime import datetime, timedelta, timezone
from typing import Dict, Any, Optional
from wallet import Wallet
from exchange import Exchange
from dataclasses import dataclass
from google.cloud import bigquery

@dataclass
class RiskProfile:
    min_confidence: float  # e.g., 0.7
    min_time_since_last_swap: timedelta  # e.g., timedelta(hours=3)
    max_daily_trades: int  # e.g., 5
    trade_amount: float  # e.g., 0.3 (30% of holding)
    prediction_horizon_hours: int  # 1, 3, 6, 12, or 24
    always_max: bool  # True to always trade max

class Speculator:
    def __init__(
        self,
        initial_holdings: Dict[str, float],
        start_time: datetime,
        exchange_fees: Dict[str, Dict[str, float]],
        slippage: float,
        risk_profile: RiskProfile
    ):
        self.wallet = Wallet(initial_holdings, start_time)
        self.exchange = Exchange(exchange_fees, slippage)
        self.risk_profile = risk_profile
        self.last_swap_time: Dict[str, datetime] = {}
        self.daily_trade_count: Dict[str, int] = {}
        # Cache attributes
        self.prediction_cache: Dict[datetime, Dict] = {}
        self.cache_start_time: Optional[datetime] = None
        self.cache_end_time: Optional[datetime] = None

    def cache_predictions(self, start_time: datetime, end_time: datetime):
        """
        Queries BigQuery for predictions in the given range and stores them in memory.
        Ensures start/end times are treated as UTC.
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

        print(f"Caching predictions from {utc_start_time} to {utc_end_time}...")
        client = bigquery.Client()
        query = """
            SELECT *
            FROM `cryptomancer-456619.AI.predictions`
            WHERE run_timestamp >= @start_time AND run_timestamp < @end_time
            ORDER BY run_timestamp
        """
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                # Use the standardized UTC times for the query as well
                bigquery.ScalarQueryParameter("start_time", "TIMESTAMP", utc_start_time),
                bigquery.ScalarQueryParameter("end_time", "TIMESTAMP", utc_end_time),
            ]
        )
        query_job = client.query(query, job_config=job_config)
        # Keys (row.run_timestamp) are already UTC from BigQuery
        self.prediction_cache = {row.run_timestamp: dict(row) for row in query_job.result()}
        # Store the standardized UTC times
        self.cache_start_time = utc_start_time
        self.cache_end_time = utc_end_time
        print(f"Cached {len(self.prediction_cache)} predictions.")

    def get_prediction(self, hour: datetime) -> Optional[Dict]:
        """
        Returns the predictions and confidences for the given hour.
        Checks cache first, then queries BigQuery if not found or outside cache range.
        Ensures the hour is treated as UTC for cache lookup.
        :param hour: datetime object representing the hour to query. Assumed UTC if naive.
        :return: dict with prediction fields, or None if not found.
        """
        # --- Standardize hour to UTC ---
        lookup_hour = hour
        if lookup_hour.tzinfo is None:
            # If naive, assume UTC (or adjust based on your input data's actual timezone)
            lookup_hour = lookup_hour.replace(tzinfo=timezone.utc)
        else:
            # If aware but not UTC, convert to UTC
            lookup_hour = lookup_hour.astimezone(timezone.utc)
        # --- End Standardization ---

        # Check cache first using the standardized UTC hour
        if self.prediction_cache and self.cache_start_time and self.cache_end_time:
            # Ensure cache boundaries are also UTC aware if they aren't already
            # (Assuming start_time/end_time passed to cache_predictions were handled correctly)
            if self.cache_start_time <= lookup_hour < self.cache_end_time:
                cached_pred = self.prediction_cache.get(lookup_hour) # Use lookup_hour
                if cached_pred:
                    # print(f"Cache hit for {lookup_hour}") # Optional: for debugging
                    return cached_pred
                else:
                    # Hour is within cache range, but no prediction exists for it
                    # print(f"Cache miss (no data) for {lookup_hour}") # Optional: for debugging
                    return None

        # If not in cache or outside range, query BigQuery (using original hour for query param)
        # print(f"Cache miss (querying BQ) for {hour}") # Optional: for debugging
        client = bigquery.Client()
        query = """
            SELECT *
            FROM `cryptomancer-456619.AI.predictions`
            WHERE run_timestamp = @hour
            LIMIT 1
        """
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("hour", "TIMESTAMP", hour) # Use original hour for BQ
            ]
        )
        query_job = client.query(query, job_config=job_config)
        result = query_job.result()
        row = next(result, None)
        if row:
            # Store in cache using UTC key if caching individual misses (optional)
            # utc_timestamp = row.run_timestamp # Already UTC from BQ
            # self.prediction_cache[utc_timestamp] = dict(row)
            return dict(row)
        return None

    def should_trade(self, timestamp: datetime) -> Dict[str, Any]:
        prediction = self.get_prediction(timestamp)
        if not prediction:
            return {"should_trade": False, "reason": "No prediction data"}

        # Reset daily trade count if new day
        day_str = timestamp.strftime("%Y-%m-%d")
        if day_str not in self.daily_trade_count:
            self.daily_trade_count = {day_str: 0}

        trades_to_make = []
        supported_currencies = ["ETH", "BTC"] # Ensure BTC is included if needed

        for currency in supported_currencies:
            horizon = self.risk_profile.prediction_horizon_hours
            # Determine correct key for singular/plural "hour(s)"
            if horizon == 1:
                dir_key = f"{currency.lower()}_{horizon}_hour_direction"
                conf_key = f"{currency.lower()}_{horizon}_hour_confidence"
            else:
                dir_key = f"{currency.lower()}_{horizon}_hours_direction"
                conf_key = f"{currency.lower()}_{horizon}_hours_confidence"
            direction = prediction.get(dir_key)
            confidence = prediction.get(conf_key)

            if direction in [1, -1] and confidence is not None and confidence >= self.risk_profile.min_confidence:
                # Check min_time_since_last_swap
                # Use a combined key for last swap time to handle USDC swaps correctly
                swap_key_buy = f"USDC_{currency}"
                swap_key_sell = f"{currency}_USDC"
                last_swap_buy = self.last_swap_time.get(swap_key_buy)
                last_swap_sell = self.last_swap_time.get(swap_key_sell)
                last_swap = max(last_swap_buy, last_swap_sell) if last_swap_buy and last_swap_sell else (last_swap_buy or last_swap_sell)

                if last_swap is not None and (timestamp - last_swap) < self.risk_profile.min_time_since_last_swap:
                    continue
                # Check max_daily_trades
                if self.daily_trade_count.get(day_str, 0) >= self.risk_profile.max_daily_trades:
                    continue

                if direction == 1:
                    # Predicting up: swap USDC to currency if possible
                    if self.wallet.holdings.get("USDC", 0) > 0:
                        pair = f"USDC_{currency}"
                        trades_to_make.append({"pair": pair, "from": "USDC", "to": currency})
                elif direction == -1:
                    # Predicting down: swap currency to USDC if possible
                    if self.wallet.holdings.get(currency, 0) > 0:
                        pair = f"{currency}_USDC"
                        trades_to_make.append({"pair": pair, "from": currency, "to": "USDC"})

        if not trades_to_make:
            return {"should_trade": False, "reason": "No trade meets criteria"}

        # Execute trades
        results = []
        for trade in trades_to_make:
            pair = trade["pair"]
            from_currency = trade["from"]
            amount = self.wallet.holdings.get(from_currency, 0)
            if amount <= 0:
                continue

            if self.risk_profile.always_max:
                result = self.exchange.submit_max_trade(timestamp, pair, self.wallet)
            else:
                trade_amt = amount * self.risk_profile.trade_amount
                result = self.exchange.submit_trade(timestamp, pair, self.wallet, trade_amt)

            if result.get("success"):
                # Update last swap time using the pair as the key
                self.last_swap_time[pair] = timestamp
                self.daily_trade_count[day_str] = self.daily_trade_count.get(day_str, 0) + 1
            results.append(result)

        return {
            "should_trade": True,
            "trades_executed": results
        }

    def get_wallet_value_in_usdc(self, timestamp: datetime) -> float:
        """
        Returns the total value of the speculator's wallet in USDC at the given timestamp.
        """
        return self.exchange.wallet_value_in_usdc(self.wallet, timestamp)