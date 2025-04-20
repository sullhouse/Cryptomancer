import pandas as pd
import numpy as np
import os
from cache_utils import write_csv, write_dataframe

# --- Constants for filenames ---
PREDICTION_CACHE_SUBDIR = os.environ.get("GCS_PREDICTION_CACHE_FOLDER", "prediction_cache")
PORTFOLIO_FILENAME = "portfolio_history.csv"
TRADES_FILENAME = "trade_log.csv"

class TradeSimulator:
    def __init__(self, pair_to_swap, goal, fee, slippage, cooldown_period, confidence_threshold, hold_band, trade_fraction):
        """
        Initialize the simulator with configurable parameters.

        Args:
            pair_to_swap (str): The trading pair (e.g., 'ETH_USDC', 'ETH_MATIC').
            goal (str): The simulation goal (e.g., 'maximize_usdc', 'maximize_btc').
            fee (float): Transaction fee as a fraction (e.g., 0.001 for 0.1%).
            slippage (float): Market slippage as a fraction (e.g., 0.002 for 0.2%).
            cooldown_period (int): Number of periods to wait after a trade.
            confidence_threshold (float): Confidence threshold to trigger a buy.
            hold_band (tuple): Probability range (low, high) to hold current position.
            trade_fraction (float): Fraction of available capital to use per trade.
        """
        self.pair_to_swap = pair_to_swap
        self.goal = goal
        self.fee = fee
        self.slippage = slippage
        self.cooldown_period = cooldown_period
        self.confidence_threshold = confidence_threshold
        self.hold_band = hold_band
        self.trade_fraction = trade_fraction
        self.portfolio = {}  # Track balances of assets (e.g., {'ETH': 1.0, 'USDC': 0.0})
        self.initial_portfolio = {}  # Initial portfolio balances
        self.initial_value = 0.0  # Initial portfolio value
        self.trade_log = []  # Log of executed trades
        self.cooldown_counter = 0

    def initialize_portfolio(self, initial_balances):
        """Set initial portfolio balances."""
        # Ensure all assets involved in the pair are in the initial balances
        base_asset, quote_asset = self.pair_to_swap.split('_')
        if base_asset not in initial_balances:
             initial_balances[base_asset] = 0.0
             print(f"Warning: Base asset {base_asset} not in initial_portfolio, setting to 0.0")
        if quote_asset not in initial_balances:
             initial_balances[quote_asset] = 0.0
             print(f"Warning: Quote asset {quote_asset} not in initial_portfolio, setting to 0.0")

        self.portfolio = initial_balances.copy() # Use a copy
        self.initial_portfolio = initial_balances.copy()

    def execute_trade(self, action, amount, price, timestamp): # Add timestamp parameter
        """
        Execute a trade (buy or sell) and update portfolio balances. Includes timestamp in log.
        """
        base_asset = self.pair_to_swap.split('_')[0]
        quote_asset = self.pair_to_swap.split('_')[1]

        # Ensure assets exist in portfolio dictionary before trade attempt
        if base_asset not in self.portfolio: self.portfolio[base_asset] = 0.0
        if quote_asset not in self.portfolio: self.portfolio[quote_asset] = 0.0


        if amount <= 1e-9: # Use a small threshold to avoid negligible trades
            print(f"Skipping {action}: amount {amount:.18f} too small.")
            return False

        # Adjust price for slippage
        adjusted_price = price * (1 + self.slippage) if action == 'buy' else price * (1 - self.slippage)
        if adjusted_price <= 0:
             print(f"Skipping {action}: adjusted price {adjusted_price} invalid.")
             return False # Avoid division by zero or negative prices

        if action == 'buy': # Buy base asset using quote asset
            # Calculate cost including fee
            cost = amount * adjusted_price * (1 + self.fee)
            # Check if cost exceeds available quote asset
            available_quote = self.portfolio[quote_asset]
            if cost > available_quote:
                print(f"Attempted buy cost {cost:.6f} {quote_asset} exceeds available {available_quote:.6f}. Adjusting amount.")
                # Adjust amount to the maximum affordable, if possible
                if available_quote < 1e-9:
                     print(f"Skipping buy: available {quote_asset} is negligible.")
                     return False
                # Calculate max affordable base amount based on available quote
                # amount = available_quote / (adjusted_price * (1 + self.fee)) # This calculates max base amount
                # Recalculate cost based on using all available quote
                cost = available_quote # Use all available quote asset
                # Calculate the base amount received for this cost
                amount = cost / (adjusted_price * (1 + self.fee)) # amount_base = cost_quote / (price * (1+fee))
                if amount <= 1e-9:
                     print(f"Skipping buy: adjusted affordable amount {amount:.18f} too small.")
                     return False
                print(f"Adjusted buy amount to {amount:.8f} {base_asset} for cost {cost:.6f} {quote_asset}")


            fee_paid = amount * adjusted_price * self.fee # Fee is paid in quote asset
            self.portfolio[base_asset] += amount
            self.portfolio[quote_asset] -= cost
            # Store timestamp directly, convert later before returning JSON
            self.trade_log.append({'timestamp': timestamp, 'action': 'buy', 'pair': self.pair_to_swap, 'amount_base': amount, 'price': adjusted_price, 'cost_quote': cost, 'fee_quote': fee_paid})
            return True

        elif action == 'sell': # Sell base asset for quote asset
             available_base = self.portfolio[base_asset]
             # Check if amount to sell exceeds available base asset
             if amount > available_base:
                 print(f"Attempted sell amount {amount:.8f} {base_asset} exceeds available {available_base:.8f}. Selling all.")
                 amount = available_base # Sell all available base asset
                 if amount <= 1e-9:
                      print(f"Skipping sell: available {base_asset} is negligible.")
                      return False

             # Calculate revenue after fee
             revenue = amount * adjusted_price * (1 - self.fee) # Revenue is received in quote asset
             fee_paid = amount * adjusted_price * self.fee # Fee is implicitly paid from revenue
             self.portfolio[base_asset] -= amount
             self.portfolio[quote_asset] += revenue
             # Store timestamp directly, convert later before returning JSON
             self.trade_log.append({'timestamp': timestamp, 'action': 'sell', 'pair': self.pair_to_swap, 'amount_base': amount, 'price': adjusted_price, 'revenue_quote': revenue, 'fee_quote': fee_paid})
             return True
        return False

    def calculate_portfolio_value(self, current_row, portfolio_override=None):
        """
        Calculates the total portfolio value in USDC.
        Requires *_USDC_CLOSE columns in current_row for all non-USDC assets held.
        """
        portfolio_to_use = portfolio_override if portfolio_override is not None else self.portfolio
        total_usdc_value = 0.0
        timestamp_str = current_row.get('timestamp', 'Unknown Timestamp') # Get timestamp for logging

        for asset, balance in portfolio_to_use.items():
            if balance == 0: # Skip assets with zero balance
                continue

            if asset == 'USDC':
                total_usdc_value += balance
            else:
                # Construct the required price column name (e.g., ETH_USDC_CLOSE)
                price_col = f'{asset}_USDC_CLOSE'

                if price_col not in current_row or pd.isna(current_row[price_col]):
                    print(f"Warning: Missing or NaN price column '{price_col}' at timestamp {timestamp_str}. Cannot value {asset}.")
                    # Option 1: Return None or NaN to indicate failure (might require downstream handling)
                    # return None
                    # Option 2: Skip this asset's value (results in underestimation)
                    continue
                    # Option 3: Could try to use the traded pair's price if applicable, but less accurate
                    # if asset == self.pair_to_swap.split('_')[0] and f'{self.pair_to_swap}_CLOSE' in current_row:
                    #    # Fallback logic - less ideal
                else:
                    price_in_usdc = current_row[price_col]
                    if price_in_usdc <= 0:
                         print(f"Warning: Invalid non-positive price ({price_in_usdc}) for {price_col} at {timestamp_str}. Skipping {asset} value.")
                         continue

                    total_usdc_value += balance * price_in_usdc

        return total_usdc_value

    def prepare_summary(self, df_sim, portfolio_df):
        """Prepares a summary dictionary of the simulation results. Converts timestamps to strings."""
        if portfolio_df.empty:
            return {"error": "Portfolio history is empty, cannot generate summary."}
        if df_sim.empty:
             return {"error": "Simulation data frame is empty, cannot calculate benchmark."}

        start_value = self.initial_value # Already calculated in USDC at simulation start
        end_value = portfolio_df['Portfolio_Value'].iloc[-1]
        total_return = (end_value - start_value) / start_value if start_value > 1e-9 else 0 # Avoid division by zero
        num_trades = len(self.trade_log)
        total_fees = sum(trade.get('fee_quote', 0.0) for trade in self.trade_log) # Assumes fees are in quote asset, might need refinement if goal asset != quote asset

        # --- Corrected Buy & Hold Benchmark Calculation (in USDC) ---
        buy_hold_end_value = 0.0
        final_row = df_sim.iloc[-1] # Get the last row for final prices
        final_timestamp_str = final_row.get('timestamp', 'Unknown Timestamp')

        for asset, initial_balance in self.initial_portfolio.items():
            if initial_balance == 0:
                continue

            if asset == 'USDC':
                buy_hold_end_value += initial_balance
            else:
                # Get the final USDC price for this asset
                price_col = f'{asset}_USDC_CLOSE'
                if price_col not in final_row or pd.isna(final_row[price_col]):
                    print(f"Warning (Benchmark): Missing or NaN final price '{price_col}' at {final_timestamp_str}. Cannot value initial {asset} for benchmark.")
                    # Decide how to handle: Could skip, or assume initial value (conservative), or return error
                    # Skipping for now, leading to potentially underestimated benchmark
                    continue
                else:
                    final_price_in_usdc = final_row[price_col]
                    if final_price_in_usdc <= 0:
                         print(f"Warning (Benchmark): Invalid non-positive final price ({final_price_in_usdc}) for {price_col} at {final_timestamp_str}. Skipping initial {asset} value.")
                         continue
                    buy_hold_end_value += initial_balance * final_price_in_usdc

        # Calculate benchmark return based on USDC values
        buy_hold_return = (buy_hold_end_value - start_value) / start_value if start_value > 1e-9 else 0 # Avoid division by zero
        # --- End Corrected Benchmark Calculation ---


        # Convert timestamps to ISO format strings for JSON serialization
        start_ts = portfolio_df['timestamp'].iloc[0]
        end_ts = portfolio_df['timestamp'].iloc[-1]
        start_ts_str = start_ts.isoformat() if pd.notna(start_ts) else None
        end_ts_str = end_ts.isoformat() if pd.notna(end_ts) else None


        summary = {
            "pair": self.pair_to_swap,
            "goal": self.goal,
            "start_timestamp": start_ts_str, # Use string version
            "end_timestamp": end_ts_str,     # Use string version
            "initial_portfolio_value": start_value,
            "final_portfolio_value": end_value,
            "total_return_pct": total_return * 100,
            "benchmark_return_pct": buy_hold_return * 100, # Corrected benchmark name
            "strategy_vs_benchmark_pct_points": (total_return - buy_hold_return) * 100, # Renamed for clarity
            "num_trades": num_trades,
            "total_fees_paid": total_fees, # Still assumes fee is in quote asset
            "final_portfolio_balances": self.portfolio,
             # Add simulation parameters
            "params": {
                "confidence_threshold": self.confidence_threshold,
                "hold_band": self.hold_band,
                "trade_fraction": self.trade_fraction,
                "fee": self.fee,
                "slippage": self.slippage,
                "cooldown_period": self.cooldown_period
            }
        }
        return summary

    def save_results(self, portfolio_df, trade_log_df, cache_location, pair):
        """Saves portfolio history and trade log to CSV in the pair's folder."""
        portfolio_filename = f"simulation_portfolio_history.csv"
        trade_log_filename = f"simulation_trade_log.csv"

        # Construct full paths using os.path.join
        base_cache_path = os.path.join(os.getcwd(), cache_location, pair)
        portfolio_full_path = os.path.join(base_cache_path, portfolio_filename)
        trade_log_full_path = os.path.join(base_cache_path, trade_log_filename)


        # Ensure directory exists using os.makedirs
        os.makedirs(os.path.dirname(portfolio_full_path), exist_ok=True)
        # No need to call makedirs again for trade_log if it's the same directory

        print(f"Saving portfolio history to: {portfolio_full_path}")
        portfolio_df.to_csv(portfolio_full_path, index=False)

        if not trade_log_df.empty:
            print(f"Saving trade log to: {trade_log_full_path}")
            trade_log_df.to_csv(trade_log_full_path, index=False)
        else:
            print("Trade log is empty, not saving file.")


    def simulate(self, df_sim, pair, cache_location): # Add pair and cache_location
        """
        Run the simulation loop for the specific pair. Saves results to pair's folder.
        Converts timestamps in results to strings for JSON compatibility.

        Args:
            df_sim (pd.DataFrame): DataFrame with simulation data (prices, predictions for this pair).
            pair (str): The pair being simulated (e.g., "ETH_USDC"). Used for saving results.
            cache_location (str): 'local' or 'gcp'.

        Returns:
            dict: Simulation results including portfolio history, trade log, and summary (with timestamps as strings).
        """
        portfolio_history = []
        self.trade_log = [] # Reset trade log
        self.cooldown_counter = 0 # Reset cooldown

        if not self.initial_portfolio:
             raise ValueError("Initial portfolio not set. Call initialize_portfolio first.")
        # Calculate initial value based on the first row's price
        self.initial_value = self.calculate_portfolio_value(df_sim.iloc[0])
        if self.initial_value is None:
             raise ValueError("Could not calculate initial portfolio value. Check prices in the first row.")
        print(f"Simulation Start. Initial Portfolio: {self.initial_portfolio}, Initial Value ({self.pair_to_swap.split('_')[1]}): {self.initial_value:.4f}")


        # Dynamically construct the prediction column name the simulator expects
        prediction_col_name = f"Predicted_Prob_{self.pair_to_swap.upper()}_UP"
        price_col_name = f'{self.pair_to_swap}_CLOSE'

        if prediction_col_name not in df_sim.columns:
             raise ValueError(f"Required prediction column '{prediction_col_name}' not found in input DataFrame.")
        if price_col_name not in df_sim.columns:
             raise ValueError(f"Required price column '{price_col_name}' not found in input DataFrame.")


        for index, row in df_sim.iterrows():
            action = "Hold"
            current_timestamp = row['timestamp'] # Keep as Timestamp object for now
            confidence = row[prediction_col_name]
            price = row[price_col_name]

            # Check for NaN price or confidence
            if pd.isna(price) or pd.isna(confidence):
                 action = "Hold (Missing Price/Confidence)"
                 print(f"Holding at {current_timestamp} due to missing Price ({price}) or Confidence ({confidence})")
            elif self.cooldown_counter > 0:
                action = f"Hold (Cooldown {self.cooldown_counter})"
                self.cooldown_counter -= 1
            else:
                # Decision logic using the specific pair's confidence
                if confidence >= self.confidence_threshold:
                    # Buy logic (buy base asset using quote asset)
                    base_asset = self.pair_to_swap.split('_')[0]
                    quote_asset = self.pair_to_swap.split('_')[1]
                    buy_amount_quote = self.portfolio.get(quote_asset, 0.0) * self.trade_fraction
                    if price > 1e-9 and buy_amount_quote > 1e-9:
                        buy_amount_base = buy_amount_quote / price # Estimate base amount
                        if buy_amount_base > 1e-9:
                            if self.execute_trade('buy', buy_amount_base, price, current_timestamp):
                                actual_amount_bought = self.trade_log[-1]['amount_base']
                                action = f"Buy {actual_amount_bought:.6f} {base_asset} @ {price:.4f} (Conf: {confidence:.2f})"
                                self.cooldown_counter = self.cooldown_period
                            else:
                                action = f"Hold (Buy failed for {base_asset})" # execute_trade prints reason
                        else:
                            action = f"Hold (Negligible {base_asset} amount to buy)"
                    else:
                        action = f"Hold (Invalid price {price} or negligible {quote_asset} {buy_amount_quote:.6f})"

                elif confidence < self.hold_band[0]:
                    # Sell logic (sell base asset for quote asset)
                    base_asset = self.pair_to_swap.split('_')[0]
                    quote_asset = self.pair_to_swap.split('_')[1]
                    sell_amount_base = self.portfolio.get(base_asset, 0.0) * self.trade_fraction
                    if sell_amount_base > 1e-9:
                        if self.execute_trade('sell', sell_amount_base, price, current_timestamp):
                            actual_amount_sold = self.trade_log[-1]['amount_base']
                            action = f"Sell {actual_amount_sold:.6f} {base_asset} @ {price:.4f} (Conf: {confidence:.2f})"
                            self.cooldown_counter = self.cooldown_period
                        else:
                            action = f"Hold (Sell failed for {base_asset})" # execute_trade prints reason
                    else:
                        action = f"Hold (Negligible {base_asset} to sell)"
                else:
                    action = f"Hold (Confidence {confidence:.2f} within band {self.hold_band})"

            # Track portfolio value and balances
            portfolio_value = self.calculate_portfolio_value(row)
            history_entry = {
                'timestamp': row['timestamp'], # Keep as Timestamp object for now
                **{f'{asset}_Balance': balance for asset, balance in self.portfolio.items()},
                f'{self.pair_to_swap}_Rate': price, # Use the price variable
                'Portfolio_Value': portfolio_value,
                prediction_col_name: confidence, # Log the specific prediction used
                'Action': action
            }
            portfolio_history.append(history_entry)

        # --- Prepare results and convert timestamps ---
        portfolio_df = pd.DataFrame(portfolio_history)
        trade_log_df = pd.DataFrame(self.trade_log)

        # Save CSV results (with original Timestamps)
        self.save_results(portfolio_df, trade_log_df, cache_location, pair)

        # Prepare summary (converts timestamps within summary)
        summary = self.prepare_summary(df_sim, portfolio_df)

        # Convert timestamps in DataFrames to strings for JSON output
        if not portfolio_df.empty and 'timestamp' in portfolio_df.columns:
            portfolio_df['timestamp'] = portfolio_df['timestamp'].apply(lambda x: x.isoformat() if pd.notna(x) else None)
        if not trade_log_df.empty and 'timestamp' in trade_log_df.columns:
            trade_log_df['timestamp'] = trade_log_df['timestamp'].apply(lambda x: x.isoformat() if pd.notna(x) else None)

        # Convert DataFrames to list of dicts for JSON output
        portfolio_history_list = portfolio_df.replace({np.nan: None}).to_dict('records')
        trade_log_list = trade_log_df.replace({np.nan: None}).to_dict('records')


        return {
            'portfolio_history': portfolio_history_list,
            'trade_log': trade_log_list,
            'summary': summary, # Summary already has string timestamps
        }