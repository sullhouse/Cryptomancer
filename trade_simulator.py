import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from cache_utils import write_csv, save_plot # Import cache utils

# --- Constants for filenames ---
PREDICTION_CACHE_SUBDIR = os.environ.get("GCS_PREDICTION_CACHE_FOLDER", "prediction_cache")
PORTFOLIO_FILENAME = "portfolio_history.csv"
TRADES_FILENAME = "trade_log.csv"
SIMULATION_PLOT_FILENAME = "simulation_plot.png"

def simulate_trades(df_sim, confidence_threshold, hold_band, trade_fraction, fee, cooldown_period, cache_location='local'):
    """
    Simulates trades based on predicted probabilities and saves results.

    Args:
        df_sim (pd.DataFrame): DataFrame with 'timestamp', 'ETH_MATIC_CLOSE', 'Predicted_Prob_Up'.
        confidence_threshold (float): Probability threshold to trigger a buy.
        hold_band (tuple): Probability range (low, high) to hold current position.
        trade_fraction (float): Fraction of available capital to use per trade.
        fee (float): Transaction fee as a fraction (e.g., 0.001 for 0.1%).
        cooldown_period (int): Number of periods to wait after a trade.
        cache_location (str): 'local' or 'gcp'.

    Returns:
        tuple: (portfolio_history_df, trade_log_df, simulation_summary)
               simulation_summary is a dict containing key results.
    """
    print("\n--- Starting Simulation ---")
    print(f"Parameters: Confidence={confidence_threshold}, Hold Band={hold_band}, Trade Fraction={trade_fraction}, Fee={fee}, Cooldown={cooldown_period}")

    # --- Input Validation ---
    required_cols = ['timestamp', 'ETH_MATIC_CLOSE', 'Predicted_Prob_Up']
    if not all(col in df_sim.columns for col in required_cols):
        missing = [col for col in required_cols if col not in df_sim.columns]
        raise ValueError(f"Simulation DataFrame missing required columns: {missing}")
    if df_sim.empty:
        raise ValueError("Simulation DataFrame is empty.")

    # --- Initialization ---
    initial_usdc = df_sim['ETH_MATIC_CLOSE'].iloc[0] # Start with value of 1 ETH in USDC
    eth_balance = 1.0
    usdc_balance = 0.0
    portfolio_history = []
    trade_log = []
    position = 'ETH' # Start holding ETH
    cooldown_counter = 0

    # --- Simulation Loop ---
    for index, row in df_sim.iterrows():
        current_price = row['ETH_MATIC_CLOSE']
        prob_up = row['Predicted_Prob_Up']
        timestamp = row['timestamp']

        # Calculate current portfolio value in USDC
        portfolio_usdc = (eth_balance * current_price) + usdc_balance
        portfolio_history.append({
            'timestamp': timestamp,
            'ETH_Balance': eth_balance,
            'USDC_Balance': usdc_balance,
            'ETH_USDC_Rate': current_price,
            'Portfolio_USDC': portfolio_usdc,
            'Predicted_Prob_Up': prob_up,
            'Action': 'Hold'
        })

        # --- Decision Logic ---
        action = 'Hold'
        if cooldown_counter > 0:
            cooldown_counter -= 1
        elif position == 'USDC' and prob_up >= confidence_threshold:
            # Buy ETH
            buy_amount_usdc = usdc_balance * trade_fraction
            eth_bought = (buy_amount_usdc / current_price) * (1 - fee)
            if eth_bought > 1e-9: # Avoid tiny trades
                eth_balance += eth_bought
                usdc_balance -= buy_amount_usdc
                position = 'ETH'
                action = f'Buy ETH @ {current_price:.4f}'
                cooldown_counter = cooldown_period
                trade_log.append({'timestamp': timestamp, 'action': action, 'price': current_price, 'amount_eth': eth_bought, 'amount_usdc': buy_amount_usdc, 'fee_paid': buy_amount_usdc / current_price * fee * current_price})
        elif position == 'ETH' and prob_up < hold_band[0]: # Sell if below lower hold band
             # Sell ETH
             sell_amount_eth = eth_balance * trade_fraction
             usdc_received = (sell_amount_eth * current_price) * (1 - fee)
             if usdc_received > 1e-9: # Avoid tiny trades
                 usdc_balance += usdc_received
                 eth_balance -= sell_amount_eth
                 position = 'USDC'
                 action = f'Sell ETH @ {current_price:.4f}'
                 cooldown_counter = cooldown_period
                 trade_log.append({'timestamp': timestamp, 'action': action, 'price': current_price, 'amount_eth': sell_amount_eth, 'amount_usdc': usdc_received, 'fee_paid': sell_amount_eth * current_price * fee})
        # Update action in history if a trade occurred
        if action != 'Hold':
            portfolio_history[-1]['Action'] = action


    # --- Post-Simulation Analysis ---
    portfolio_df = pd.DataFrame(portfolio_history)
    trade_df = pd.DataFrame(trade_log)

    final_portfolio_value = portfolio_df['Portfolio_USDC'].iloc[-1]
    total_return_pct = (final_portfolio_value / initial_usdc - 1) * 100
    total_fees_paid = trade_df['fee_paid'].sum() if not trade_df.empty else 0

    # Benchmark: Buy and Hold 1 ETH
    benchmark_final_value = df_sim['ETH_MATIC_CLOSE'].iloc[-1] * 1 # Started with 1 ETH
    benchmark_return_pct = (benchmark_final_value / initial_usdc - 1) * 100
    strategy_vs_benchmark = total_return_pct - benchmark_return_pct

    # --- Prepare Summary ---
    simulation_summary = {
        "initial_portfolio_value_usdc": initial_usdc,
        "final_portfolio_value_usdc": final_portfolio_value,
        "total_return_pct": total_return_pct,
        "benchmark_initial_value_usdc": initial_usdc,
        "benchmark_final_value_usdc": benchmark_final_value,
        "benchmark_return_pct": benchmark_return_pct,
        "strategy_vs_benchmark_pct_points": strategy_vs_benchmark,
        "number_of_trades": len(trade_df),
        "total_fees_paid_usdc": total_fees_paid
    }

    print("\n--- Simulation Results ---")
    for key, value in simulation_summary.items():
        print(f"{key.replace('_', ' ').title()}: {value:.4f}" if isinstance(value, (int, float)) else f"{key.replace('_', ' ').title()}: {value}")


    # --- Plotting ---
    try:
        fig_sim, ax_sim = plt.subplots(figsize=(14, 7))
        ax_sim.plot(portfolio_df['timestamp'], portfolio_df['Portfolio_USDC'], label='Strategy Value (USDC)', color='blue')
        # Add benchmark plot
        benchmark_value = df_sim['ETH_MATIC_CLOSE'] * 1 # Value of holding 1 ETH
        ax_sim.plot(df_sim['timestamp'], benchmark_value, label='Benchmark (Hold 1 ETH)', color='orange', linestyle='--')

        # Mark trades
        buys = trade_df[trade_df['action'].str.startswith('Buy')]
        sells = trade_df[trade_df['action'].str.startswith('Sell')]
        ax_sim.plot(buys['timestamp'], portfolio_df.loc[buys.index, 'Portfolio_USDC'], '^', markersize=8, color='green', label='Buy ETH', alpha=0.7)
        ax_sim.plot(sells['timestamp'], portfolio_df.loc[sells.index, 'Portfolio_USDC'], 'v', markersize=8, color='red', label='Sell ETH', alpha=0.7)

        ax_sim.set_title('Portfolio Value vs. Benchmark')
        ax_sim.set_xlabel('Timestamp')
        ax_sim.set_ylabel('Portfolio Value (USDC)')
        ax_sim.legend()
        ax_sim.grid(True)
        fig_sim.tight_layout()
        save_plot(fig_sim, SIMULATION_PLOT_FILENAME, cache_location)
        simulation_summary["plot_path"] = f"{cache_location}/{SIMULATION_PLOT_FILENAME}"
    except Exception as e:
        print(f"Error generating simulation plot: {e}")
        simulation_summary["plot_path"] = None


    # --- Save Results ---
    write_csv(portfolio_df, PORTFOLIO_FILENAME, cache_location)
    write_csv(trade_df, TRADES_FILENAME, cache_location)

    print(f"Portfolio history and trade log saved to '{PREDICTION_CACHE_SUBDIR}' subdirectory within {cache_location}.")

    return portfolio_df, trade_df, simulation_summary
