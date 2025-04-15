import pandas as pd
import numpy as np
from google.cloud import bigquery
import os
from datetime import datetime # Import datetime

# --- Constants ---
PROJECT_ID = os.environ.get("GCP_PROJECT_ID")
TABLES = {
    "eth_matic": "Coindesk.eth-matic",
    "eth_btc": "Coindesk.eth-btc",
    "btc_usdc": "Coindesk.btc-usdc",
    "matic_usdc": "Coindesk.matic-usdc",
    "btc_eth": "Coindesk.btc-eth",
    "eth_usdc": "Coindesk.eth-usdc"
}
INTERVAL_HOURS = 3

# --- Data Fetching ---
def fetch_data(table_name, start_date, end_date, project_id=PROJECT_ID):
    """Fetches data from BigQuery for a specific date range, then resamples."""

    # Validate date format (optional but recommended)
    try:
        datetime.strptime(start_date, '%Y-%m-%d')
        datetime.strptime(end_date, '%Y-%m-%d')
    except ValueError:
        raise ValueError("Incorrect date format, should be YYYY-MM-DD")

    query = f"""
        SELECT
            TIMESTAMP AS timestamp,
            OPEN, HIGH, LOW, CLOSE,
            VOLUME, QUOTE_VOLUME, VOLATILITY, SPREAD_CLOSE_OPEN
        FROM `{project_id}.{table_name}`
        WHERE DATE(TIMESTAMP) BETWEEN DATE("{start_date}") AND DATE("{end_date}") -- Use DATE() for comparison
        ORDER BY timestamp
    """

    client = bigquery.Client(project=project_id)
    try:
        print(f"Fetching data for {table_name} ({start_date} to {end_date}) from BigQuery...")
        df = client.query(query).to_dataframe()
        if df.empty:
                print(f"Warning: No data returned from BigQuery for {table_name} in range {start_date} to {end_date}.")
                # Return an empty DataFrame with expected columns if no data found
                return pd.DataFrame(columns=['timestamp', 'OPEN', 'HIGH', 'LOW', 'CLOSE', 'VOLUME', 'QUOTE_VOLUME', 'VOLATILITY', 'SPREAD_CLOSE_OPEN'])

        # Ensure timestamp is timezone-aware after fetching
        if df['timestamp'].dt.tz is None:
                df['timestamp'] = df['timestamp'].dt.tz_localize('UTC') # Assuming UTC, adjust if needed
        else:
                df['timestamp'] = df['timestamp'].dt.tz_convert('UTC') # Convert to UTC if already timezone-aware
    finally:
        client.close()

    if df.empty:
        print(f"No data to resample for {table_name} in range {start_date} to {end_date}.")
        return df # Return empty df

    # Resampling
    df_indexed = df.set_index("timestamp")
    df_resampled = df_indexed.resample(f"{INTERVAL_HOURS}h").agg({
        'OPEN': 'first',
        'HIGH': 'max',
        'LOW': 'min',
        'CLOSE': 'last',
        'VOLUME': 'sum',
        'QUOTE_VOLUME': 'sum',
        'VOLATILITY': 'mean',
        'SPREAD_CLOSE_OPEN': 'mean'
    }).dropna().reset_index() # Keep timestamp as a column after resampling

    return df_resampled

# --- Indicator Calculations ---
def calculate_rsi(series, length=14):
    """Calculates Relative Strength Index (RSI)"""
    delta = series.diff()
    gain = delta.where(delta > 0, 0).fillna(0)
    loss = -delta.where(delta < 0, 0).fillna(0)
    avg_gain = gain.ewm(com=length - 1, min_periods=length).mean()
    avg_loss = loss.ewm(com=length - 1, min_periods=length).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(series, fast=12, slow=26, signal=9):
    """Calculates Moving Average Convergence Divergence (MACD)"""
    exp1 = series.ewm(span=fast, adjust=False).mean()
    exp2 = series.ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    histogram = macd - signal_line
    return macd, signal_line, histogram

def calculate_bbands(series, length=20, std=2):
    """Calculates Bollinger Bands"""
    middle_band = series.rolling(window=length).mean()
    std_dev = series.rolling(window=length).std()
    upper_band = middle_band + (std_dev * std)
    lower_band = middle_band - (std_dev * std)
    bandwidth = ((upper_band - lower_band) / middle_band.replace(0, np.nan)).fillna(0) # Avoid division by zero
    percent_b = ((series - lower_band) / (upper_band - lower_band).replace(0, np.nan)).fillna(0.5) # Avoid division by zero, default to 0.5
    return lower_band, middle_band, upper_band, bandwidth, percent_b

# --- Feature Engineering ---
def build_features(df):
    """Adds features (indicators, time, interactions) to the merged DataFrame."""
    print("Starting feature engineering...")

    # --- Target Definition ---
    if "ETH_MATIC_CLOSE" not in df.columns:
        raise ValueError("ETH_MATIC_CLOSE column required for target definition is missing.")
    df["FUTURE_ETH_MATIC"] = df["ETH_MATIC_CLOSE"].shift(-1)
    df["TARGET"] = (df["FUTURE_ETH_MATIC"] > df["ETH_MATIC_CLOSE"]).astype(int)

    # --- Price Percentage Changes & Basic Features ---
    pairs = ["ETH_MATIC", "ETH_BTC", "BTC_USDC", "MATIC_USDC", "BTC_ETH", "ETH_USDC"]
    for pair in pairs:
        close_col = f"{pair}_CLOSE"
        if close_col in df.columns:
            df[f"{pair}_PCT_CHANGE"] = df[close_col].pct_change()
        else:
            print(f"Warning: {close_col} not found for PCT_CHANGE calculation.")

    change_cols = [col for col in df.columns if "_PCT_CHANGE" in col]
    lag_periods = [1, 2, 3, 5]
    rolling_windows = [3, 5, 10]

    for col in change_cols:
        for lag in lag_periods:
            df[f"{col}_lag{lag}"] = df[col].shift(lag)
        for window in rolling_windows:
            df[f"{col}_ma{window}"] = df[col].rolling(window=window).mean()
            df[f"{col}_std{window}"] = df[col].rolling(window=window).std()

    # --- Add Technical Indicators Manually ---
    print("Adding technical indicators...")
    indicator_pairs = ["ETH_MATIC", "ETH_USDC", "MATIC_USDC", "BTC_USDC"]
    for pair in indicator_pairs:
        close_col = f"{pair}_CLOSE"
        if close_col in df.columns:
            df[f'{pair}_RSI_14'] = calculate_rsi(df[close_col], length=14)
            macd, signal, hist = calculate_macd(df[close_col], fast=12, slow=26, signal=9)
            df[f'{pair}_MACD_12_26_9'] = macd
            df[f'{pair}_MACDS_12_26_9'] = signal
            df[f'{pair}_MACDH_12_26_9'] = hist
            bbl, bbm, bbu, bbb, bbp = calculate_bbands(df[close_col], length=20, std=2)
            df[f'{pair}_BBL_20_2.0'] = bbl
            df[f'{pair}_BBM_20_2.0'] = bbm
            df[f'{pair}_BBU_20_2.0'] = bbu
            df[f'{pair}_BBB_20_2.0'] = bbb
            df[f'{pair}_BBP_20_2.0'] = bbp
        else:
            print(f"Warning: Close column {close_col} not found for indicators.")

    # --- Add Time-Based Features ---
    if 'timestamp' in df.columns:
        print("Adding time-based features...")
        df['time_hour'] = df['timestamp'].dt.hour
        df['time_dayofweek'] = df['timestamp'].dt.dayofweek
    else:
        print("Warning: 'timestamp' column not found for time-based features.")


    # --- Add Interaction Features ---
    print("Adding interaction features...")
    interaction_pairs = ["ETH_MATIC", "BTC_USDC"]
    for pair in interaction_pairs:
        vol_col = f"{pair}_VOLUME"
        volatility_col = f"{pair}_VOLATILITY"
        close_col = f"{pair}_CLOSE"
        bbm_col = f"{pair}_BBM_20_2.0"

        if vol_col in df.columns and volatility_col in df.columns:
            df[f'{pair}_VOL_X_VOLATILITY'] = df[vol_col] * df[volatility_col]
        if close_col in df.columns and bbm_col in df.columns:
            df[f'{pair}_CLOSE_DIV_BBM'] = (df[close_col] / df[bbm_col].replace(0, np.nan)).fillna(1)

    if 'ETH_MATIC_RSI_14' in df.columns and 'BTC_USDC_RSI_14' in df.columns:
        df['RSI_DIFF_ETHMATIC_BTCUSDC'] = df['ETH_MATIC_RSI_14'] - df['BTC_USDC_RSI_14']

    # --- Cleanup ---
    df = df.drop(columns=["FUTURE_ETH_MATIC"], errors='ignore') # Ignore error if already dropped

    # --- Add this line to de-fragment the DataFrame ---
    print("De-fragmenting DataFrame before final NaN drop...")
    df = df.copy()
    # --- End Add this line ---

    # De-fragment and handle NaNs (dropna was already here)
    # df = df.copy() # Remove this line if it was added previously here
    df = df.dropna() # Drop NaNs created by indicators and other features

    print(f"Dataset shape after feature engineering and NaN drop: {df.shape}")
    if df.empty:
        raise ValueError("DataFrame is empty after feature engineering and NaN drop.")

    # Ensure all columns are numeric for the model (except timestamp if kept)
    cols_to_check = [col for col in df.columns if col != 'timestamp']
    numeric_cols = df[cols_to_check].select_dtypes(include=np.number).columns.tolist()
    if len(numeric_cols) != len(cols_to_check):
        non_numeric = [col for col in cols_to_check if col not in numeric_cols]
        print(f"Warning: Non-numeric columns found: {non_numeric}. Attempting conversion.")
        for col in non_numeric:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df = df.dropna()
        print(f"Dataset shape after numeric conversion and NaN drop: {df.shape}")
        if df.empty:
            raise ValueError("DataFrame is empty after numeric conversion.")

    return df

def build_dataset(start_date, end_date):
    """Fetches data for all pairs for a given date range, merges, and builds features."""
    print(f"Building dataset from {start_date} to {end_date}...")
    # Fetch all pairs for the specified date range
    dfs = {}
    for name, table_id in TABLES.items():
        # Define column renames based on the pair name
        rename_map = {
            "OPEN": f"{name.upper()}_OPEN", "HIGH": f"{name.upper()}_HIGH",
            "LOW": f"{name.upper()}_LOW", "CLOSE": f"{name.upper()}_CLOSE",
            "VOLUME": f"{name.upper()}_VOLUME", "QUOTE_VOLUME": f"{name.upper()}_QUOTE_VOLUME",
            "VOLATILITY": f"{name.upper()}_VOLATILITY", "SPREAD_CLOSE_OPEN": f"{name.upper()}_SPREAD_CLOSE_OPEN"
        }
        # Pass start_date and end_date to fetch_data
        dfs[name] = fetch_data(
            table_id,
            start_date=start_date,
            end_date=end_date,
        ).rename(columns=rename_map)

    # Merge DataFrames iteratively
    merged_df = None
    for name, df_pair in dfs.items():
        if df_pair.empty:
            print(f"Warning: DataFrame for {name} is empty for the period {start_date} to {end_date}. Skipping merge.")
            continue # Skip empty dataframes

        if merged_df is None:
            merged_df = df_pair
        else:
            # Ensure timestamp columns are compatible before merge
            if 'timestamp' not in merged_df.columns or 'timestamp' not in df_pair.columns:
                 raise ValueError(f"Timestamp column missing in DataFrame for merging: {name}")
            merged_df = pd.merge(merged_df, df_pair, on="timestamp", how="inner") # Use inner merge to keep only common timestamps

    if merged_df is None or merged_df.empty:
        # Raise error only if ALL dataframes were empty or merge failed completely
        all_empty = all(df.empty for df in dfs.values())
        if all_empty:
             raise ValueError(f"All data sources returned empty DataFrames for the period {start_date} to {end_date}.")
        else:
             raise ValueError(f"Data merging resulted in an empty DataFrame for the period {start_date} to {end_date}. Check for timestamp mismatches or gaps.")


    print(f"Shape after merging: {merged_df.shape}")

    # Build features on the merged DataFrame
    final_df = build_features(merged_df)

    return final_df
