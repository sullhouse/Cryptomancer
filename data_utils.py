import pandas as pd
import numpy as np
import os
from google.cloud import bigquery
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
INTERVAL_HOURS = 1

# --- Data Fetching ---
def fetch_data(table_name, start_date, end_date, project_id=PROJECT_ID, interval_hours=1):
    """Fetches data from BigQuery for a specific date range, then resamples."""
    # ... (existing fetch_data code remains the same) ...
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
        df = client.query(query).to_dataframe()
    finally:
        client.close()

    if df.empty:
        print(f"Warning: No data fetched for {table_name} between {start_date} and {end_date}.")
        # Return an empty DataFrame with expected columns to avoid merge errors later
        return pd.DataFrame(columns=['timestamp', 'OPEN', 'HIGH', 'LOW', 'CLOSE', 'VOLUME', 'QUOTE_VOLUME', 'VOLATILITY', 'SPREAD_CLOSE_OPEN'])


    # Convert timestamp to datetime objects if not already
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Resampling using interval_hours
    df_indexed = df.set_index("timestamp")
    df_resampled = df_indexed.resample(f"{interval_hours}h").agg({
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

def fetch_and_save_minute_data_for_hour(token_pairs, hour_timestamp, cache_folder="cache"):
    """
    Fetches minute-level data for the specified hour for the given token pairs,
    merges them on timestamp, and saves as {hour-timestamp}_minute_coindesk_data.csv in the cache folder.

    Args:
        token_pairs (list of str): List of pair names (e.g., ["eth_usdc", "matic_usdc"]).
        hour_timestamp (str): Hour in 'YYYY-MM-DD HH:00:00' format (UTC).
        cache_folder (str): Folder to save the CSV file.
    """
    import os
    from datetime import datetime, timedelta

    # Parse the hour timestamp
    hour_dt = pd.to_datetime(hour_timestamp)
    start_time = hour_dt
    end_time = hour_dt + timedelta(hours=1) - timedelta(minutes=1)

    dfs = {}
    for pair in token_pairs:
        table_name = TABLES.get(pair.lower())
        if not table_name:
            print(f"Pair {pair} not found in TABLES. Skipping.")
            continue

        # Query for minute-level data for this hour
        query = f"""
            SELECT
                TIMESTAMP AS timestamp,
                OPEN, HIGH, LOW, CLOSE,
                VOLUME, QUOTE_VOLUME, VOLATILITY, SPREAD_CLOSE_OPEN
            FROM `{PROJECT_ID}.{table_name}`
            WHERE TIMESTAMP >= TIMESTAMP('{start_time.strftime('%Y-%m-%d %H:%M:%S')}')
              AND TIMESTAMP <= TIMESTAMP('{end_time.strftime('%Y-%m-%d %H:%M:%S')}')
            ORDER BY timestamp
        """

        client = bigquery.Client(project=PROJECT_ID)
        try:
            df = client.query(query).to_dataframe()
        finally:
            client.close()

        if df.empty:
            print(f"Warning: No minute data for {pair} in hour {hour_timestamp}.")
            continue

        # Rename columns to include pair prefix
        rename_map = {
            "OPEN": f"{pair.upper()}_OPEN", "HIGH": f"{pair.upper()}_HIGH",
            "LOW": f"{pair.upper()}_LOW", "CLOSE": f"{pair.upper()}_CLOSE",
            "VOLUME": f"{pair.upper()}_VOLUME", "QUOTE_VOLUME": f"{pair.upper()}_QUOTE_VOLUME",
            "VOLATILITY": f"{pair.upper()}_VOLATILITY", "SPREAD_CLOSE_OPEN": f"{pair.upper()}_SPREAD_CLOSE_OPEN"
        }
        df = df.rename(columns=rename_map)
        dfs[pair] = df

    # Merge all DataFrames on timestamp (outer join to keep all minutes)
    merged_df = None
    for df in dfs.values():
        if merged_df is None:
            merged_df = df
        else:
            merged_df = pd.merge(merged_df, df, on="timestamp", how="outer")

    if merged_df is None or merged_df.empty:
        print("No data to save for the specified hour and pairs.")
        return

    # Sort by timestamp and keep only the 60 minutes for the hour
    merged_df['timestamp'] = pd.to_datetime(merged_df['timestamp'], utc=True)
    # Ensure start_time and end_time are also timezone-aware
    if start_time.tzinfo is None:
        start_time = start_time.replace(tzinfo=pd.Timestamp.utcnow().tzinfo or pd.Timestamp('UTC').tzinfo)
    if end_time.tzinfo is None:
        end_time = end_time.replace(tzinfo=pd.Timestamp.utcnow().tzinfo or pd.Timestamp('UTC').tzinfo)

    merged_df = merged_df.sort_values('timestamp')
    merged_df = merged_df[(merged_df['timestamp'] >= start_time) & (merged_df['timestamp'] <= end_time)]
    merged_df = merged_df.reset_index(drop=True)

    # Output to CSV
    os.makedirs(cache_folder, exist_ok=True)
    out_filename = f"{hour_dt.strftime('%Y-%m-%dT%H')}_minute_coindesk_data.csv"
    out_path = os.path.join(cache_folder, out_filename)
    merged_df.to_csv(out_path, index=False)
    print(f"Minute-level data saved to {out_path}")

def api_fetch_and_save_minute_data_for_hour(request):
    """
    API endpoint: Fetches minute-level data for the specified hour and token pairs,
    merges them on timestamp, and saves as a CSV in the cache folder.

    Expects a POST request with JSON:
    {
        "token_pairs": ["eth_usdc", "matic_usdc"],
        "hour_timestamp": "2025-04-20 18:00:00",  # UTC hour start
        "filename": "custom_filename.csv"         # Optional, defaults to {hour-timestamp}_minute_coindesk_data.csv
    }
    """
    import os
    from datetime import timedelta

    try:
        data = request.get_json()
        token_pairs = data.get("token_pairs")
        hour_timestamp = data.get("hour_timestamp")
        filename = data.get("filename")

        if not token_pairs or not hour_timestamp:
            return {"error": "token_pairs and hour_timestamp are required."}, 400

        hour_dt = pd.to_datetime(hour_timestamp)
        start_time = hour_dt
        end_time = hour_dt + timedelta(hours=1) - timedelta(minutes=1)

        dfs = {}
        for pair in token_pairs:
            table_name = TABLES.get(pair.lower())
            if not table_name:
                print(f"Pair {pair} not found in TABLES. Skipping.")
                continue

            query = f"""
                SELECT
                    TIMESTAMP AS timestamp,
                    OPEN, HIGH, LOW, CLOSE,
                    VOLUME, QUOTE_VOLUME, VOLATILITY, SPREAD_CLOSE_OPEN
                FROM `{PROJECT_ID}.{table_name}`
                WHERE TIMESTAMP >= TIMESTAMP('{start_time.strftime('%Y-%m-%d %H:%M:%S')}')
                  AND TIMESTAMP <= TIMESTAMP('{end_time.strftime('%Y-%m-%d %H:%M:%S')}')
                ORDER BY timestamp
            """

            client = bigquery.Client(project=PROJECT_ID)
            try:
                df = client.query(query).to_dataframe()
            finally:
                client.close()

            if df.empty:
                print(f"Warning: No minute data for {pair} in hour {hour_timestamp}.")
                continue

            rename_map = {
                "OPEN": f"{pair.upper()}_OPEN", "HIGH": f"{pair.upper()}_HIGH",
                "LOW": f"{pair.upper()}_LOW", "CLOSE": f"{pair.upper()}_CLOSE",
                "VOLUME": f"{pair.upper()}_VOLUME", "QUOTE_VOLUME": f"{pair.upper()}_QUOTE_VOLUME",
                "VOLATILITY": f"{pair.upper()}_VOLATILITY", "SPREAD_CLOSE_OPEN": f"{pair.upper()}_SPREAD_CLOSE_OPEN"
            }
            df = df.rename(columns=rename_map)
            dfs[pair] = df

        merged_df = None
        for df in dfs.values():
            if merged_df is None:
                merged_df = df
            else:
                merged_df = pd.merge(merged_df, df, on="timestamp", how="outer")

        if merged_df is None or merged_df.empty:
            return {"error": "No data to save for the specified hour and pairs."}, 200

        merged_df['timestamp'] = pd.to_datetime(merged_df['timestamp'], utc=True)
        # Ensure start_time and end_time are also timezone-aware
        if start_time.tzinfo is None:
            start_time = start_time.replace(tzinfo=pd.Timestamp.utcnow().tzinfo or pd.Timestamp('UTC').tzinfo)
        if end_time.tzinfo is None:
            end_time = end_time.replace(tzinfo=pd.Timestamp.utcnow().tzinfo or pd.Timestamp('UTC').tzinfo)

        merged_df = merged_df.sort_values('timestamp')
        merged_df = merged_df[(merged_df['timestamp'] >= start_time) & (merged_df['timestamp'] <= end_time)]
        merged_df = merged_df.reset_index(drop=True)

        os.makedirs("cache", exist_ok=True)
        if not filename:
            filename = f"{hour_dt.strftime('%Y-%m-%dT%H')}_minute_coindesk_data.csv"
        out_path = os.path.join("cache", filename)
        merged_df.to_csv(out_path, index=False)
        print(f"Minute-level data saved to {out_path}")

        return {"message": f"Minute-level data saved to {out_path}", "rows": len(merged_df)}, 200

    except Exception as e:
        print(f"Error in api_fetch_and_save_minute_data_for_hour: {e}")
        return {"error": str(e)}, 500

# --- Indicator Calculations ---
def calculate_rsi(series, length=14):
    """Calculates Relative Strength Index (RSI)"""
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=length).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=length).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(series, fast=12, slow=26, signal=9):
    """Calculates MACD, Signal Line, and Histogram"""
    exp1 = series.ewm(span=fast, adjust=False).mean()
    exp2 = series.ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    histogram = macd - signal_line
    return macd, signal_line, histogram

def calculate_bbands(series, length=20, std=2):
    """Calculates Bollinger Bands (Middle, Upper, Lower), Bandwidth, and %B"""
    middle_band = series.rolling(window=length).mean()
    std_dev = series.rolling(window=length).std()
    upper_band = middle_band + (std_dev * std)
    lower_band = middle_band - (std_dev * std)
    bandwidth = (upper_band - lower_band) / middle_band
    percent_b = (series - lower_band) / (upper_band - lower_band)
    return lower_band, middle_band, upper_band, bandwidth, percent_b

# --- Feature Engineering ---
def build_features(df, target_pair): # Add target_pair parameter
    """
    Adds features (indicators, time, interactions) to the merged DataFrame.
    Generates the target variable ONLY for the specified target_pair.
    Uses pd.concat for better performance when adding many columns.
    """
    print(f"Starting feature engineering (Targeting: {target_pair})...")
    df_original = df # Keep original for concatenation later

    # --- Target Definition (Specific Pair) ---
    target_base, target_quote = target_pair.upper().split('_')
    close_col = f"{target_base}_{target_quote}_CLOSE"
    future_col = f"FUTURE_{target_base}_{target_quote}_CLOSE"
    target_col = f"TARGET_{target_base}_{target_quote}_UP"

    if close_col not in df.columns:
        raise ValueError(f"Required close column '{close_col}' for target pair '{target_pair}' not found in merged data.")

    # Calculate target separately first
    future_close = df[close_col].shift(-1)
    target_series = (future_close > df[close_col]).astype(int)
    target_series.name = target_col # Name the series for concatenation

    # --- Feature Calculations (Using ALL available pairs) ---
    all_pairs_in_df = list(set([col.split('_')[0] + '_' + col.split('_')[1] for col in df.columns if '_' in col and col.endswith('_CLOSE')]))
    print(f"Calculating features based on available pairs: {all_pairs_in_df}")

    new_features_dict = {} # Dictionary to hold new feature Series

    for pair_str in all_pairs_in_df:
        close_col_feat = f"{pair_str}_CLOSE"
        if close_col_feat in df.columns:
            # --- Basic Features ---
            pct_change = df[close_col_feat].pct_change()
            new_features_dict[f"{pair_str}_PCT_CHANGE"] = pct_change

            # --- Indicators ---
            new_features_dict[f'{pair_str}_RSI_14'] = calculate_rsi(df[close_col_feat], length=14)
            macd, signal, hist = calculate_macd(df[close_col_feat], fast=12, slow=26, signal=9)
            new_features_dict[f'{pair_str}_MACD_12_26_9'] = macd
            new_features_dict[f'{pair_str}_MACDS_12_26_9'] = signal
            new_features_dict[f'{pair_str}_MACDH_12_26_9'] = hist
            bbl, bbm, bbu, bbb, bbp = calculate_bbands(df[close_col_feat], length=20, std=2)
            new_features_dict[f'{pair_str}_BBL_20_2.0'] = bbl
            new_features_dict[f'{pair_str}_BBM_20_2.0'] = bbm
            new_features_dict[f'{pair_str}_BBU_20_2.0'] = bbu
            new_features_dict[f'{pair_str}_BBB_20_2.0'] = bbb
            new_features_dict[f'{pair_str}_BBP_20_2.0'] = bbp

            # --- Lags/Rolling on PCT_CHANGE ---
            change_col = f"{pair_str}_PCT_CHANGE" # Use the calculated pct_change Series
            if pct_change is not None: # Check if pct_change was calculated
                 lag_periods = [1, 2, 3, 5]
                 rolling_windows = [3, 5, 10]
                 for lag in lag_periods:
                     new_features_dict[f"{change_col}_lag{lag}"] = pct_change.shift(lag)
                 for window in rolling_windows:
                     new_features_dict[f"{change_col}_ma{window}"] = pct_change.rolling(window=window).mean()
                     new_features_dict[f"{change_col}_std{window}"] = pct_change.rolling(window=window).std()
        else:
            print(f"Skipping feature calculation for {pair_str} as {close_col_feat} not found.")

    # --- Add Time-Based Features ---
    if 'timestamp' in df.columns:
        print("Adding time-based features...")
        new_features_dict['time_hour'] = df['timestamp'].dt.hour
        new_features_dict['time_dayofweek'] = df['timestamp'].dt.dayofweek
    else:
        print("Warning: 'timestamp' column not found for time-based features.")

    # --- Add Interaction Features (Example) ---
    # Calculate interactions based on features already in the dictionary
    print("Adding interaction features...")
    rsi_eth_matic_key = 'ETH_MATIC_RSI_14'
    rsi_btc_usdc_key = 'BTC_USDC_RSI_14'
    if rsi_eth_matic_key in new_features_dict and rsi_btc_usdc_key in new_features_dict:
        new_features_dict['RSI_DIFF_ETHMATIC_BTCUSDC'] = new_features_dict[rsi_eth_matic_key] - new_features_dict[rsi_btc_usdc_key]
    # Add more interactions as needed

    # --- Concatenate all new features ---
    print(f"Concatenating {len(new_features_dict)} new features...")
    new_features_df = pd.DataFrame(new_features_dict) # Create DataFrame from dictionary of Series

    # Concatenate original df, the target series, and the new features df
    df_combined = pd.concat([df_original, target_series, new_features_df], axis=1)

    # --- Cleanup ---
    # No need to drop future_cols anymore as it wasn't added to df_combined
    print("De-fragmenting DataFrame after concat...")
    df_final = df_combined.copy() # Create a de-fragmented copy

    print("Dropping rows with NaN values...")
    df_final = df_final.dropna() # Drop NaNs created by indicators, lags, rolling windows, target shift etc.

    print(f"Dataset shape after feature engineering for {target_pair} and NaN drop: {df_final.shape}")
    if df_final.empty:
        raise ValueError(f"DataFrame is empty after feature engineering for {target_pair} and NaN drop.")

    # --- Numeric Conversion Check ---
    cols_to_check = [col for col in df_final.columns if col != 'timestamp']
    numeric_cols = df_final[cols_to_check].select_dtypes(include=np.number).columns.tolist()
    if len(numeric_cols) != len(cols_to_check):
        non_numeric = [col for col in cols_to_check if col not in numeric_cols]
        print(f"Warning: Non-numeric columns found: {non_numeric}. Attempting conversion.")
        for col in non_numeric:
            df_final[col] = pd.to_numeric(df_final[col], errors='coerce')
        df_final = df_final.dropna() # Drop NaNs introduced by coercion
        print(f"Dataset shape after numeric conversion and NaN drop: {df_final.shape}")
        if df_final.empty:
            raise ValueError("DataFrame is empty after numeric conversion.")

    return df_final

# --- build_dataset function remains the same ---
def build_dataset(target_pair, start_date, end_date, interval_hours=3): # Add target_pair
    """
    Fetches data for ALL pairs, merges, and builds features targeting the specific target_pair.
    """
    print(f"Building dataset from {start_date} to {end_date} (Interval: {interval_hours}h) - Targeting: {target_pair}")
    dfs = {}
    # Fetch data for ALL pairs defined in TABLES
    for name, table_id in TABLES.items():
        # Rename map logic remains the same
        rename_map = {
            "OPEN": f"{name.upper()}_OPEN", "HIGH": f"{name.upper()}_HIGH",
            "LOW": f"{name.upper()}_LOW", "CLOSE": f"{name.upper()}_CLOSE",
            "VOLUME": f"{name.upper()}_VOLUME", "QUOTE_VOLUME": f"{name.upper()}_QUOTE_VOLUME",
            "VOLATILITY": f"{name.upper()}_VOLATILITY", "SPREAD_CLOSE_OPEN": f"{name.upper()}_SPREAD_CLOSE_OPEN"
        }
        dfs[name] = fetch_data(
            table_id,
            start_date=start_date,
            end_date=end_date,
            interval_hours=interval_hours
        ).rename(columns=rename_map)

    # Merge DataFrames iteratively
    merged_df = None
    # Merge logic remains the same
    for name, df_pair in dfs.items():
        if df_pair.empty:
            print(f"Warning: DataFrame for {name} is empty for the period {start_date} to {end_date}. Skipping merge.")
            continue
        if 'timestamp' not in df_pair.columns:
             print(f"Warning: Timestamp column missing in fetched data for {name}. Skipping merge.")
             continue # Skip if timestamp is missing

        if merged_df is None:
            merged_df = df_pair
        else:
            # Ensure the current df_pair also has a timestamp before merging
            if 'timestamp' in merged_df.columns:
                 # Use outer merge initially to keep all timestamps, then handle NaNs in build_features
                 merged_df = pd.merge(merged_df, df_pair, on="timestamp", how="inner") # Keep inner merge for now
            else:
                 print(f"Error: merged_df lost timestamp column before merging {name}.")
                 raise ValueError("Timestamp column lost during merge process.")


    if merged_df is None or merged_df.empty:
        raise ValueError(f"Data merging resulted in an empty DataFrame for the period {start_date} to {end_date}.")

    print(f"Shape after merging all pairs: {merged_df.shape}")

    # Build features on the merged DataFrame, passing the specific target_pair
    final_df = build_features(merged_df, target_pair=target_pair)

    return final_df
