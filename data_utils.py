import pandas as pd
import numpy as np
import os
from google.cloud import bigquery
from datetime import datetime # Import datetime
import requests

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

from cache_utils import write_csv
import os

def api_fetch_and_save_minute_data_for_hour(request):
    """
    API endpoint: Fetches minute-level data for the specified hour and token pairs,
    merges them on timestamp, and saves as a CSV in the cache folder or GCP bucket.
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

            from google.cloud import bigquery
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
        start_time = pd.to_datetime(start_time).tz_localize('UTC') if pd.to_datetime(start_time).tzinfo is None else pd.to_datetime(start_time)
        end_time = pd.to_datetime(end_time).tz_localize('UTC') if pd.to_datetime(end_time).tzinfo is None else pd.to_datetime(end_time)

        merged_df = merged_df.sort_values('timestamp')
        merged_df = merged_df[(merged_df['timestamp'] >= start_time) & (merged_df['timestamp'] <= end_time)]
        merged_df = merged_df.reset_index(drop=True)

        # --- Determine environment and output path ---
        RUNNING_LOCALLY = not os.environ.get('K_SERVICE', '')
        cache_location = 'local' if RUNNING_LOCALLY else 'gcp'

        # Set default filename if not provided
        if not filename:
            filename = f"{hour_dt.strftime('%Y-%m-%dT%H')}_minute_coindesk_data.csv"

        # For GCP, prefix with data_exports/
        if cache_location == 'gcp':
            filename = f"data_exports/{filename}"

        # Use cache_utils.write_csv to save
        write_csv(merged_df, filename, cache_location)

        print(f"Minute-level data saved to {filename} ({cache_location})")
        return {"message": f"Minute-level data saved to {filename}", "rows": len(merged_df)}, 200

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

import openai
from google.cloud import bigquery
import json

def run_llm_and_save_to_bigquery(request):
    """
    API endpoint: 
    - Reads the LLM primer from GCS.
    - Reads a CSV file from GCS (filename provided in JSON).
    - Sends both as context to OpenAI or Gemini, requesting a JSON response in a specific format.
    - Saves the resulting JSON to BigQuery in the AI dataset, table 'llm_predictions'.
    - Logs the run in AI.run_log.
    - Notifies Discord on success or error.

    Expects POST JSON:
    {
        "csv_filename": "data_exports/2025-04-20T18_minute_coindesk_data.csv",
        "llm": "openai" or "gemini",           # optional, default "openai"
        "model": "gpt-3.5-turbo" or "2.5",     # optional, default per llm
    }
    """
    import os
    import re
    import json
    from google.cloud import storage
    from datetime import datetime

    try:
        data = request.get_json()
        csv_filename = data.get("csv_filename")
        llm = data.get("llm", "openai").lower()
        model = data.get("model")
        if not csv_filename:
            return {"error": "csv_filename is required"}, 400

        # Set defaults
        if llm == "openai":
            model = model or "gpt-3.5-turbo"
        elif llm == "gemini":
            model = model or "gemini-1.5-pro-latest"
        else:
            return {"error": f"Unsupported llm: {llm}"}, 400

        # --- Read the primer and llm_content.json from GCS ---
        storage_client = storage.Client()
        bucket = storage_client.bucket("cryptomancer")
        primer_blob = bucket.blob("primers/llm_data_collection_and_prediction_primer.md")
        primer_text = primer_blob.download_as_text()
        llm_content_blob = bucket.blob("primers/llm_content.json")
        llm_content_example = llm_content_blob.download_as_text()

        # --- Read the CSV file from GCS ---
        csv_blob = bucket.blob(f"prediction_cache/{csv_filename}")
        csv_text = csv_blob.download_as_text()

        # --- Compose the prompt for LLM ---
        prompt = (
            f"{primer_text}\n\n"
            f"--- LLM_CONTENT.JSON START ---\n"
            f"{llm_content_example}\n"
            f"--- LLM_CONTENT.JSON END ---\n\n"
            f"Below is the latest hourly crypto data (CSV):\n"
            f"```\n{csv_text}\n```\n"
            "Please analyze the data and return ONLY a JSON object following the exact format described above. Only use the data provided above. Do not mention any inability to access data. Return ONLY the JSON object as described."
        )

        # --- Route to the correct LLM ---
        llm_response_text = None
        llm_status = "success"
        llm_error = None

        if llm == "openai":
            import openai
            client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": "You are a crypto market analyst and data engineer."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.2,
                    max_tokens=2000
                )
                llm_response_text = response.choices[0].message.content
            except Exception as e:
                llm_status = "failure"
                llm_error = f"OpenAI error: {e}"
        elif llm == "gemini":
            try:
                import google.generativeai as genai

                genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))

                model_obj = genai.GenerativeModel(model)
                response = model_obj.generate_content(prompt)
                llm_response_text = response.text

            except Exception as e:
                llm_status = "failure"
                llm_error = f"Gemini error: {e}"

        # --- Parse the JSON from the LLM response ---
        llm_json = None
        parse_error = None
        if llm_status == "success":
            try:
                cleaned = llm_response_text.strip()
                if cleaned.startswith("```"):
                    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
                    cleaned = re.sub(r"\s*```$", "", cleaned)
                llm_json = json.loads(cleaned)
            except Exception as e:
                llm_status = "failure"
                parse_error = f"Failed to parse LLM JSON: {e}"
                llm_error = parse_error

        # --- Save to BigQuery ---
        bq_client = bigquery.Client(project="cryptomancer-456619")
        run_timestamp = None
        sentiment_errors = None
        predictions_errors = None

        if llm_status == "success" and llm_json:
            try:
                sentiment_table = "cryptomancer-456619.AI.sentiment"
                ds = llm_json["data_summary"]
                run_timestamp = llm_json.get("run_timestamp")

                sentiment_row = {
                    "run_timestamp": run_timestamp,
                    "btc_google_trends_summary": ds["btc"].get("google_trends_summary"),
                    "btc_google_trends_sentiment": ds["btc"].get("google_trends_sentiment"),
                    "btc_social_media_summary": ds["btc"].get("social_media_summary"),
                    "btc_social_media_sentiment": ds["btc"].get("social_media_sentiment"),
                    "btc_news_summary": ds["btc"].get("news_summary"),
                    "btc_news_sentiment": ds["btc"].get("news_sentiment"),
                    "btc_indicators_summary": ds["btc"].get("indicators_summary"),
                    "btc_indicators_sentiment": ds["btc"].get("indicators_sentiment"),
                    "btc_minute_data_summary": ds["btc"].get("minute_data_summary"),
                    "btc_minute_data_sentiment": ds["btc"].get("minute_data_sentiment"),
                    "btc_overall_sentiment_score": ds["btc"].get("overall_sentiment_score"),
                    "eth_google_trends_summary": ds["eth"].get("google_trends_summary"),
                    "eth_google_trends_sentiment": ds["eth"].get("google_trends_sentiment"),
                    "eth_social_media_summary": ds["eth"].get("social_media_summary"),
                    "eth_social_media_sentiment": ds["eth"].get("social_media_sentiment"),
                    "eth_news_summary": ds["eth"].get("news_summary"),
                    "eth_news_sentiment": ds["eth"].get("news_sentiment"),
                    "eth_indicators_summary": ds["eth"].get("indicators_summary"),
                    "eth_indicators_sentiment": ds["eth"].get("indicators_sentiment"),
                    "eth_minute_data_summary": ds["eth"].get("minute_data_summary"),
                    "eth_minute_data_sentiment": ds["eth"].get("minute_data_sentiment"),
                    "eth_overall_sentiment_score": ds["eth"].get("overall_sentiment_score"),
                }

                sentiment_errors = bq_client.insert_rows_json(sentiment_table, [sentiment_row])
                if sentiment_errors:
                    llm_status = "failure"
                    llm_error = f"BigQuery sentiment insert errors: {sentiment_errors}"

                # --- Insert predictions into AI.predictions table ---
                predictions_table = "cryptomancer-456619.AI.predictions"
                preds = llm_json["predictions"]

                def pred_field(token, hours, field):
                    suffix = f"{hours}_hour" if hours == 1 else f"{hours}_hours"
                    return f"{token.lower()}_{suffix}_{field}"

                pred_row = {"run_timestamp": run_timestamp}
                for token in ["BTC", "ETH"]:
                    for hours in [1, 3, 6, 12, 24]:
                        pred = next((p for p in preds if p["token"] == token and p["timeframe_hours"] == hours), None)
                        if pred:
                            pred_row[pred_field(token, hours, "direction")] = pred.get("prediction_direction")
                            pred_row[pred_field(token, hours, "confidence")] = pred.get("prediction_confidence")
                        else:
                            pred_row[pred_field(token, hours, "direction")] = None
                            pred_row[pred_field(token, hours, "confidence")] = None

                predictions_errors = bq_client.insert_rows_json(predictions_table, [pred_row])
                if predictions_errors:
                    llm_status = "failure"
                    llm_error = f"BigQuery predictions insert errors: {predictions_errors}"

            except Exception as e:
                llm_status = "failure"
                llm_error = f"BigQuery error: {e}"

        # --- Log the run in AI.run_log ---
        try:
            run_log_table = "cryptomancer-456619.AI.run_log"
            log_row = {
                "run_timestamp": run_timestamp or datetime.utcnow().isoformat(),
                "llm": llm,
                "model": model,
                "status": llm_status,
                "response": llm_response_text if llm_response_text else (llm_error or "No response")
            }
            bq_client.insert_rows_json(run_log_table, [log_row])
        except Exception as e:
            print(f"Failed to log run in run_log: {e}")

        # --- Discord notifications ---
        if llm_status == "success":
            send_discord_message(
                f":white_check_mark: LLM predictions saved to BigQuery for {run_timestamp} (LLM: {llm}, Model: {model})"
            )
            return {"status": "success"}, 200
        else:
            send_discord_message(
                f":x: LLM prediction job failed (LLM: {llm}, Model: {model}): {llm_error}"
            )
            return {"error": llm_error or "Unknown error"}, 500

    except Exception as e:
        # Log to run_log and notify Discord on top-level error
        try:
            bq_client = bigquery.Client(project="cryptomancer-456619")
            run_log_table = "cryptomancer-456619.AI.run_log"
            log_row = {
                "run_timestamp": datetime.utcnow().isoformat(),
                "llm": data.get("llm", "openai") if 'data' in locals() else "openai",
                "model": data.get("model", "gpt-3.5-turbo") if 'data' in locals() else "gpt-3.5-turbo",
                "status": "failure",
                "response": str(e)
            }
            bq_client.insert_rows_json(run_log_table, [log_row])
        except Exception as log_e:
            print(f"Failed to log run in run_log (outer except): {log_e}")
        send_discord_message(f":x: LLM prediction job failed (outer except): {e}")
        return {"error": str(e)}, 500

def send_discord_message(content, webhook_url=None):
    """
    Sends a message to Discord via webhook.
    """
    if webhook_url is None:
        webhook_url = os.environ.get("DISCORD_WEBHOOK_URL")
    if not webhook_url:
        print("Discord webhook URL not set.")
        return
    try:
        response = requests.post(webhook_url, json={"content": content})
        if response.status_code != 204:
            print(f"Discord webhook error: {response.status_code} {response.text}")
    except Exception as e:
        print(f"Failed to send Discord message: {e}")

from google.cloud import bigquery
import pandas as pd

def api_calculate_actuals(request):
    """
    API endpoint to calculate actual price changes and direction for BTC and ETH
    over 1, 3, 6, 12, 24 hour timeframes, and store results in AI.actuals.
    Expects POST JSON:
    {
        "run_timestamp": "2025-04-20T19:00:00Z"  # ISO8601 UTC
    }
    """
    import os
    import json

    try:
        data = request.get_json()
        run_timestamp = data.get("run_timestamp")
        if not run_timestamp:
            return {"error": "run_timestamp is required"}, 400

        project_id = "cryptomancer-456619"
        dataset = "Coindesk"

        results = []
        bq_client = bigquery.Client(project=project_id)

        query_base = f"""
            -- Replace this with your target hour
        DECLARE run_timestamp TIMESTAMP DEFAULT TIMESTAMP('{run_timestamp}');

        WITH
        btc AS (
        SELECT
            MAX(IF(TIMESTAMP = run_timestamp - INTERVAL 1 MINUTE, CLOSE, NULL)) AS close_0,
            MAX(IF(TIMESTAMP = run_timestamp - INTERVAL 61 MINUTE, CLOSE, NULL)) AS close_1h,
            MAX(IF(TIMESTAMP = run_timestamp - INTERVAL 181 MINUTE, CLOSE, NULL)) AS close_3h,
            MAX(IF(TIMESTAMP = run_timestamp - INTERVAL 361 MINUTE, CLOSE, NULL)) AS close_6h,
            MAX(IF(TIMESTAMP = run_timestamp - INTERVAL 721 MINUTE, CLOSE, NULL)) AS close_12h,
            MAX(IF(TIMESTAMP = run_timestamp - INTERVAL 1441 MINUTE, CLOSE, NULL)) AS close_24h
        FROM `cryptomancer-456619.Coindesk.btc-usdc`
        WHERE TIMESTAMP BETWEEN run_timestamp - INTERVAL 25 HOUR AND run_timestamp
        ),
        eth AS (
        SELECT
            MAX(IF(TIMESTAMP = run_timestamp - INTERVAL 1 MINUTE, CLOSE, NULL)) AS close_0,
            MAX(IF(TIMESTAMP = run_timestamp - INTERVAL 61 MINUTE, CLOSE, NULL)) AS close_1h,
            MAX(IF(TIMESTAMP = run_timestamp - INTERVAL 181 MINUTE, CLOSE, NULL)) AS close_3h,
            MAX(IF(TIMESTAMP = run_timestamp - INTERVAL 361 MINUTE, CLOSE, NULL)) AS close_6h,
            MAX(IF(TIMESTAMP = run_timestamp - INTERVAL 721 MINUTE, CLOSE, NULL)) AS close_12h,
            MAX(IF(TIMESTAMP = run_timestamp - INTERVAL 1441 MINUTE, CLOSE, NULL)) AS close_24h
        FROM `cryptomancer-456619.Coindesk.eth-usdc`
        WHERE TIMESTAMP BETWEEN run_timestamp - INTERVAL 25 HOUR AND run_timestamp
        )
        SELECT
        run_timestamp,
        -- BTC
        CASE WHEN btc.close_1h IS NULL OR btc.close_0 IS NULL THEN NULL
            ELSE CASE WHEN ((btc.close_0 - btc.close_1h) / btc.close_1h) * 100 >= 0.25 THEN 1
                        WHEN ((btc.close_0 - btc.close_1h) / btc.close_1h) * 100 <= -0.25 THEN -1
                        ELSE 0 END END AS btc_1_hour_actual_direction,
        CASE WHEN btc.close_1h IS NULL OR btc.close_0 IS NULL THEN NULL
            ELSE ((btc.close_0 - btc.close_1h) / btc.close_1h) * 100 END AS btc_1_hour_pct_change,

        CASE WHEN btc.close_3h IS NULL OR btc.close_0 IS NULL THEN NULL
            ELSE CASE WHEN ((btc.close_0 - btc.close_3h) / btc.close_3h) * 100 >= 0.25 THEN 1
                        WHEN ((btc.close_0 - btc.close_3h) / btc.close_3h) * 100 <= -0.25 THEN -1
                        ELSE 0 END END AS btc_3_hours_actual_direction,
        CASE WHEN btc.close_3h IS NULL OR btc.close_0 IS NULL THEN NULL
            ELSE ((btc.close_0 - btc.close_3h) / btc.close_3h) * 100 END AS btc_3_hours_pct_change,

        CASE WHEN btc.close_6h IS NULL OR btc.close_0 IS NULL THEN NULL
            ELSE CASE WHEN ((btc.close_0 - btc.close_6h) / btc.close_6h) * 100 >= 0.25 THEN 1
                        WHEN ((btc.close_0 - btc.close_6h) / btc.close_6h) * 100 <= -0.25 THEN -1
                        ELSE 0 END END AS btc_6_hours_actual_direction,
        CASE WHEN btc.close_6h IS NULL OR btc.close_0 IS NULL THEN NULL
            ELSE ((btc.close_0 - btc.close_6h) / btc.close_6h) * 100 END AS btc_6_hours_pct_change,

        CASE WHEN btc.close_12h IS NULL OR btc.close_0 IS NULL THEN NULL
            ELSE CASE WHEN ((btc.close_0 - btc.close_12h) / btc.close_12h) * 100 >= 0.25 THEN 1
                        WHEN ((btc.close_0 - btc.close_12h) / btc.close_12h) * 100 <= -0.25 THEN -1
                        ELSE 0 END END AS btc_12_hours_actual_direction,
        CASE WHEN btc.close_12h IS NULL OR btc.close_0 IS NULL THEN NULL
            ELSE ((btc.close_0 - btc.close_12h) / btc.close_12h) * 100 END AS btc_12_hours_pct_change,

        CASE WHEN btc.close_24h IS NULL OR btc.close_0 IS NULL THEN NULL
            ELSE CASE WHEN ((btc.close_0 - btc.close_24h) / btc.close_24h) * 100 >= 0.25 THEN 1
                        WHEN ((btc.close_0 - btc.close_24h) / btc.close_24h) * 100 <= -0.25 THEN -1
                        ELSE 0 END END AS btc_24_hours_actual_direction,
        CASE WHEN btc.close_24h IS NULL OR btc.close_0 IS NULL THEN NULL
            ELSE ((btc.close_0 - btc.close_24h) / btc.close_24h) * 100 END AS btc_24_hours_pct_change,

        -- ETH
        CASE WHEN eth.close_1h IS NULL OR eth.close_0 IS NULL THEN NULL
            ELSE CASE WHEN ((eth.close_0 - eth.close_1h) / eth.close_1h) * 100 >= 0.25 THEN 1
                        WHEN ((eth.close_0 - eth.close_1h) / eth.close_1h) * 100 <= -0.25 THEN -1
                        ELSE 0 END END AS eth_1_hour_actual_direction,
        CASE WHEN eth.close_1h IS NULL OR eth.close_0 IS NULL THEN NULL
            ELSE ((eth.close_0 - eth.close_1h) / eth.close_1h) * 100 END AS eth_1_hour_pct_change,

        CASE WHEN eth.close_3h IS NULL OR eth.close_0 IS NULL THEN NULL
            ELSE CASE WHEN ((eth.close_0 - eth.close_3h) / eth.close_3h) * 100 >= 0.25 THEN 1
                        WHEN ((eth.close_0 - eth.close_3h) / eth.close_3h) * 100 <= -0.25 THEN -1
                        ELSE 0 END END AS eth_3_hours_actual_direction,
        CASE WHEN eth.close_3h IS NULL OR eth.close_0 IS NULL THEN NULL
            ELSE ((eth.close_0 - eth.close_3h) / eth.close_3h) * 100 END AS eth_3_hours_pct_change,

        CASE WHEN eth.close_6h IS NULL OR eth.close_0 IS NULL THEN NULL
            ELSE CASE WHEN ((eth.close_0 - eth.close_6h) / eth.close_6h) * 100 >= 0.25 THEN 1
                        WHEN ((eth.close_0 - eth.close_6h) / eth.close_6h) * 100 <= -0.25 THEN -1
                        ELSE 0 END END AS eth_6_hours_actual_direction,
        CASE WHEN eth.close_6h IS NULL OR eth.close_0 IS NULL THEN NULL
            ELSE ((eth.close_0 - eth.close_6h) / eth.close_6h) * 100 END AS eth_6_hours_pct_change,

        CASE WHEN eth.close_12h IS NULL OR eth.close_0 IS NULL THEN NULL
            ELSE CASE WHEN ((eth.close_0 - eth.close_12h) / eth.close_12h) * 100 >= 0.25 THEN 1
                        WHEN ((eth.close_0 - eth.close_12h) / eth.close_12h) * 100 <= -0.25 THEN -1
                        ELSE 0 END END AS eth_12_hours_actual_direction,
        CASE WHEN eth.close_12h IS NULL OR eth.close_0 IS NULL THEN NULL
            ELSE ((eth.close_0 - eth.close_12h) / eth.close_12h) * 100 END AS eth_12_hours_pct_change,

        CASE WHEN eth.close_24h IS NULL OR eth.close_0 IS NULL THEN NULL
            ELSE CASE WHEN ((eth.close_0 - eth.close_24h) / eth.close_24h) * 100 >= 0.25 THEN 1
                        WHEN ((eth.close_0 - eth.close_24h) / eth.close_24h) * 100 <= -0.25 THEN -1
                        ELSE 0 END END AS eth_24_hours_actual_direction,
        CASE WHEN eth.close_24h IS NULL OR eth.close_0 IS NULL THEN NULL
            ELSE ((eth.close_0 - eth.close_24h) / eth.close_24h) * 100 END AS eth_24_hours_pct_change

        FROM btc, eth;
        """
        # Execute the query
        query_job = bq_client.query(query_base)
        results = query_job.result().to_dataframe().to_dict(orient="records")
        if not results:
            return {"error": "No results found for the specified run_timestamp."}, 404
        # Print the results of the query to the terminal
        print("Query Results:")
        for row in results:
            print(row)
        # Convert any pandas.Timestamp to string (ISO format) for BigQuery compatibility
        for row in results:
            if "run_timestamp" in row and hasattr(row["run_timestamp"], "isoformat"):
                row["run_timestamp"] = row["run_timestamp"].isoformat()

        # Insert results into BigQuery
        actuals_table = f"{project_id}.AI.actuals"
        errors = bq_client.insert_rows_json(actuals_table, results)
        
        if errors:
            return {"error": f"BigQuery insert errors: {errors}"}, 500

        return {"status": "success", "rows": results}, 200

    except Exception as e:
        return {"error": str(e)}, 500
