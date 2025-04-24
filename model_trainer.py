import pandas as pd
import numpy as np
import warnings

# Suppress specific warnings
warnings.filterwarnings("ignore", category=UserWarning, 
                       module='joblib.externals.loky.backend.resource_tracker')
warnings.filterwarnings("ignore", message='.*The feature names should match.*')
warnings.filterwarnings("ignore", message='.*Parameters:.*are not used.*')
warnings.filterwarnings("ignore", message='.*X does not have valid feature names.*')

# Updated imports for tuning
from sklearn.model_selection import train_test_split, RandomizedSearchCV, TimeSeriesSplit
# from sklearn.ensemble import RandomForestClassifier # Remove or comment out
from xgboost import XGBClassifier # Import XGBoost
from sklearn.metrics import classification_report, accuracy_score, make_scorer, f1_score
from sklearn.metrics import roc_auc_score, precision_score, recall_score  # Add these
from scipy.stats import randint, uniform # Added uniform for float hyperparameters
# ... rest of the imports
from google.cloud import bigquery
import joblib
from datetime import timedelta, datetime # Add datetime import
from functools import reduce
import os
import ta
import json # Added for printing best params

# --- Configuration ---
PROJECT_ID = "cryptomancer-456619" # Replace with your project ID
DATASET_ID = "Coindesk" # Replace with your dataset ID
# List of pairs corresponding to table names (lowercase, hyphenated)
PAIRS_TO_LOAD = ["eth-usdc", "btc-usdc", "matic-usdc", "eth-btc", "eth-matic"]
TARGET_ASSET = "MATIC" # Asset to predict (e.g., 'ETH', 'BTC') - Used later for target generation
QUOTE_ASSET = "USDC" # Primary quote asset - Used later for target generation
FEATURE_WINDOW_MINUTES = 24 * 60 # Look back window in minutes (e.g., 24 hours)
PREDICTION_HORIZONS_HOURS = [3] # Hour horizons for prediction (e.g., [1, 2, 3])
PRICE_CHANGE_THRESHOLD = 0.0025 # 0.25% threshold for UP/DOWN classification
MODEL_OUTPUT_DIR = "trained_models" # Directory to save models
CACHE_DIR = "cache" # Directory for data cache files

# --- 1. Data Loading ---
def load_minute_data(pairs: list, start_date: str, end_date: str, overwrite_cache: bool = False) -> pd.DataFrame:
    """
    Loads minute-by-minute data for multiple asset pairs, utilizing a local Parquet cache.
    If cache exists and overwrite_cache is False, loads from cache.
    Otherwise, queries BigQuery, merges data, and saves to cache.
    Selects TIMESTAMP, CLOSE, VOLUME, VOLATILITY columns.
    """
    # Ensure cache directory exists
    os.makedirs(CACHE_DIR, exist_ok=True)

    # Generate a cache filename based on pairs and date range
    pairs_str = "_".join(sorted(pairs))
    cache_filename = f"raw_data_{pairs_str}_{start_date}_to_{end_date}.parquet"
    cache_filepath = os.path.join(CACHE_DIR, cache_filename)

    # Check for cache file
    if not overwrite_cache and os.path.exists(cache_filepath):
        print(f"Loading merged data from cache file: {cache_filepath}")
        try:
            merged_df = pd.read_parquet(cache_filepath)
            # Ensure index is datetime after loading from parquet
            if not pd.api.types.is_datetime64_any_dtype(merged_df.index):
                 merged_df.index = pd.to_datetime(merged_df.index, utc=True)
            print(f"Loaded {len(merged_df)} rows from cache.")
            return merged_df
        except Exception as e:
            print(f"Error loading from cache file {cache_filepath}: {e}. Will try loading from BigQuery.")

    # If cache doesn't exist or overwrite is True, load from BigQuery
    print(f"Cache not found or overwrite=True. Loading data for pairs {pairs} from {start_date} to {end_date} via BigQuery...")
    client = bigquery.Client(project=PROJECT_ID)
    all_dfs = []

    for pair in pairs:
        table_name = pair # Assumes table name matches the pair string
        prefix = pair.upper().replace('-', '_') + '_'

        query = f"""
            SELECT
                TIMESTAMP as timestamp,
                CLOSE as {prefix}CLOSE,
                VOLUME as {prefix}VOLUME,
                VOLATILITY as {prefix}VOLATILITY
            FROM
                `{PROJECT_ID}.{DATASET_ID}.{table_name}`
            WHERE
                TIMESTAMP >= TIMESTAMP(@start_date)
                AND TIMESTAMP < TIMESTAMP(@end_date)
            ORDER BY
                timestamp ASC
        """
        print(f"  Querying {table_name}...")
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("start_date", "STRING", start_date),
                bigquery.ScalarQueryParameter("end_date", "STRING", end_date),
            ]
        )
        try:
            df_pair = client.query(query, job_config=job_config).to_dataframe() # Renamed to df_pair
            if df_pair.empty:
                print(f"  Warning: No data returned for {pair} in the specified range.")
                continue
            df_pair['timestamp'] = pd.to_datetime(df_pair['timestamp'], utc=True)
            df_pair = df_pair.set_index('timestamp')
            all_dfs.append(df_pair)
            print(f"  Loaded {len(df_pair)} rows for {pair}.")
        except Exception as e:
            print(f"  Error loading data for {pair}: {e}")

    if not all_dfs:
        print("Error: No data loaded for any pair. Exiting.")
        return pd.DataFrame()

    # Merge all dataframes using outer join on the timestamp index
    print("Merging dataframes...")
    merged_df = reduce(lambda left, right: pd.merge(left, right, on='timestamp', how='outer'), all_dfs)
    merged_df = merged_df.sort_index()
    print(f"Finished loading and merging. Final DataFrame shape: {merged_df.shape}")

    # Save the newly loaded and merged data to cache
    try:
        print(f"Saving merged data to cache file: {cache_filepath}")
        merged_df.to_parquet(cache_filepath)
    except Exception as e:
        print(f"Error saving data to cache file {cache_filepath}: {e}")

    return merged_df

# Modify the load_code_repository_features function to use asset names instead of pairs
def load_code_repository_features(project_id, dataset_id, pairs, start_date, end_date):
    """
    Load code repository features from BigQuery and align them with price data.
    Uses asset names (ETH, BTC) instead of pairs for repository data.
    """
    client = bigquery.Client(project=project_id)
    code_features = {}
    
    # Extract unique assets from the pairs
    assets = set()
    for pair in pairs:
        # Split pairs like "eth-usdc" and take first part
        assets.add(pair.split('-')[0])
    
    print(f"Loading code repository data for assets: {assets}")
    
    for asset in assets:
        table_id = f"{project_id}.{dataset_id}.{asset.lower()}_code_repository"
        print(f"Querying code repository table: {table_id}")
        
        try:
            query = f"""
                SELECT
                    time,
                    commits,
                    lines_added,
                    lines_removed,
                    code_changes,
                    forks,
                    stars,
                    open_pull_requests,
                    closed_pull_requests,
                    open_issues,
                    closed_issues
                FROM
                    `{table_id}`
                WHERE
                    time BETWEEN '{start_date}' AND '{end_date}'
                ORDER BY
                    time
            """
            
            df = client.query(query).to_dataframe()
            if not df.empty:
                # Convert to datetime index
                df['time'] = pd.to_datetime(df['time'])
                df.set_index('time', inplace=True)
                
                # Rename columns to include asset name
                df.columns = [f"{asset.lower()}_{col}" for col in df.columns]
                
                code_features[asset] = df
                print(f"Loaded {len(df)} code repository records for {asset}")
            else:
                print(f"No code repository data found for {asset}")
                
        except Exception as e:
            print(f"Error loading code repository data for {asset}: {e}")
            # Continue with other assets even if one fails
    
    return code_features

# Then in your main workflow, merge this data with your pricing data:
def merge_code_features_with_price_data(price_data, code_features):
    """
    Merge code repository features with price data.
    """
    # Create a copy of the price data
    merged_df = price_data.copy()
    
    # Resample code features to match price data frequency (e.g., minutely)
    for asset, df in code_features.items():
        # Forward fill to create daily values for each minute
        resampled = df.asfreq('1min', method='ffill')
        
        # Merge with price data
        if not resampled.empty:
            # Ensure indices are compatible
            common_index = merged_df.index.intersection(resampled.index)
            if len(common_index) > 0:
                merged_df = pd.concat([merged_df, resampled.loc[common_index]], axis=1)
    
    return merged_df

# --- 2. Feature Engineering ---
def create_features(df: pd.DataFrame, window_minutes: int) -> pd.DataFrame:
    """
    Creates features based on rolling windows of past data, including technical indicators
    and time-based features like hour of day and day of week.
    Uses a dictionary collection approach to avoid DataFrame fragmentation.
    """
    print(f"Creating features with {window_minutes}-minute window...")
    # Ensure the input dataframe is float to avoid issues with ta library
    df = df.astype(float)
    
    # Create a dictionary to collect all features
    feature_dict = {}
    
    # --- Time-Based Features ---
    feature_dict['hour_of_day'] = df.index.hour
    feature_dict['day_of_week'] = df.index.dayofweek
    feature_dict['is_weekend'] = (df.index.dayofweek >= 5).astype(int)
    feature_dict['hour_sin'] = np.sin(2 * np.pi * df.index.hour / 24)
    feature_dict['hour_cos'] = np.cos(2 * np.pi * df.index.hour / 24)
    feature_dict['day_sin'] = np.sin(2 * np.pi * df.index.dayofweek / 7)
    feature_dict['day_cos'] = np.cos(2 * np.pi * df.index.dayofweek / 7)
    
    print("  Added time-based features (hour, day, cyclical encodings)")
    
    # --- Basic Rolling Features & Volume Change ---
    for col in df.columns:
        # Rolling averages
        feature_dict[f'{col}_rolling_mean_{window_minutes}m'] = df[col].rolling(window=window_minutes).mean()
        # Rolling std dev (volatility)
        feature_dict[f'{col}_rolling_std_{window_minutes}m'] = df[col].rolling(window=window_minutes).std()
        # Price/Volume change over the window
        denominator = df[col].shift(periods=window_minutes)
        feature_dict[f'{col}_pct_change_{window_minutes}m'] = (df[col] - denominator) / denominator.replace(0, np.nan)

        # Volume Change (Absolute) over window
        if 'VOLUME' in col:
             feature_dict[f'{col}_abs_change_{window_minutes}m'] = df[col].diff(periods=window_minutes)

    # --- Technical Indicators (RSI, MACD) ---
    # Calculate for primary pairs against USDC (or adapt as needed)
    indicator_pairs = [p for p in PAIRS_TO_LOAD if p.endswith("-usdc")]
    for pair in indicator_pairs:
        prefix = pair.upper().replace('-', '_') + '_'
        close_col = f"{prefix}CLOSE"
        if close_col in df.columns:
            print(f"  Calculating TA for {pair}...")
            # RSI
            rsi_window = 14
            feature_dict[f'{prefix}RSI_{rsi_window}m'] = ta.momentum.RSIIndicator(close=df[close_col], window=rsi_window).rsi()

            # MACD
            macd = ta.trend.MACD(close=df[close_col], window_slow=26, window_fast=12, window_sign=9)
            feature_dict[f'{prefix}MACD'] = macd.macd()
            feature_dict[f'{prefix}MACD_signal'] = macd.macd_signal()
            feature_dict[f'{prefix}MACD_diff'] = macd.macd_diff()

            # Bollinger Bands
            bb = ta.volatility.BollingerBands(close=df[close_col], window=20, window_dev=2)
            feature_dict[f'{prefix}BB_high_band'] = bb.bollinger_hband()
            feature_dict[f'{prefix}BB_low_band'] = bb.bollinger_lband()
            feature_dict[f'{prefix}BB_ma_band'] = bb.bollinger_mavg()

    # --- Inter-Asset Correlations ---
    eth_close_col = 'ETH_USDC_CLOSE'
    btc_close_col = 'BTC_USDC_CLOSE'
    matic_close_col = 'MATIC_USDC_CLOSE'

    if eth_close_col in df.columns and btc_close_col in df.columns:
        print("  Calculating ETH/BTC correlation...")
        feature_dict[f'corr_ETH_BTC_{window_minutes}m'] = df[eth_close_col].rolling(window=window_minutes).corr(df[btc_close_col])

    if eth_close_col in df.columns and matic_close_col in df.columns:
        print("  Calculating ETH/MATIC correlation...")
        feature_dict[f'corr_ETH_MATIC_{window_minutes}m'] = df[eth_close_col].rolling(window=window_minutes).corr(df[matic_close_col])

    if btc_close_col in df.columns and matic_close_col in df.columns:
        print("  Calculating BTC/MATIC correlation...")
        feature_dict[f'corr_BTC_MATIC_{window_minutes}m'] = df[btc_close_col].rolling(window=window_minutes).corr(df[matic_close_col])

    # --- Developer Activity Features (collect in dictionary) ---
    dev_features_dict = create_developer_activity_features_dict(df)
    # Merge dictionaries
    feature_dict.update(dev_features_dict)

    # Cross-asset repository activity ratios
    if 'eth_commits' in df.columns and 'btc_commits' in df.columns:
        feature_dict['eth_btc_commit_ratio'] = df['eth_commits'] / (df['btc_commits'] + 1)
        
    # Correlate code activity with price movement
    if 'eth_commits' in df.columns and 'ETH_USDC_CLOSE' in df.columns:
        # Use shorter windows (1-day and 3-day instead of 7 and 30)
        eth_commits_1d = df['eth_commits'].rolling(window=1*24*60).mean()
        eth_price_1d = df['ETH_USDC_CLOSE'].rolling(window=1*24*60).mean()
        feature_dict['eth_price_commit_corr'] = eth_commits_1d.rolling(window=3*24*60).corr(eth_price_1d)

    # Normalize code repository metrics to their historical ranges
    for asset in ['eth', 'btc', 'matic']:
        for metric in ['commits', 'stars', 'forks', 'open_issues']:
            col = f"{asset}_{metric}"
            if col in df.columns:
                # Use 7-day window instead of 90-day
                mean = df[col].rolling(window=7*24*60).mean()
                std = df[col].rolling(window=7*24*60).std()
                feature_dict[f'{col}_zscore'] = (df[col] - mean) / (std + 1e-8)

    # --- Create DataFrame from dictionary all at once ---
    features = pd.DataFrame(feature_dict, index=df.index)
    
    # --- Handle Infinite Values ---
    print(f"Shape before replacing inf: {features.shape}")
    num_infinities = np.isinf(features).sum().sum()
    if num_infinities > 0:
        print(f"Found and replacing {num_infinities} infinite values with NaN.")
        features = features.replace([np.inf, -np.inf], np.nan)

    # Fill NaNs with appropriate values based on feature type
    # Time features - fill with mode (most common value)
    for col in ['hour_of_day', 'day_of_week', 'is_weekend']:
        if col in features.columns and features[col].isna().any():
            features[col] = features[col].fillna(features[col].mode()[0])

    # Simple stats features - fill with median
    for col in features.columns:
        if '_rolling_mean_' in col or '_rolling_std_' in col:
            features[col] = features[col].fillna(features[col].median())
        elif '_pct_change_' in col:
            features[col] = features[col].fillna(0)  # No change for missing values
        elif 'corr_' in col or '_zscore' in col:
            features[col] = features[col].fillna(0)  # Neutral correlation/z-score

    print(f"Shape after filling missing values: {features.shape}")

    # Only drop rows with excessive missing values (e.g., >50% missing)
    missing_percent = features.isna().mean(axis=1)
    too_many_missing = missing_percent > 0.5
    if too_many_missing.any():
        print(f"Dropping {too_many_missing.sum()} rows with >50% missing values")
        features = features[~too_many_missing]

    print(f"Final shape after handling missing values: {features.shape}")
    
    return features

def create_developer_activity_features_dict(df, window_days=7):
    """Create features from developer activity data as a dictionary"""
    feature_dict = {}
    
    # For each asset with code repository data
    for asset in ['eth', 'btc', 'matic']:
        # Check if we have the columns
        commit_col = f"{asset}_commits"
        if commit_col in df.columns:
            # Use shorter windows for initial implementation
            # Developer activity momentum (1-day vs 7-day instead of 7 vs 30)
            short_window = 1*24*60  # 1 day
            medium_window = 7*24*60  # 7 days (was 30)
            
            feature_dict[f'{asset}_commit_momentum'] = (
                df[commit_col].rolling(window=short_window).mean() / 
                df[commit_col].rolling(window=medium_window).mean()
            )
            
            # PR closure rate
            if f"{asset}_closed_pull_requests" in df.columns and f"{asset}_open_pull_requests" in df.columns:
                feature_dict[f'{asset}_pr_health'] = (
                    df[f"{asset}_closed_pull_requests"] / 
                    (df[f"{asset}_closed_pull_requests"] + df[f"{asset}_open_pull_requests"] + 1)
                )
            
            # Github engagement ratio
            if f"{asset}_stars" in df.columns and f"{asset}_forks" in df.columns:
                feature_dict[f'{asset}_github_engagement'] = df[f"{asset}_stars"] / (df[f"{asset}_forks"] + 1)
    
    # Day ratio - use hourly instead of daily
    if 'eth_commits' in df.columns:
        feature_dict['eth_commits_hour_ratio'] = df['eth_commits'] / df['eth_commits'].rolling(window=60).mean()
    
    return feature_dict

# In create_features, use different windows for different data types
# Price technical indicators (short-term)
rsi_window = 14  # Hours
macd_fast = 12   # Hours
macd_slow = 26   # Hours

# Repository metrics (longer-term)
repo_window_short = 7 * 24 * 60  # 7 days in minutes
repo_window_long = 30 * 24 * 60  # 30 days in minutes

# --- 3. Target Engineering ---
def create_targets(df: pd.DataFrame, target_asset: str, quote_asset: str, horizons_hours: list, threshold: float) -> pd.DataFrame:
    """
    Creates binary target variables (direction: 1 for UP, -1 for DOWN) for future price changes
    using the specified target_asset/quote_asset pair's CLOSE price.
    Excludes 'flat' movements (where price change is smaller than threshold).
    """
    print(f"Creating binary targets (UP/DOWN only) for {target_asset}/{quote_asset} at horizons {horizons_hours} hours...")
    targets = pd.DataFrame(index=df.index)
    # Construct the specific column name for the target pair's close price
    price_col = f"{target_asset.upper()}_{quote_asset.upper()}_CLOSE"

    if price_col not in df.columns:
        print(f"Error: Target price column '{price_col}' not found in the loaded data. Available columns: {df.columns.tolist()}")
        return pd.DataFrame() # Return empty DataFrame

    for hours in horizons_hours:
        periods = hours * 60 # Convert hours to minutes
        future_price = df[price_col].shift(-periods)
        price_change_pct = (future_price - df[price_col]) / df[price_col]

        target_col_name = f'TARGET_{target_asset}_{hours}h_binary'  # Renamed to indicate binary classification

        # Binary classification (1 for UP, -1 for DOWN)
        # Only create target where absolute change exceeds threshold
        targets[target_col_name] = np.where(price_change_pct > 0, 1, -1)  # All non-zero changes classified as UP or DOWN
        
        # Create a mask for rows where absolute change is less than threshold (to be dropped later)
        flat_mask = (price_change_pct.abs() < threshold)
        targets[f'{target_col_name}_is_flat'] = flat_mask
        
        # Report stats on distribution
        total_rows = len(targets)
        flat_count = flat_mask.sum()
        flat_pct = flat_count / total_rows * 100
        print(f"  For {hours}h horizon: {flat_count} rows ({flat_pct:.1f}%) have change < {threshold:.4f} and will be filtered out")

    # Drop rows where future price is NaN (end of dataframe)
    targets = targets.dropna()
    
    print(f"Created targets. Shape before filtering flat movements: {targets.shape}")
    return targets

# --- 4. Model Training ---
def train_horizon_model(features: pd.DataFrame, target: pd.Series, horizon_hours: int):
    """
    Trains an XGBoost binary classification model for a specific prediction horizon using
    RandomizedSearchCV for hyperparameter tuning with TimeSeriesSplit cross-validation.
    Maps target labels from [-1, 1] to [0, 1] for XGBoost compatibility.
    Handles class imbalance with scale_pos_weight parameter.
    Returns the best trained model and evaluation metrics on the test set.
    """
    start_time = datetime.now()
    print(f"Training XGBoost binary classifier for {horizon_hours}h horizon with RandomizedSearchCV...")
    print(f"Start time: {start_time.strftime('%H:%M:%S')}")

    # --- Label Mapping ---
    original_labels = sorted(target.unique())  # Should be [-1, 1]
    if list(original_labels) != [-1, 1]:
        print(f"Warning: Unexpected original labels found: {original_labels}. Binary classification expects [-1, 1].")
    label_map = {label: i for i, label in enumerate(original_labels)}  # e.g., {-1: 0, 1: 1}
    inverse_label_map = {i: label for label, i in label_map.items()}  # e.g., {0: -1, 1: 1}
    # --- End Label Mapping ---

    # Split data into training and a final hold-out test set
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        features, target, test_size=0.2, random_state=42, shuffle=False
    )

    # --- Apply mapping to training/validation target ---
    y_train_val_mapped = y_train_val.map(label_map)
    # ---
    
    # Calculate class imbalance for proper scale_pos_weight setting
    class_counts = y_train_val_mapped.value_counts()
    print(f"  Class distribution in training set: {class_counts.to_dict()}")
    
    # For binary classification in XGBoost, scale_pos_weight should be:
    # number of negative examples / number of positive examples
    if len(class_counts) == 2:
        # Assuming the classes are 0 and 1, and 1 is the positive class
        neg_count = class_counts.get(0, 0)
        pos_count = class_counts.get(1, 0)
        if pos_count > 0:
            imbalance_ratio = neg_count / pos_count
            print(f"  Class imbalance ratio (neg/pos): {imbalance_ratio:.3f}")
        else:
            imbalance_ratio = 1.0
            print("  Warning: No positive examples found, using default scale_pos_weight")
    else:
        imbalance_ratio = 1.0
        print(f"  Warning: Expected 2 classes, found {len(class_counts)}. Using default scale_pos_weight.")
    
    # Define scale_pos_weight values based on the imbalance ratio
    # We'll test values around the calculated ratio
    scale_pos_values = [
        imbalance_ratio * 0.8,  # Slightly lower than calculated
        imbalance_ratio,        # Exactly calculated ratio
        imbalance_ratio * 1.2,  # Slightly higher than calculated
    ]
    
    print(f"  Using scale_pos_weight values: {[round(x, 3) for x in scale_pos_values]}")

    # Define the parameter distribution for XGBoost RandomizedSearchCV
    param_dist = {
        'n_estimators': randint(100, 500),
        'learning_rate': uniform(0.01, 0.3),
        'max_depth': randint(3, 10),
        'subsample': uniform(0.6, 0.4),
        'colsample_bytree': uniform(0.6, 0.4),
        'scale_pos_weight': scale_pos_values,  # Updated with actual imbalance-based values
        'gamma': [0, 0.1, 0.2, 0.3],
        'reg_alpha': [0, 0.001, 0.01, 0.1],
        'reg_lambda': [0.1, 1, 5, 10]
    }

    # Base XGBoost model - binary classification
    xgb_model = XGBClassifier(
        objective='binary:logistic',  # Changed to binary classification
        eval_metric='logloss',        # Changed to logloss for binary
        random_state=42
    )

    # TimeSeriesSplit for cross-validation
    n_splits = 5
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    # Calculate total iterations for progress tracking
    n_iter = 10
    total_fits = n_iter * n_splits
    print(f"Will perform {n_iter} parameter combinations × {n_splits} CV splits = {total_fits} total model fits")
    
    # Randomized Search setup
    # For binary classification, we'll use AUC instead of F1-macro
    scorer = make_scorer(roc_auc_score)  # AUC is a better metric for binary classification
    
    random_search = RandomizedSearchCV(
        estimator=xgb_model,
        param_distributions=param_dist,
        n_iter=n_iter,
        cv=tscv,
        scoring=scorer,
        n_jobs=-1,
        random_state=42,
        verbose=3
    )

    print(f"  Running RandomizedSearchCV for XGBoost (n_iter={n_iter}, cv={n_splits} splits)...")
    print(f"  Expected to complete approximately {total_fits} model fits")
    print(f"  Started at {datetime.now().strftime('%H:%M:%S')}")
    
    # --- Fit using the mapped target ---
    try:
        random_search.fit(X_train_val, y_train_val_mapped)
        print(f"  RandomizedSearchCV completed at {datetime.now().strftime('%H:%M:%S')}")
        print(f"  Total time: {datetime.now() - start_time}")
    except Exception as e:
        print(f"Error during model training: {e}")
        return None, None
    # ---

    print(f"  Best parameters found for {horizon_hours}h: {json.dumps(random_search.best_params_, indent=2)}")
    print(f"  Best cross-validation AUC score: {random_search.best_score_:.4f}")

    # Get the best model
    best_model = random_search.best_estimator_

    # Evaluate the *best* model on the final hold-out test set
    y_pred_proba = best_model.predict_proba(X_test)
    threshold = 0.75  # Set your custom threshold here
    y_pred_mapped = (y_pred_proba[:, 1] >= threshold).astype(int)  # Higher threshold
    print(f"Using prediction threshold: {threshold}")

    # --- Map predicted indices back to original labels ---
    y_pred_original = pd.Series(y_pred_mapped).map(inverse_label_map)
    # ---

    print(f"\nEvaluation of BEST XGBoost model for {horizon_hours}h on HOLD-OUT test set:")
    # --- Evaluate using original y_test and inverse-mapped predictions ---
    print(classification_report(y_test, y_pred_original, zero_division=0))
    accuracy = accuracy_score(y_test, y_pred_original)
    print(f"Hold-out Accuracy: {accuracy:.4f}")
    
    # Add ROC AUC score (better for binary classification)
    y_test_mapped = y_test.map(label_map)
    auc_score = roc_auc_score(y_test_mapped, y_pred_proba[:, 1])
    print(f"Hold-out ROC AUC: {auc_score:.4f}")
    
    # Calculate additional binary metrics
    precision = precision_score(y_test, y_pred_original)
    recall = recall_score(y_test, y_pred_original)
    f1 = f1_score(y_test, y_pred_original)
    
    print(f"Hold-out Precision: {precision:.4f}")
    print(f"Hold-out Recall: {recall:.4f}")
    print(f"Hold-out F1: {f1:.4f}")
    # ---

    # Store metrics
    metrics = classification_report(y_test, y_pred_original, output_dict=True, zero_division=0)
    metrics['accuracy'] = accuracy
    metrics['roc_auc'] = auc_score
    metrics['f1'] = f1
    metrics['precision'] = precision
    metrics['recall'] = recall
    metrics['best_cv_score'] = random_search.best_score_
    metrics['best_params'] = random_search.best_params_
    metrics['training_time_seconds'] = (datetime.now() - start_time).total_seconds()
    metrics['label_mapping'] = label_map

    # After training XGBoost model:
    feature_importance = pd.DataFrame({
        'feature': best_model.feature_names_in_,
        'importance': best_model.feature_importances_
    }).sort_values('importance', ascending=False)

    print("Top 20 features:")
    print(feature_importance.head(20))

    # Group by feature type to see which data sources matter most
    feature_types = {
        'price': [col for col in feature_importance['feature'] if 'CLOSE' in col],
        'volume': [col for col in feature_importance['feature'] if 'VOLUME' in col],
        'repo': [col for col in feature_importance['feature'] if any(x in col for x in ['commit', 'star', 'fork', 'issue'])]
    }

    for ftype, cols in feature_types.items():
        print(f"{ftype} importance: {feature_importance[feature_importance['feature'].isin(cols)]['importance'].sum():.4f}")

    return best_model, metrics

# --- 5. Prediction Function ---
def make_predictions(model, current_features_df: pd.DataFrame, threshold=0.75):  # Add threshold parameter
    """
    Uses a trained model (XGBoost or other) to predict direction and confidence.
    'current_features_df' should contain a single row with the latest features.
    Handles potential label mapping for XGBoost.
    Applies custom threshold for decision boundary (default 0.75).
    """
    if model is None:
        return None, None

    # Ensure columns match training
    try:
        # Standard sklearn way
        if hasattr(model, 'feature_names_in_'):
            current_features_df = current_features_df[model.feature_names_in_]
        # Fallback for raw XGBoost booster
        elif hasattr(model, 'get_booster') and hasattr(model.get_booster(), 'feature_names'):
             current_features_df = current_features_df[model.get_booster().feature_names]
        else:
            print("Warning: Cannot verify feature names. Ensure input columns match training.")
    except KeyError as e:
        print(f"Error: Missing expected feature column in input data: {e}")
        return None, None
    except Exception as e:
        print(f"Error aligning feature columns for prediction: {e}")
        return None, None

    probabilities = model.predict_proba(current_features_df)[0]
    # Instead of using argmax (which always picks the higher probability class),
    # explicitly check if the probability exceeds our threshold
    if probabilities[1] >= threshold:  # Class 1 (UP) probability exceeds threshold
        predicted_class_index = 1
    else:
        predicted_class_index = 0
    
    # --- Map predicted index back to original label ---
    # We need the inverse mapping here
    inverse_label_map = {0: -1, 1: 1} # For binary model
    direction = inverse_label_map.get(predicted_class_index)
    # ---

    if direction is None:
        print(f"Error: Could not map predicted class index {predicted_class_index} back to original label.")
        return None, None

    confidence = probabilities[predicted_class_index]

    return direction, confidence


def export_training_results(trained_models, all_metrics, features_df, config_params):
    """
    Export detailed training results to a markdown file in the models directory
    """
    import os
    from datetime import datetime
    
    # Create timestamp for the filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{MODEL_OUTPUT_DIR}/training_results_{TARGET_ASSET}_{timestamp}.md"
    
    with open(filename, 'w') as f:
        # 1. Summary Header
        f.write(f"# {TARGET_ASSET} Model Training Results - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # 2. Configuration Summary
        f.write("## Configuration\n\n")
        f.write(f"- **Target Asset:** {config_params['target_asset']}\n")
        f.write(f"- **Quote Asset:** {config_params['quote_asset']}\n")
        f.write(f"- **Feature Window:** {config_params['feature_window_minutes']} minutes\n")
        f.write(f"- **Prediction Horizons:** {config_params['prediction_horizons_hours']} hour(s)\n")
        f.write(f"- **Price Change Threshold:** {config_params['price_change_threshold']}\n")
        f.write(f"- **Training Period:** {config_params['start_date']} to {config_params['end_date']}\n\n")
        
        # 3. Data Overview
        f.write("## Data Overview\n\n")
        f.write(f"- **Total Features:** {len(features_df.columns)}\n")
        
        # Count feature types
        price_features = sum(1 for col in features_df.columns if 'CLOSE' in col)
        volume_features = sum(1 for col in features_df.columns if 'VOLUME' in col)
        repo_features = sum(1 for col in features_df.columns if any(x in col for x in ['commit', 'star', 'fork', 'issue', 'pull']))
        
        f.write(f"- **Price Features:** {price_features}\n")
        f.write(f"- **Volume Features:** {volume_features}\n")
        f.write(f"- **Repository Features:** {repo_features}\n")
        f.write(f"- **Other Features:** {len(features_df.columns) - price_features - volume_features - repo_features}\n\n")
        
        # 4. Results Summary Table
        f.write("## Results Summary\n\n")
        f.write("| Horizon | Accuracy | ROC AUC | Precision | Recall | F1 Score |\n")
        f.write("|---------|----------|---------|-----------|--------|----------|\n")
        
        for horizon, metrics in all_metrics.items():
            f.write(f"| {horizon} | {metrics['accuracy']:.4f} | {metrics['roc_auc']:.4f} | ")
            f.write(f"{metrics['precision']:.4f} | {metrics['recall']:.4f} | {metrics['f1']:.4f} |\n")
        
        f.write("\n")
        
        # 5. Detailed Metrics for Each Horizon
        for horizon, metrics in all_metrics.items():
            f.write(f"## Detailed Metrics for {horizon}\n\n")
            
            # Classification Report
            f.write("### Classification Report\n\n")
            f.write("| Class | Precision | Recall | F1-Score | Support |\n")
            f.write("|-------|-----------|--------|----------|--------|\n")
            
            for class_label, class_metrics in {k: v for k, v in metrics.items() if k in ['-1', '1']}.items():
                f.write(f"| {class_label} | {class_metrics['precision']:.4f} | {class_metrics['recall']:.4f} | ")
                f.write(f"{class_metrics['f1-score']:.4f} | {class_metrics['support']} |\n")
            
            f.write("\n")
            
            # Best Parameters
            f.write("### Best Hyperparameters\n\n")
            f.write("```json\n")
            f.write(json.dumps(metrics['best_params'], indent=2))
            f.write("\n```\n\n")
            
            # Feature Importance - Find the model for this horizon
            if horizon.rstrip('h') in [str(h) for h in trained_models.keys()]:
                model = trained_models[int(horizon.rstrip('h'))]
                
                if model and hasattr(model, 'feature_importances_'):
                    # Get feature importances
                    feature_importance = pd.DataFrame({
                        'feature': model.feature_names_in_,
                        'importance': model.feature_importances_
                    }).sort_values('importance', ascending=False)
                    
                    # Top 20 features
                    f.write("### Top 20 Features by Importance\n\n")
                    f.write("| Feature | Importance |\n")
                    f.write("|---------|------------|\n")
                    
                    for _, row in feature_importance.head(20).iterrows():
                        f.write(f"| {row['feature']} | {row['importance']:.6f} |\n")
                    
                    f.write("\n")
                    
                    # Feature importance by category
                    f.write("### Feature Importance by Category\n\n")
                    feature_types = {
                        'price': [col for col in feature_importance['feature'] if 'CLOSE' in col],
                        'volume': [col for col in feature_importance['feature'] if 'VOLUME' in col],
                        'repo': [col for col in feature_importance['feature'] if any(x in col for x in ['commit', 'star', 'fork', 'issue', 'pull'])]
                    }
                    
                    f.write("| Category | Total Importance |\n")
                    f.write("|----------|------------------|\n")
                    
                    for ftype, cols in feature_types.items():
                        importance_sum = feature_importance[feature_importance['feature'].isin(cols)]['importance'].sum()
                        f.write(f"| {ftype} | {importance_sum:.4f} |\n")
                    
                    f.write("\n")
            
            # Additional metrics
            f.write("### Additional Metrics\n\n")
            f.write(f"- **Best CV Score:** {metrics['best_cv_score']:.4f}\n")
            f.write(f"- **Training Time:** {metrics['training_time_seconds']:.2f} seconds\n\n")
        
    print(f"Training results exported to: {filename}")
    return filename

# --- Main Workflow ---
if __name__ == "__main__":
    from test import setup_environment
    setup_environment()
    overall_start_time = datetime.now()
    print(f"Starting Model Training Workflow at {overall_start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Only training for {PREDICTION_HORIZONS_HOURS} hour prediction horizons")
    print(f"Using n_iter={10} for RandomizedSearchCV")

    # Define time range for training data
    end_date_train = "2025-04-21"
    start_date_train = "2024-06-01" # Example date range

    # --- Control Cache Overwrite ---
    FORCE_RELOAD_FROM_BQ = False # Set to True to ignore cache and reload from BigQuery
    # -----------------------------

    # 1. Load Data for all specified pairs (using cache logic)
    raw_data = load_minute_data(PAIRS_TO_LOAD, start_date_train, end_date_train, overwrite_cache=FORCE_RELOAD_FROM_BQ)

    if raw_data.empty:
        print("Failed to load data. Exiting.")
        exit()

    # --- Data Preprocessing (Example: Forward Fill NaNs) ---
    # Decide on an imputation strategy for missing values from the merge
    print("Forward filling missing values...")
    raw_data_filled = raw_data.ffill()
    # Check if NaNs still exist after ffill (e.g., at the beginning)
    if raw_data_filled.isnull().values.any():
        print("Warning: NaNs still exist after forward fill. Consider backfill or dropping.")
        # raw_data_filled = raw_data_filled.bfill().dropna() # Example: backfill then drop remaining
        raw_data_filled = raw_data_filled.dropna() # Simplest: drop rows with any NaNs left
        print(f"Shape after dropping remaining NaNs: {raw_data_filled.shape}")

    if raw_data_filled.empty:
        print("Data became empty after NaN handling. Exiting.")
        exit()
    # --- End Preprocessing ---

    # Load code repository features
    code_features = load_code_repository_features(PROJECT_ID, DATASET_ID, PAIRS_TO_LOAD, start_date_train, end_date_train)

    # Merge code repository features with price data
    raw_data_with_code_features = merge_code_features_with_price_data(raw_data_filled, code_features)

    # 2. Create Features using the filled data
    features_df = create_features(raw_data_with_code_features, FEATURE_WINDOW_MINUTES)
    
    # Fix the time feature distribution printing
    if not features_df.empty:
        print("\n--- Time Feature Distribution ---")
        # Hour of day - apply round() to the Series first
        hour_dist = features_df['hour_of_day'].value_counts(normalize=True).sort_index() * 100
        print(f"Hour of day distribution: \n{hour_dist.round(1)}%")
        
        # Day of week - apply round() to the Series first
        day_dist = features_df['day_of_week'].value_counts(normalize=True).sort_index() * 100
        print(f"Day of week distribution: \n{day_dist.round(1)}%")
        
        # Weekend vs Weekday - apply round() to the Series first
        weekend_dist = features_df['is_weekend'].value_counts(normalize=True) * 100
        print(f"Weekend vs Weekday: \n{weekend_dist.round(1)}%")

    # 3. Create Targets using the filled data
    all_targets_df = create_targets(raw_data_filled, TARGET_ASSET, QUOTE_ASSET, PREDICTION_HORIZONS_HOURS, PRICE_CHANGE_THRESHOLD)

    # Align features and targets (important!)
    common_index = features_df.index.intersection(all_targets_df.index)
    features_aligned = features_df.loc[common_index]
    targets_aligned = all_targets_df.loc[common_index]

    print(f"Aligned features shape: {features_aligned.shape}")
    print(f"Aligned targets shape: {targets_aligned.shape}")

    if features_aligned.empty or targets_aligned.empty:
        print("No aligned data after feature/target creation. Check window/horizon settings. Exiting.")
        exit()

    # 4. Train Model for each horizon
    trained_models = {}
    all_metrics = {}
    # Ensure MODEL_OUTPUT_DIR exists
    os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)

    for hours in PREDICTION_HORIZONS_HOURS:
        target_col = f'TARGET_{TARGET_ASSET}_{hours}h_binary'  # Updated column name
        flat_mask_col = f'{target_col}_is_flat'
        
        if target_col not in all_targets_df.columns:
            print(f"Target column {target_col} not found. Skipping {hours}h.")
            continue

        # Filter out flat movements
        non_flat_indices = all_targets_df[~all_targets_df[flat_mask_col]].index
        # Get the intersection of valid feature rows and non-flat target rows
        valid_indices = features_aligned.index.intersection(non_flat_indices)
        
        # Subset both features and targets to only include rows where price movement exceeds threshold
        features_for_model = features_aligned.loc[valid_indices]
        targets_for_model = all_targets_df.loc[valid_indices, target_col]
        
        print(f"For {hours}h horizon: Using {len(features_for_model)} samples after removing flat movements")
        print(f"Target distribution: {targets_for_model.value_counts(normalize=True).round(3) * 100}%")
        
        if len(features_for_model) < 1000:  # Arbitrary minimum sample size
            print(f"Warning: Too few samples for {hours}h horizon after filtering. Skipping.")
            continue
            
        horizon_start_time = datetime.now()
        print(f"\n--- Starting binary classification training for {hours}h horizon at {horizon_start_time.strftime('%H:%M:%S')} ---")
        
        model, metrics = train_horizon_model(features_for_model, targets_for_model, hours)
        
        horizon_end_time = datetime.now()
        training_duration = horizon_end_time - horizon_start_time
        print(f"--- Finished {hours}h horizon training in {training_duration} ---")
        
        if model:
            trained_models[hours] = model
            if metrics:  # Check if metrics were generated
                all_metrics[f'{hours}h'] = metrics
            # Save the model
            model_filename = f"{MODEL_OUTPUT_DIR}/{TARGET_ASSET}_{hours}h_model.joblib"
            joblib.dump(model, model_filename)
            print(f"Saved model to {model_filename}")

    overall_duration = datetime.now() - overall_start_time
    print(f"\nTotal training time: {overall_duration}")

    print("\n--- Training Summary ---")
    # Convert the metrics dictionary to be JSON serializable
    def convert_to_json_serializable(obj):
        """
        Recursively convert a dictionary with numpy values to native Python types
        that can be serialized to JSON.
        """
        if isinstance(obj, dict):
            return {str(k): convert_to_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list) or isinstance(obj, tuple):
            return [convert_to_json_serializable(i) for i in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        else:
            return obj

    # Convert metrics to JSON-serializable format
    json_serializable_metrics = convert_to_json_serializable(all_metrics)

    # Pretty print the metrics
    print(json.dumps(json_serializable_metrics, indent=2))


    # 5. Example Prediction (using the last available feature row)
    if trained_models:
        print("\n--- Example Prediction ---")
        # Ensure last_features only contains columns that were used for training
        # This requires aligning before prediction if columns were added/removed
        last_row_aligned = features_aligned.iloc[[-1]]

        print(f"Using features from timestamp: {last_row_aligned.index[0]}")
        predictions = {}
        for hours, model in trained_models.items():
            # Ensure the input columns match exactly what the model was trained on
            try:
                if hasattr(model, 'feature_names_in_'):
                    prediction_input = last_row_aligned[model.feature_names_in_]
                else:
                    prediction_input = last_row_aligned # Assume columns match if attribute missing
            except KeyError as e:
                 print(f"Prediction failed for {hours}h: Missing feature column {e}")
                 continue # Skip prediction for this model

            direction, confidence = make_predictions(model, prediction_input, threshold=0.75)
            if direction is not None: # Check if prediction was successful
                print(f"Prediction for {hours}h: Direction={direction}, Confidence={confidence:.4f}")
                predictions[f'{hours}h'] = {'direction': int(direction), 'confidence': confidence} # Ensure direction is int
            else:
                 print(f"Prediction failed for {hours}h.")
    else:
        print("\nNo models were trained.")

    print("\nModel Training Workflow Finished.")
    
    # Clean up joblib temporary files
    import shutil
    import tempfile
    import glob
    import time
    
    try:
        # Sleep briefly to let background processes finish
        time.sleep(1)
        
        # Find and remove temp folders
        temp_pattern = os.path.join(tempfile.gettempdir(), 'joblib_memmapping_folder_*')
        for folder in glob.glob(temp_pattern):
            try:
                shutil.rmtree(folder, ignore_errors=True)
            except Exception:
                pass
    except Exception:
        pass  # Silently ignore any cleanup errors

    # Export the training results
    config_params = {
        'target_asset': TARGET_ASSET,
        'quote_asset': QUOTE_ASSET,
        'feature_window_minutes': FEATURE_WINDOW_MINUTES,
        'prediction_horizons_hours': PREDICTION_HORIZONS_HOURS,
        'price_change_threshold': PRICE_CHANGE_THRESHOLD,
        'start_date': start_date_train,
        'end_date': end_date_train
    }
    
    results_file = export_training_results(trained_models, all_metrics, features_df, config_params)
    print(f"\nDetailed results exported to: {results_file}")