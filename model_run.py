import pandas as pd
import numpy as np
import json
from data_utils import build_dataset
from model_utils import train_model
from trade_simulator import simulate_trades
# Import write_json specifically
from cache_utils import read_dataframe, write_dataframe, read_json, write_json
from datetime import datetime

# --- Constants for filenames ---
PREPPED_DATA_FILENAME = "prepped_data.pkl"
FEATURES_FILENAME = "selected_features.json"
EVALUATION_RESULTS_FILENAME = "evaluation_results.json"
SIMULATION_SUMMARY_FILENAME = "simulation_summary.json" # Add filename for simulation summary
# Model/Prediction/Plot filenames are defined within model_utils and trade_simulator

# --- Mock Request Class (for local testing) ---
class MockRequest:
    def __init__(self, json_data):
        self._json_data = json_data

    def get_json(self):
        return self._json_data

# --- API Function: Refresh Data ---
def refresh_data(request):
    """
    API Endpoint: Fetches latest data for a date range, builds features, and saves the prepped DataFrame.

    Request JSON Body:
    {
        "start_date": "2024-07-01", // str: Start date (YYYY-MM-DD)
        "end_date": "2024-12-31",   // str: End date (YYYY-MM-DD)
        "use_bq_cache": false, // bool: Use local CSV cache for BQ results? (Default: False)
        "cache_location": "local" // str: Where to save final prepped data ('local' or 'gcp')
    }
    """
    try:
        config = request.get_json()
        # Get dates from request, provide defaults if needed (e.g., last 90 days)
        # Example default: yesterday
        default_end = (datetime.utcnow() - pd.Timedelta(days=1)).strftime('%Y-%m-%d')
        # Example default: 90 days before yesterday
        default_start = (datetime.utcnow() - pd.Timedelta(days=91)).strftime('%Y-%m-%d')

        start_date = config.get('start_date', default_start)
        end_date = config.get('end_date', default_end)
        use_bq_cache = config.get('use_bq_cache', False) # Default to False to force BQ fetch unless specified
        cache_location = config.get('cache_location', 'local')

        print(f"\n[STEP 1] Refreshing data from {start_date} to {end_date} and building features...")
        # Pass dates and BQ cache flag to build_dataset
        df = build_dataset(start_date=start_date, end_date=end_date) # Removed use_cache=use_bq_cache as it's not used in the latest data_utils

        # Save the final prepped data (using cache_location for *this* output)
        write_dataframe(df, PREPPED_DATA_FILENAME, cache_location)

        return {"status": "success", "message": f"Data refreshed ({start_date} to {end_date}) and saved to {cache_location}", "shape": df.shape}, 200

    except ValueError as ve: # Catch specific date format errors
         print(f"Error in refresh_data: {ve}")
         return {"status": "error", "message": str(ve)}, 400 # Bad request
    except Exception as e:
        print(f"Error in refresh_data: {e}")
        # Consider logging traceback here
        # import traceback
        # traceback.print_exc()
        return {"status": "error", "message": str(e)}, 500

# --- API Function: Update Features ---
def update_features(request):
    """
    API Endpoint: Loads prepped data, selects features based on exclude list, saves feature list.

    Request JSON Body:
    {
        "cache_location": "local", // str: 'local' or 'gcp'
        "exclude_cols": ["col1", "col2", ...] // list: Columns to exclude from features
    }
    """
    try:
        config = request.get_json()
        cache_location = config.get('cache_location', 'local')
        exclude_cols = config.get('exclude_cols', []) # Get exclude list from request

        print("\n[STEP 2] Updating feature selection...")
        # Load prepped data
        df = read_dataframe(PREPPED_DATA_FILENAME, cache_location)
        if df is None:
            return {"status": "error", "message": f"Prepped data '{PREPPED_DATA_FILENAME}' not found in {cache_location}."}, 404

        # --- Feature Selection Logic (adapted from original script) ---
        print(f"Columns available before selection: {len(df.columns)}")
        # Ensure core non-feature columns are always excluded
        base_exclude = ['timestamp', 'TARGET']
        final_exclude_cols = list(set(base_exclude + exclude_cols))

        potential_features = [col for col in df.columns if col not in final_exclude_cols]
        # Ensure selected features are numeric
        feature_columns = df[potential_features].select_dtypes(include=np.number).columns.tolist()

        print(f"Selected {len(feature_columns)} numeric features.")
        dropped_non_numeric = set(potential_features) - set(feature_columns)
        if dropped_non_numeric:
            print(f"Warning: Dropped non-numeric potential features: {list(dropped_non_numeric)}")
        # --- End Feature Selection Logic ---

        # Save the selected feature list
        write_json(feature_columns, FEATURES_FILENAME, cache_location)

        return {"status": "success", "message": f"Feature list updated and saved to {cache_location}", "selected_features_count": len(feature_columns), "selected_features": feature_columns}, 200

    except Exception as e:
        print(f"Error in update_features: {e}")
        return {"status": "error", "message": str(e)}, 500

# --- API Function: Run Training ---
def run_training(request):
    """
    API Endpoint: Loads data and features, trains model, saves model, predictions, and evaluation results.

    Request JSON Body:
    {
        "cache_location": "local", // str: 'local' or 'gcp'
        "use_model_cache": false, // bool: Load/Save trained model from/to cache?
        "n_iter_search": 100,    // int: Hyperparameter tuning iterations
        "scoring_metric": "f1",  // str: Tuning metric ('f1', 'roc_auc', 'accuracy')
        "test_size": 0.2         // float: Test set proportion
    }
    """
    try:
        config = request.get_json()
        cache_location = config.get('cache_location', 'local')
        use_model_cache = config.get('use_model_cache', True)
        n_iter_search = config.get('n_iter_search', 100)
        scoring_metric = config.get('scoring_metric', 'f1')
        test_size = config.get('test_size', 0.2)

        print("\n[STEP 3] Running model training...")
        # Load data and features
        df = read_dataframe(PREPPED_DATA_FILENAME, cache_location)
        if df is None:
            return {"status": "error", "message": f"Prepped data '{PREPPED_DATA_FILENAME}' not found in {cache_location}."}, 404

        feature_columns = read_json(FEATURES_FILENAME, cache_location)
        if feature_columns is None:
            return {"status": "error", "message": f"Features file '{FEATURES_FILENAME}' not found in {cache_location}. Run update_features first."}, 404

        # Call the updated train_model function
        model, df_with_predictions, evaluation_results = train_model(
            df=df,
            feature_columns=feature_columns,
            target_column="TARGET",
            cache_location=cache_location,
            use_model_cache=use_model_cache,
            test_size=test_size,
            n_iter_search=n_iter_search,
            scoring_metric=scoring_metric
        )

        # --- Save Evaluation Results ---
        write_json(evaluation_results, EVALUATION_RESULTS_FILENAME, cache_location)
        # --- End Save Evaluation Results ---

        # Model and predictions are saved within train_model now
        return {"status": "success", "message": f"Training complete. Model, predictions, and evaluation results saved to {cache_location}.", "evaluation": evaluation_results}, 200

    except Exception as e:
        print(f"Error in run_training: {e}")
        # Log traceback here if needed: import traceback; traceback.print_exc()
        return {"status": "error", "message": str(e)}, 500

# --- API Function: Run Simulation ---
def run_simulation(request):
    """
    API Endpoint: Loads predictions, runs trade simulation, saves results and summary.

    Request JSON Body:
    {
        "cache_location": "local",       // str: 'local' or 'gcp'
        "confidence_threshold": 0.80,    // float
        "hold_band": [0.45, 0.55],       // list/tuple [low, high]
        "trade_fraction": 0.3,           // float
        "fee": 0.001,                    // float
        "cooldown_period": 4             // int
    }
    """
    try:
        config = request.get_json()
        cache_location = config.get('cache_location', 'local')
        confidence_threshold = config.get('confidence_threshold', 0.80)
        hold_band = tuple(config.get('hold_band', [0.45, 0.55])) # Ensure tuple
        trade_fraction = config.get('trade_fraction', 0.3)
        fee = config.get('fee', 0.001)
        cooldown_period = config.get('cooldown_period', 4)

        print("\n[STEP 4] Running trade simulation...")
        # Load predictions data (saved by run_training)
        # Assumes PREDICTIONS_FILENAME is defined in model_utils or consistent
        from model_utils import PREDICTIONS_FILENAME as PRED_FN # Get filename
        df_predicted = read_dataframe(PRED_FN, cache_location)
        if df_predicted is None:
            return {"status": "error", "message": f"Predictions data '{PRED_FN}' not found in {cache_location}. Run run_training first."}, 404

        # Call the updated simulate_trades function
        portfolio_df, trade_df, simulation_summary = simulate_trades(
            df_sim=df_predicted,
            confidence_threshold=confidence_threshold,
            hold_band=hold_band,
            trade_fraction=trade_fraction,
            fee=fee,
            cooldown_period=cooldown_period,
            cache_location=cache_location # Pass cache location
        )

        # --- Save Simulation Summary ---
        write_json(simulation_summary, SIMULATION_SUMMARY_FILENAME, cache_location)
        # --- End Save Simulation Summary ---

        # Simulation results (CSV, plot) are saved within simulate_trades now
        return {"status": "success", "message": f"Simulation complete. Results and summary saved to {cache_location}.", "summary": simulation_summary}, 200

    except Exception as e:
        print(f"Error in run_simulation: {e}")
        # Log traceback here if needed
        return {"status": "error", "message": str(e)}, 500


# --- Main Execution (for local testing) ---
if __name__ == "__main__":
    print("--- Running Local Test ---")

    # --- Test refresh_data ---
    print("\nTesting refresh_data...")
    req_refresh = MockRequest({
        "start_date": "2024-07-01", # Specify date range
        "end_date": "2024-12-31",
        "use_bq_cache": True, # Use local CSV cache if available for this range
        "cache_location": "local" # Save final prepped data locally
    })
    res_refresh, code_refresh = refresh_data(req_refresh)
    print(f"Response ({code_refresh}): {json.dumps(res_refresh, indent=2)}")
    if code_refresh != 200: exit()

    # --- Test update_features ---
    print("\nTesting update_features...")
    req_features = MockRequest({
        "cache_location": "local",
        "exclude_cols": [ # Example exclude list
            'ETH_USDC_VOLUME', 'ETH_BTC_VOLUME', 'BTC_ETH_VOLUME',
            'ETH_USDC_VOLATILITY', 'MATIC_USDC_OPEN', 'MATIC_USDC_LOW',
            'MATIC_USDC_HIGH', 'MATIC_USDC_BBL_20_2.0', 'BTC_ETH_OPEN',
            # Original Close columns (if they exist in prepped_data)
            'ETH_MATIC_CLOSE', 'ETH_BTC_CLOSE', 'BTC_USDC_CLOSE',
            'MATIC_USDC_CLOSE', 'BTC_ETH_CLOSE', 'ETH_USDC_CLOSE',
        ]
    })
    res_features, code_features = update_features(req_features)
    print(f"Response ({code_features}): {json.dumps(res_features, indent=2)}")
    if code_features != 200: exit()

    # --- Test run_training ---
    print("\nTesting run_training...")
    req_train = MockRequest({
        "cache_location": "local",
        "use_model_cache": False, # Force retraining for test
        "n_iter_search": 50, # Reduce iterations for faster test
        "scoring_metric": "f1",
        "test_size": 0.2
    })
    res_train, code_train = run_training(req_train)
    print(f"Response ({code_train}): {json.dumps(res_train, indent=2)}")
    if code_train != 200: exit()

    # --- Test run_simulation ---
    print("\nTesting run_simulation...")
    req_sim = MockRequest({
        "cache_location": "local",
        "confidence_threshold": 0.80,
        "hold_band": [0.45, 0.55],
        "trade_fraction": 0.3,
        "fee": 0.001,
        "cooldown_period": 4
    })
    res_sim, code_sim = run_simulation(req_sim)
    print(f"Response ({code_sim}): {json.dumps(res_sim, indent=2)}")

    print("\n--- Local Test Finished ---")