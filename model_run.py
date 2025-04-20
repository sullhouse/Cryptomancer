import pandas as pd
import numpy as np
import json
import os
import joblib
import matplotlib.pyplot as plt
from data_utils import build_dataset
from model_utils import train_model
from trade_simulator import TradeSimulator
from cache_utils import read_dataframe, write_dataframe, read_json, write_json, load_joblib, save_joblib
from datetime import datetime

# --- Constants for filenames (base names) ---
PREPPED_DATA_FILENAME = "prepped_data.pkl"
FEATURES_FILENAME = "selected_features.json"
MODEL_FILENAME = "model.pkl" # Base name for model file
PREDICTIONS_FILENAME = "df_predicted.pkl" # Base name for prediction file
EVALUATION_RESULTS_FILENAME = "evaluation_results.json" # Base name for evaluation
SIMULATION_SUMMARY_FILENAME = "simulation_summary.json" # Base name for simulation summary
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
    API Endpoint: Fetches latest data for ALL pairs, builds features targeting a SPECIFIC pair,
                  and saves the prepped DataFrame in the pair's subfolder.

    Request JSON Body:
    {
        "pair": "ETH_USDC",         // str: The specific pair to generate the target for (e.g., ETH_USDC, ETH_MATIC)
        "start_date": "2024-07-01", // str: Start date (YYYY-MM-DD)
        "end_date": "2024-12-31",   // str: End date (YYYY-MM-DD)
        "cache_location": "local",  // str: 'local' or 'gcp'
        "interval_hours": 1         // int: Resampling interval in hours
    }
    """
    try:
        config = request.get_json()
        pair = config.get('pair')
        if not pair:
            return {"status": "error", "message": "Missing 'pair' parameter in request."}, 400
        default_end = (datetime.utcnow() - pd.Timedelta(days=1)).strftime('%Y-%m-%d')
        default_start = (datetime.utcnow() - pd.Timedelta(days=91)).strftime('%Y-%m-%d')
        start_date = config.get('start_date', default_start)
        end_date = config.get('end_date', default_end)
        cache_location = config.get('cache_location', 'local')
        interval_hours = config.get('interval_hours', 3)

        print(f"\n[STEP 1 - {pair}] Refreshing data (all pairs) and building features for target {pair}...")
        # build_dataset now needs the target 'pair' to pass to build_features
        df = build_dataset(
            target_pair=pair, # Pass the specific pair for target generation
            start_date=start_date,
            end_date=end_date,
            interval_hours=interval_hours
        )

        # Save the final prepped data to the pair-specific folder
        write_dataframe(df, PREPPED_DATA_FILENAME, cache_location, pair=pair)

        return {"status": "success", "message": f"Data refreshed and prepped for {pair} saved to {cache_location}/{pair}", "shape": df.shape}, 200

    except ValueError as ve: # Catch specific date format errors or empty data errors from build_dataset/features
         print(f"Error in refresh_data for {pair}: {ve}")
         return {"status": "error", "message": str(ve)}, 400
    except Exception as e:
        print(f"Error in refresh_data for {pair}: {e}")
        import traceback
        traceback.print_exc()
        return {"status": "error", "message": str(e)}, 500

# --- API Function: Update Features ---
def update_features(request):
    """
    API Endpoint: Loads prepped data for a specific pair, selects features, saves feature list
                  in the pair's subfolder.

    Request JSON Body:
    {
        "pair": "ETH_USDC",         // str: The specific pair being processed
        "cache_location": "local", // str: 'local' or 'gcp'
        "exclude_cols": ["col1", ...] // list: Columns to exclude from features
    }
    """
    try:
        config = request.get_json()
        pair = config.get('pair')
        if not pair:
            return {"status": "error", "message": "Missing 'pair' parameter in request."}, 400
        cache_location = config.get('cache_location', 'local')
        exclude_cols = config.get('exclude_cols', [])

        print(f"\n[STEP 2 - {pair}] Updating feature selection for {pair}...")
        # Load prepped data from the pair-specific folder
        df = read_dataframe(PREPPED_DATA_FILENAME, cache_location, pair=pair)
        if df is None:
            return {"status": "error", "message": f"Prepped data '{PREPPED_DATA_FILENAME}' not found in {cache_location}/{pair}. Run refresh_data first."}, 404

        # --- Feature Selection Logic ---
        print(f"Columns available before selection: {len(df.columns)}")
        # Define the target column name dynamically for exclusion
        target_col_name = f"TARGET_{pair.upper()}_UP"
        base_exclude = ['timestamp', target_col_name] # Exclude timestamp and the specific target
        # Also exclude any OTHER potential target columns if they exist from previous runs (shouldn't happen with new flow ideally)
        other_targets = [col for col in df.columns if col.startswith("TARGET_") and col.endswith("_UP") and col != target_col_name]
        final_exclude_cols = list(set(base_exclude + other_targets + exclude_cols))

        potential_features = [col for col in df.columns if col not in final_exclude_cols]
        feature_columns = df[potential_features].select_dtypes(include=np.number).columns.tolist()

        print(f"Selected {len(feature_columns)} numeric features for {pair}.")
        dropped_non_numeric = set(potential_features) - set(feature_columns)
        if dropped_non_numeric:
            print(f"Warning: Dropped non-numeric potential features: {list(dropped_non_numeric)}")

        # Save the selected feature list to the pair-specific folder
        write_json(feature_columns, FEATURES_FILENAME, cache_location, pair=pair)

        return {"status": "success", "message": f"Feature list for {pair} updated and saved to {cache_location}/{pair}", "selected_features_count": len(feature_columns)}, 200

    except Exception as e:
        print(f"Error in update_features for {pair}: {e}")
        return {"status": "error", "message": str(e)}, 500

# --- API Function: Run Training ---
def run_training(request):
    """
    API Endpoint: Loads data and features for a specific pair, trains a model for that pair,
                  saves model, predictions, and evaluation results in the pair's subfolder.

    Request JSON Body:
    {
        "pair": "ETH_USDC",         // str: The specific pair to train a model for
        "cache_location": "local", // str: 'local' or 'gcp'
        "use_model_cache": true,   // bool: Load/save model from/to cache?
        "n_iter_search": 100,      // int: Hyperparameter search iterations
        "scoring_metric": "f1",    // str: Metric for tuning
        "test_size": 0.2           // float: Test set split proportion
    }
    """
    try:
        config = request.get_json()
        pair = config.get('pair')
        if not pair:
            return {"status": "error", "message": "Missing 'pair' parameter in request."}, 400
        cache_location = config.get('cache_location', 'local')
        use_model_cache = config.get('use_model_cache', True)
        n_iter_search = config.get('n_iter_search', 100)
        scoring_metric = config.get('scoring_metric', 'f1')
        test_size = config.get('test_size', 0.2)

        print(f"\n[STEP 3 - {pair}] Running model training for {pair}...")
        # Load data and features from the pair-specific folder
        df = read_dataframe(PREPPED_DATA_FILENAME, cache_location, pair=pair)
        if df is None:
            return {"status": "error", "message": f"Prepped data '{PREPPED_DATA_FILENAME}' not found in {cache_location}/{pair}. Run refresh_data first."}, 404

        feature_columns = read_json(FEATURES_FILENAME, cache_location, pair=pair)
        if feature_columns is None:
            return {"status": "error", "message": f"Features file '{FEATURES_FILENAME}' not found in {cache_location}/{pair}. Run update_features first."}, 404

        # Define the target column name dynamically
        target_col_name = f"TARGET_{pair.upper()}_UP"
        if target_col_name not in df.columns:
             return {"status": "error", "message": f"Target column '{target_col_name}' not found in prepped data for {pair}."}, 400

        # Call train_model for the current target pair
        # Pass pair and cache_location so train_model saves artifacts correctly
        model, df_with_prediction, evaluation_results = train_model(
            df=df.copy(),
            feature_columns=feature_columns,
            target_column=target_col_name, # Use the specific target
            pair=pair, # Pass pair for artifact naming/saving
            cache_location=cache_location,
            use_model_cache=use_model_cache,
            test_size=test_size,
            n_iter_search=n_iter_search,
            scoring_metric=scoring_metric
        )

        # df_with_prediction is already saved by train_model in the correct folder
        # Evaluation results (JSON) also need to be saved in the pair's folder
        write_json(evaluation_results, EVALUATION_RESULTS_FILENAME, cache_location, pair=pair)

        return {"status": "success", "message": f"Training complete for {pair}. Model, predictions, and evaluation saved to {cache_location}/{pair}.", "evaluation_summary": evaluation_results}, 200

    except Exception as e:
        print(f"Error in run_training for {pair}: {e}")
        import traceback
        traceback.print_exc()
        return {"status": "error", "message": str(e)}, 500

# --- API Function: Run Simulation ---
def run_simulation(request):
    """
    API Endpoint: Loads predictions for a specific pair, runs trade simulation,
                  returns results, saving artifacts in the pair's subfolder.

    Request JSON Body:
    {
        "pair_to_swap": "ETH_USDC", // str: The pair to simulate trading (MUST match a pair for which predictions exist)
        "cache_location": "local",  // str: 'local' or 'gcp'
        "goal": "maximize_usdc",    // str: Simulation goal
        // ... other simulation parameters ...
        "initial_portfolio": {"ETH": 1.0, "USDC": 0.0}
    }
    """
    try:
        config = request.get_json()
        if not config:
             return {"status": "error", "message": "Missing JSON request body"}, 400

        # Extract parameters
        pair_to_swap = config.get('pair_to_swap') # This is the key parameter now
        if not pair_to_swap:
            return {"status": "error", "message": "Missing 'pair_to_swap' parameter in request."}, 400
        cache_location = config.get('cache_location', 'local')
        goal = config.get('goal', f'maximize_{pair_to_swap.split("_")[1].lower()}')
        confidence_threshold = config.get('confidence_threshold', 0.75)
        hold_band_input = config.get('hold_band', [0.40, 0.60])
        hold_band = tuple(hold_band_input) if isinstance(hold_band_input, list) and len(hold_band_input) == 2 else (0.40, 0.60)
        trade_fraction = config.get('trade_fraction', 0.5)
        fee = config.get('fee', 0.001)
        cooldown_period = config.get('cooldown_period', 1)
        slippage = config.get('slippage', 0.0005)
        initial_portfolio = config.get('initial_portfolio')
        if not initial_portfolio:
             base_asset_default = pair_to_swap.split('_')[0]
             quote_asset_default = pair_to_swap.split('_')[1]
             initial_portfolio = {base_asset_default: 1.0, quote_asset_default: 0.0}

        print(f"\n[STEP 4 - {pair_to_swap}] Running trade simulation for {pair_to_swap}...")
        # Load predictions from the specific pair's folder
        df_predicted = read_dataframe(PREDICTIONS_FILENAME, cache_location, pair=pair_to_swap)
        if df_predicted is None:
            return {"status": "error", "message": f"Predictions data '{PREDICTIONS_FILENAME}' not found in {cache_location}/{pair_to_swap}. Run run_training for this pair first."}, 404
        # Ensure the correct prediction column exists (it should be Predicted_Prob_Up from train_model)
        expected_pred_col = f"Predicted_Prob_Up" # train_model saves it like this now
        if expected_pred_col not in df_predicted.columns:
             return {"status": "error", "message": f"Prediction column '{expected_pred_col}' not found in {cache_location}/{pair_to_swap}/{PREDICTIONS_FILENAME}."}, 400

        # Rename the generic 'Predicted_Prob_Up' to the specific one the simulator expects
        specific_pred_col = f"Predicted_Prob_{pair_to_swap.upper()}_UP"
        df_predicted = df_predicted.rename(columns={expected_pred_col: specific_pred_col})

        df_predicted['timestamp'] = pd.to_datetime(df_predicted['timestamp'])
        df_predicted = df_predicted.sort_values('timestamp').reset_index(drop=True)

        # Initialize and run the simulator
        simulator = TradeSimulator(
            pair_to_swap=pair_to_swap, # Use the parameter from request
            goal=goal,
            fee=fee,
            slippage=slippage,
            cooldown_period=cooldown_period,
            confidence_threshold=confidence_threshold,
            hold_band=hold_band,
            trade_fraction=trade_fraction
        )
        simulator.initialize_portfolio(initial_portfolio)
        # Pass the pair name to simulate so it can save results in the correct folder
        results = simulator.simulate(df_predicted, pair=pair_to_swap, cache_location=cache_location)

        # Save the summary separately in the pair's folder (optional)
        # write_json(results['summary'], SIMULATION_SUMMARY_FILENAME, cache_location, pair=pair_to_swap)

        return {"status": "success", "message": f"Simulation complete for {pair_to_swap}. Results saved to {cache_location}/{pair_to_swap}.", **results}, 200

    except KeyError as ke:
         print(f"Error in run_simulation: Missing key {ke}")
         return {"status": "error", "message": f"Configuration or data missing key: {ke}"}, 400
    except ValueError as ve:
         print(f"Error in run_simulation: {ve}")
         return {"status": "error", "message": str(ve)}, 400
    except Exception as e:
        print(f"Error in run_simulation: {e}")
        import traceback
        traceback.print_exc()
        return {"status": "error", "message": f"An unexpected error occurred: {str(e)}"}, 500

# --- Main Execution (Update for pair-specific testing) ---
if __name__ == "__main__":
    print("--- Running Local Test (Pair-Specific) ---")
    test_pair = "ETH_USDC" # Choose a pair to test

    # --- Test refresh_data ---
    print(f"\nTesting refresh_data for {test_pair}...")
    req_refresh = MockRequest({
        "pair": test_pair,
        "start_date": "2024-11-01", # Shorter range for faster testing
        "end_date": "2025-03-31",
        "cache_location": "local",
        "interval_hours": 1
    })
    res_refresh, code_refresh = refresh_data(req_refresh)
    print(f"Response ({code_refresh}): {json.dumps(res_refresh, indent=2)}")
    if code_refresh != 200: exit()

    # --- Test update_features ---
    print(f"\nTesting update_features for {test_pair}...")
    req_features = MockRequest({
        "pair": test_pair,
        "cache_location": "local",
        "exclude_cols": [] # Minimal excludes for testing
    })
    res_features, code_features = update_features(req_features)
    print(f"Response ({code_features}): {json.dumps(res_features, indent=2)}")
    if code_features != 200: exit()

    # --- Test run_training ---
    print(f"\nTesting run_training for {test_pair}...")
    req_train = MockRequest({
        "pair": test_pair,
        "cache_location": "local",
        "use_model_cache": False, # Force retraining
        "n_iter_search": 10, # Very few iterations for speed
        "scoring_metric": "accuracy", # Faster metric
        "test_size": 0.2
    })
    res_train, code_train = run_training(req_train)
    print(f"Response ({code_train}): {json.dumps(res_train, indent=2)}")
    if code_train != 200: exit()

    # --- Test run_simulation ---
    print(f"\nTesting run_simulation for {test_pair}...")
    req_sim = MockRequest({
        "pair_to_swap": test_pair, # Must match the trained pair
        "cache_location": "local",
        "goal": f"maximize_{test_pair.split('_')[1].lower()}",
        "confidence_threshold": 0.60, # Adjust as needed
        "hold_band": [0.40, 0.60],
        "trade_fraction": 0.5,
        "fee": 0.001,
        "cooldown_period": 1,
        "slippage": 0.0005,
        "initial_portfolio": {test_pair.split('_')[0]: 1.0, test_pair.split('_')[1]: 0.0}
    })
    res_sim, code_sim = run_simulation(req_sim)
    print(f"Response ({code_sim}): {json.dumps(res_sim, indent=2)}")

    print("\n--- Local Test Finished ---")