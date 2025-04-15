import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from scipy.stats import uniform, randint
from cache_utils import save_joblib, load_joblib, write_dataframe, save_plot

# --- Constants for filenames ---
MODEL_FILENAME = "model.pkl"
PREDICTIONS_FILENAME = "df_predicted.pkl"
IMPORTANCE_PLOT_FILENAME = "feature_importance.png"
CONFUSION_MATRIX_PLOT_FILENAME = "confusion_matrix.png"


def train_model(df, feature_columns, target_column="TARGET", cache_location='local', use_model_cache=True, test_size=0.2, n_iter_search=50, scoring_metric='f1'):
    """
    Trains the XGBoost model with hyperparameter tuning, evaluates, and saves predictions & model.

    Args:
        df (pd.DataFrame): DataFrame containing features and target.
        feature_columns (list): List of column names to use as features.
        target_column (str): Name of the target variable column.
        cache_location (str): 'local' or 'gcp'. Determines where to load/save artifacts.
        use_model_cache (bool): Whether to attempt loading/saving the trained model from/to cache.
        test_size (float): Proportion for test set split.
        n_iter_search (int): Iterations for RandomizedSearchCV.
        scoring_metric (str): Metric for hyperparameter tuning.

    Returns:
        tuple: (trained_model, df_with_predictions, evaluation_results)
               evaluation_results is a dict containing metrics and plot paths/info.
    """
    # --- Validation ---
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in DataFrame.")
    if not all(col in df.columns for col in feature_columns):
        missing_features = [col for col in feature_columns if col not in df.columns]
        raise ValueError(f"Feature columns not found in DataFrame: {missing_features}")

    # --- Data Split (Time-Series) ---
    if 'timestamp' in df.columns:
        df = df.sort_values('timestamp').reset_index(drop=True)

    X = df[feature_columns]
    y = df[target_column]

    split_index = int(len(df) * (1 - test_size))
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]

    if len(X_train) == 0 or len(X_test) == 0:
         raise ValueError("Training or test set is empty after split.")

    print(f"Training set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")
    print(f"Training target distribution:\n{y_train.value_counts(normalize=True)}")
    print(f"Test target distribution:\n{y_test.value_counts(normalize=True)}")

    # --- Calculate scale_pos_weight ---
    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    scale_pos_weight_value = neg_count / pos_count if pos_count > 0 else 1
    print(f"Calculated scale_pos_weight: {scale_pos_weight_value:.4f} ({neg_count} neg / {pos_count} pos)")

    # --- Scaling ---
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    X_full_scaled = scaler.transform(X) # Scale full dataset for final predictions

    # --- Model Training & Hyperparameter Tuning ---
    model = None
    best_params = None

    # Check cache only if use_model_cache is True
    if use_model_cache:
        model = load_joblib(MODEL_FILENAME, cache_location)
        if model:
             if not hasattr(model, 'predict_proba'):
                  print("Warning: Cached object doesn't look like a classifier. Retraining.")
                  model = None
             else:
                  print("Loaded model from cache. Skipping hyperparameter tuning.")
        else:
             print("No model found in cache or error loading. Proceeding with training.")


    # Train if not loaded from cache
    if model is None:
        print(f"Starting Hyperparameter Tuning (RandomizedSearchCV, {n_iter_search} iterations, optimizing for '{scoring_metric}')...")
        # ... (param_dist definition remains the same) ...
        param_dist = {
            'n_estimators': randint(100, 500),
            'learning_rate': uniform(0.01, 0.2),
            'max_depth': randint(3, 10),
            'subsample': uniform(0.6, 0.4),
            'colsample_bytree': uniform(0.6, 0.4),
            'gamma': uniform(0, 0.5),
            'reg_alpha': uniform(0, 1),
            'reg_lambda': uniform(0, 1)
        }

        xgb_model = XGBClassifier(
            eval_metric='logloss',
            random_state=42,
            scale_pos_weight=scale_pos_weight_value
        )

        # ... (RandomizedSearchCV setup and fit remain the same) ...
        random_search = RandomizedSearchCV(
            xgb_model,
            param_distributions=param_dist,
            n_iter=n_iter_search,
            scoring=scoring_metric,
            cv=3, # Consider TimeSeriesSplit here for more robust CV
            n_jobs=-1,
            random_state=42,
            verbose=1
        )
        random_search.fit(X_train_scaled, y_train)


        print(f"Best parameters found (using '{random_search.scoring}' scoring): {random_search.best_params_}")
        print(f"Best cross-validation {random_search.scoring}: {random_search.best_score_:.4f}")

        model = random_search.best_estimator_
        best_params = random_search.best_params_ # Store best params info

        # Save the newly trained best model if caching is enabled
        if use_model_cache:
            save_joblib(model, MODEL_FILENAME, cache_location)

    # --- Evaluation on Test Set ---
    print("\n--- Model Evaluation (on Test Set) ---")
    y_pred_test = model.predict(X_test_scaled)
    y_proba_test = model.predict_proba(X_test_scaled)[:, 1]

    test_accuracy = accuracy_score(y_test, y_pred_test)
    report = classification_report(y_test, y_pred_test, output_dict=True)
    cm = confusion_matrix(y_test, y_pred_test)

    print(f"Test Set Accuracy: {test_accuracy:.4f}")
    print("Test Set Classification Report:")
    print(classification_report(y_test, y_pred_test)) # Keep text output for logs
    print("Test Set Confusion Matrix:")
    print(cm)

    # --- Generate and Save Confusion Matrix Plot ---
    try:
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
        fig, ax = plt.subplots(figsize=(8, 6))
        disp.plot(ax=ax, cmap=plt.cm.Blues)
        ax.set_title('Confusion Matrix (Test Set)')
        save_plot(fig, CONFUSION_MATRIX_PLOT_FILENAME, cache_location)
        cm_plot_path = f"{cache_location}/{CONFUSION_MATRIX_PLOT_FILENAME}" # Indicate location
    except Exception as e:
        print(f"Could not generate/save confusion matrix plot: {e}")
        cm_plot_path = None


    # --- Generate Predictions for the Entire Dataset ---
    print("\nGenerating predictions for the full dataset...")
    y_proba_full = model.predict_proba(X_full_scaled)[:, 1]

    df_out = df.copy()
    df_out["Predicted_Prob_Up"] = y_proba_full

    # Save predictions
    write_dataframe(df_out, PREDICTIONS_FILENAME, cache_location)

    # --- Feature Importance ---
    importance_plot_path = None
    importance_list = []
    try:
        importances = model.feature_importances_
        feature_names = feature_columns
        importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
        importance_df = importance_df.sort_values(by='Importance', ascending=False).reset_index(drop=True)
        importance_list = importance_df.to_dict('records') # Get list for response

        print("\n--- Feature Importances (Sorted) ---")
        print(importance_df.to_string())

        # Plot Top N
        fig_imp, ax_imp = plt.subplots(figsize=(12, 8))
        top_n = min(20, len(importance_df))
        ax_imp.bar(range(top_n), importance_df['Importance'].iloc[:top_n], align="center")
        ax_imp.set_xticks(range(top_n))
        ax_imp.set_xticklabels(importance_df['Feature'].iloc[:top_n], rotation=60, ha="right")
        ax_imp.set_title(f"Top {top_n} Feature Importances (XGBoost)")
        ax_imp.set_ylabel("Importance")
        ax_imp.set_xlabel("Feature")
        fig_imp.tight_layout()
        save_plot(fig_imp, IMPORTANCE_PLOT_FILENAME, cache_location)
        importance_plot_path = f"{cache_location}/{IMPORTANCE_PLOT_FILENAME}" # Indicate location

    except Exception as e:
        print(f"Could not generate/print/save feature importance: {e}")

    # --- Consolidate Evaluation Results ---
    evaluation_results = {
        "accuracy": test_accuracy,
        "classification_report": report,
        "confusion_matrix": cm.tolist(), # Convert numpy array for JSON compatibility
        "confusion_matrix_plot": cm_plot_path,
        "feature_importances": importance_list,
        "feature_importance_plot": importance_plot_path,
        "best_cv_score": model.best_score_ if hasattr(model, 'best_score_') else (random_search.best_score_ if 'random_search' in locals() else None),
        "best_params": best_params
    }

    return model, df_out, evaluation_results
