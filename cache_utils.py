import os
import pandas as pd
import joblib
import json
from google.cloud import storage
from io import BytesIO, StringIO
import matplotlib.pyplot as plt

# --- Configuration ---
GCS_BUCKET_NAME = os.environ.get("GCS_BUCKET_NAME")
GCS_PREDICTION_CACHE_FOLDER = os.environ.get("GCS_PREDICTION_CACHE_FOLDER", "prediction_cache")
LOCAL_PREDICTION_CACHE_FOLDER = os.path.join("cache", "prediction_cache")

# --- Helper Functions ---

def _get_gcs_blob(filename):
    """Gets a GCS blob object."""
    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(GCS_BUCKET_NAME)
        blob_path = f"{GCS_PREDICTION_CACHE_FOLDER}/{filename}"
        return bucket.blob(blob_path)
    except Exception as e:
        print(f"Error accessing GCS bucket '{GCS_BUCKET_NAME}': {e}")
        raise

def _ensure_local_dir():
    """Ensures the local cache directory exists."""
    os.makedirs(LOCAL_PREDICTION_CACHE_FOLDER, exist_ok=True)

def _get_local_path(filename):
    """Gets the full local file path."""
    _ensure_local_dir()
    return os.path.join(LOCAL_PREDICTION_CACHE_FOLDER, filename)

def _ensure_dir(file_path):
    """Ensures the directory for the given file path exists."""
    directory = os.path.dirname(file_path)
    if directory and not os.path.exists(directory):
        print(f"Creating directory: {directory}")
        os.makedirs(directory, exist_ok=True)

def _get_full_path(filename, cache_location='local', pair=None):
    """Constructs the full path, potentially including a pair subfolder."""
    base_path = os.getcwd() if cache_location == 'local' else f"gs://{os.environ.get('GCS_BUCKET_NAME', 'your-bucket-name')}" # Adjust GCS path if needed
    if pair:
        # Insert the pair name as a subfolder
        full_path = os.path.join(base_path, cache_location, pair, filename)
    else:
        # Original behavior if no pair is specified
        full_path = os.path.join(base_path, cache_location, filename)
    return full_path

# --- Read Functions ---

def read_dataframe(filename, cache_location='local', pair=None):
    """Loads DataFrame from pickle, checking pair directory."""
    full_path = _get_full_path(filename, cache_location, pair)
    if os.path.exists(full_path):
        print(f"Loading DataFrame from: {full_path}")
        return pd.read_pickle(full_path)
    else:
        print(f"DataFrame file not found at: {full_path}")
        return None

def read_json(filename, cache_location='local', pair=None):
    """Loads JSON data, checking pair directory."""
    full_path = _get_full_path(filename, cache_location, pair)
    if os.path.exists(full_path):
        print(f"Loading JSON from: {full_path}")
        with open(full_path, 'r') as f:
            return json.load(f)
    else:
        print(f"JSON file not found at: {full_path}")
        return None

def load_joblib(filename, cache_location='local', pair=None):
    """Loads object using joblib, checking pair directory."""
    full_path = _get_full_path(filename, cache_location, pair)
    if os.path.exists(full_path):
        print(f"Loading joblib object from: {full_path}")
        try:
            return joblib.load(full_path)
        except Exception as e:
            print(f"Error loading joblib file {full_path}: {e}")
            return None
    else:
        print(f"Joblib file not found at: {full_path}")
        return None

# --- Write Functions ---

def write_dataframe(df, filename, cache_location='local', pair=None):
    """Saves DataFrame to pickle, creating pair directory if needed."""
    full_path = _get_full_path(filename, cache_location, pair)
    _ensure_dir(full_path) # Ensure directory exists
    print(f"Saving DataFrame to: {full_path}")
    df.to_pickle(full_path)

def write_json(data, filename, cache_location='local', pair=None):
    """Saves JSON data, creating pair directory if needed."""
    full_path = _get_full_path(filename, cache_location, pair)
    _ensure_dir(full_path) # Ensure directory exists
    print(f"Saving JSON to: {full_path}")
    with open(full_path, 'w') as f:
        json.dump(data, f, indent=4)

def save_joblib(obj, filename, cache_location='local', pair=None):
    """Saves object using joblib, creating pair directory if needed."""
    full_path = _get_full_path(filename, cache_location, pair)
    _ensure_dir(full_path) # Ensure directory exists
    print(f"Saving joblib object to: {full_path}")
    joblib.dump(obj, full_path)

def save_plot(fig, filename, cache_location='local', pair=None):
    """Saves matplotlib plot, creating pair directory if needed."""
    full_path = _get_full_path(filename, cache_location, pair)
    _ensure_dir(full_path) # Ensure directory exists
    print(f"Saving plot to: {full_path}")
    fig.savefig(full_path)
    plt.close(fig) # Close the figure after saving

def write_csv(df, filename, cache_location):
    """Writes a DataFrame to CSV in GCS or local cache."""
    if cache_location == 'gcp':
        print(f"Writing CSV '{filename}' to GCS bucket '{GCS_BUCKET_NAME}'...")
        blob = _get_gcs_blob(filename)
        csv_buffer = StringIO()
        df.to_csv(csv_buffer, index=False)
        blob.upload_from_string(csv_buffer.getvalue(), content_type='text/csv')
        print(f"Write successful to gs://{GCS_BUCKET_NAME}/{blob.name}")
    elif cache_location == 'local':
        local_path = _get_local_path(filename)
        print(f"Writing CSV '{filename}' to local cache: {local_path}...")
        df.to_csv(local_path, index=False)
        print("Write successful.")
    else:
        raise ValueError("Invalid cache_location. Use 'gcp' or 'local'.")
