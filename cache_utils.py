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

# --- Read Functions ---

def read_dataframe(filename, cache_location):
    """Reads a DataFrame from GCS or local cache."""
    if cache_location == 'gcp':
        print(f"Reading DataFrame '{filename}' from GCS bucket '{GCS_BUCKET_NAME}'...")
        blob = _get_gcs_blob(filename)
        if not blob.exists():
            print(f"File not found in GCS: {blob.name}")
            return None
        buffer = BytesIO()
        blob.download_to_file(buffer)
        buffer.seek(0)
        df = pd.read_pickle(buffer)
        print("Read successful.")
        return df
    elif cache_location == 'local':
        local_path = _get_local_path(filename)
        print(f"Reading DataFrame '{filename}' from local cache: {local_path}...")
        if not os.path.exists(local_path):
            print(f"File not found locally: {local_path}")
            return None
        df = pd.read_pickle(local_path)
        print("Read successful.")
        return df
    else:
        raise ValueError("Invalid cache_location. Use 'gcp' or 'local'.")

def read_json(filename, cache_location):
    """Reads a JSON file from GCS or local cache."""
    if cache_location == 'gcp':
        print(f"Reading JSON '{filename}' from GCS bucket '{GCS_BUCKET_NAME}'...")
        blob = _get_gcs_blob(filename)
        if not blob.exists():
            print(f"File not found in GCS: {blob.name}")
            return None
        content = blob.download_as_text()
        data = json.loads(content)
        print("Read successful.")
        return data
    elif cache_location == 'local':
        local_path = _get_local_path(filename)
        print(f"Reading JSON '{filename}' from local cache: {local_path}...")
        if not os.path.exists(local_path):
            print(f"File not found locally: {local_path}")
            return None
        with open(local_path, 'r') as f:
            data = json.load(f)
        print("Read successful.")
        return data
    else:
        raise ValueError("Invalid cache_location. Use 'gcp' or 'local'.")

def load_joblib(filename, cache_location):
    """Loads a joblib file (like a model) from GCS or local cache."""
    if cache_location == 'gcp':
        print(f"Loading joblib file '{filename}' from GCS bucket '{GCS_BUCKET_NAME}'...")
        blob = _get_gcs_blob(filename)
        if not blob.exists():
            print(f"File not found in GCS: {blob.name}")
            return None
        buffer = BytesIO()
        blob.download_to_file(buffer)
        buffer.seek(0)
        model = joblib.load(buffer)
        print("Load successful.")
        return model
    elif cache_location == 'local':
        local_path = _get_local_path(filename)
        print(f"Loading joblib file '{filename}' from local cache: {local_path}...")
        if not os.path.exists(local_path):
            print(f"File not found locally: {local_path}")
            return None
        model = joblib.load(local_path)
        print("Load successful.")
        return model
    else:
        raise ValueError("Invalid cache_location. Use 'gcp' or 'local'.")

# --- Write Functions ---

def write_dataframe(df, filename, cache_location):
    """Writes a DataFrame to GCS or local cache."""
    if cache_location == 'gcp':
        print(f"Writing DataFrame '{filename}' to GCS bucket '{GCS_BUCKET_NAME}'...")
        blob = _get_gcs_blob(filename)
        buffer = BytesIO()
        df.to_pickle(buffer)
        buffer.seek(0)
        blob.upload_from_file(buffer, content_type='application/octet-stream')
        print(f"Write successful to gs://{GCS_BUCKET_NAME}/{blob.name}")
    elif cache_location == 'local':
        local_path = _get_local_path(filename)
        print(f"Writing DataFrame '{filename}' to local cache: {local_path}...")
        df.to_pickle(local_path)
        print("Write successful.")
    else:
        raise ValueError("Invalid cache_location. Use 'gcp' or 'local'.")

def write_json(data, filename, cache_location):
    """Writes a dictionary/list to a JSON file in GCS or local cache."""
    if cache_location == 'gcp':
        print(f"Writing JSON '{filename}' to GCS bucket '{GCS_BUCKET_NAME}'...")
        blob = _get_gcs_blob(filename)
        blob.upload_from_string(json.dumps(data, indent=2), content_type='application/json')
        print(f"Write successful to gs://{GCS_BUCKET_NAME}/{blob.name}")
    elif cache_location == 'local':
        local_path = _get_local_path(filename)
        print(f"Writing JSON '{filename}' to local cache: {local_path}...")
        with open(local_path, 'w') as f:
            json.dump(data, f, indent=2)
        print("Write successful.")
    else:
        raise ValueError("Invalid cache_location. Use 'gcp' or 'local'.")

def save_joblib(model, filename, cache_location):
    """Saves a joblib file (like a model) to GCS or local cache."""
    if cache_location == 'gcp':
        print(f"Saving joblib file '{filename}' to GCS bucket '{GCS_BUCKET_NAME}'...")
        blob = _get_gcs_blob(filename)
        buffer = BytesIO()
        joblib.dump(model, buffer)
        buffer.seek(0)
        blob.upload_from_file(buffer, content_type='application/octet-stream')
        print(f"Save successful to gs://{GCS_BUCKET_NAME}/{blob.name}")
    elif cache_location == 'local':
        local_path = _get_local_path(filename)
        print(f"Saving joblib file '{filename}' to local cache: {local_path}...")
        joblib.dump(model, local_path)
        print("Save successful.")
    else:
        raise ValueError("Invalid cache_location. Use 'gcp' or 'local'.")

def save_plot(figure, filename, cache_location):
    """Saves a matplotlib figure to GCS or local cache."""
    if cache_location == 'gcp':
        print(f"Saving plot '{filename}' to GCS bucket '{GCS_BUCKET_NAME}'...")
        blob = _get_gcs_blob(filename)
        buffer = BytesIO()
        figure.savefig(buffer, format='png', bbox_inches='tight')
        buffer.seek(0)
        blob.upload_from_file(buffer, content_type='image/png')
        print(f"Save successful to gs://{GCS_BUCKET_NAME}/{blob.name}")
    elif cache_location == 'local':
        local_path = _get_local_path(filename)
        print(f"Saving plot '{filename}' to local cache: {local_path}...")
        figure.savefig(local_path, bbox_inches='tight')
        print("Save successful.")
    else:
        raise ValueError("Invalid cache_location. Use 'gcp' or 'local'.")
    plt.close(figure) # Close the figure after saving

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
