import functions_framework
from flask import Response
# Only import storage if needed, or handle potential import error if not running locally
# from google.cloud import storage
from flask import Response, Flask, request
from flask_cors import CORS
import json
import datetime
import uuid
import logging
import os # Import os

app = Flask(__name__)
CORS(app)

# Determine RUNNING_LOCALLY based on an environment variable (more robust)
# GCF sets specific env vars, e.g., FUNCTION_TARGET, K_SERVICE
RUNNING_LOCALLY = not os.environ.get('K_SERVICE', '')

def add_cors_headers(response, origin='*'):
    """
    Add CORS headers to a Flask response object
    
    Args:
        response: Flask Response object
        origin: Origin to allow, defaults to '*'
        
    Returns:
        Flask Response object with CORS headers added
    """
    response.headers.add('Access-Control-Allow-Origin', origin)
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,x-access-token')
    response.headers.add('Access-Control-Allow-Methods', 'GET,POST,OPTIONS')
    response.headers.add('Access-Control-Allow-Credentials', 'true')
    return response

@functions_framework.http
def hello_http(request):
    # Get the origin from request headers
    origin = request.headers.get('Origin', '*')

    # --- Defer GCS client initialization ---
    # storage_client = None
    # bucket = None
    # if not RUNNING_LOCALLY:
    #     try:
    #         from google.cloud import storage # Import here
    #         storage_client = storage.Client()
    #         bucket_name = "cryptomancer"  # Replace with your bucket name
    #         bucket = storage_client.bucket(bucket_name)
    #     except Exception as e:
    #         logging.error(f"Failed to initialize GCS client: {str(e)}")
    #         # Decide how to handle this - maybe proceed without GCS logging?

    # Generate a timestamp for the filename
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    short_uuid = str(uuid.uuid4())[:8]

    # Extract the function name from the request URL, ignoring query parameters
    path_without_params = request.path.split('?')[0]
    function_name = path_without_params.strip('/').split('/')[-1]

    # Define a dictionary mapping function names to modules
    functions = {
        "coindesk_batch_update_exchange_rates": "coindesk_rates.batch_update_bigquery_tables",
        "coindesk_backfill_update_exchange_rates": "coindesk_rates.batch_backfill_bigquery_tables",
        "model_run_refresh_data": "model_run.refresh_data",
        "model_run_update_features": "model_run.update_features",
        "model_run_training": "model_run.run_training",
        "model_run_simulation": "model_run.run_simulation",
        "fetch_and_save_minute_data_for_hour": "data_utils.api_fetch_and_save_minute_data_for_hour",
        "run_llm_and_save_to_bigquery": "data_utils.run_llm_and_save_to_bigquery",
        "calculate_actuals": "data_utils.api_calculate_actuals"
    }

    # Log the request details to the GCS bucket as early as possible
    if not RUNNING_LOCALLY:
        try:
            # --- Initialize GCS client ONLY if needed ---
            from google.cloud import storage # Import here
            storage_client = storage.Client()
            bucket_name = "cryptomancer"
            bucket = storage_client.bucket(bucket_name)
            # --- End GCS initialization ---

            folder_name = "requests"
            filename = f"request_{timestamp}_{short_uuid}.json"
            blob = bucket.blob(f"{folder_name}/{filename}")

            # Create a dictionary to hold the request data
            request_data = {
                "method": request.method,
                "path": request.path,
                "headers": dict(request.headers),
                "query_parameters": dict(request.args.to_dict()),
                "timestamp": timestamp
            }

            # For POST requests, include the JSON body
            if request.method == 'POST':
                 # Use get_data() and decode for robustness if get_json fails
                 try:
                     request_data["json"] = request.get_json()
                 except Exception:
                     request_data["body_raw"] = request.get_data(as_text=True)


            # Save the request data to the GCS bucket
            blob.upload_from_string(
                data=json.dumps(request_data, indent=2), # Add indent for readability
                content_type='application/json'
            )
        except Exception as e:
            # Log an error if saving the request fails
            logging.error(f"Failed to log request to GCS: {str(e)}")

    # Proceed with the rest of the function
    try:
        if request.method == 'GET':
            if function_name in functions:
                module_name = functions[function_name]
                module_name, function_name = module_name.rsplit(".", 1)
                imported_module = __import__(module_name)
                function = getattr(imported_module, function_name)

                # --- Execute the target function and unpack the tuple ---
                response_tuple = function(request)
                if isinstance(response_tuple, tuple) and len(response_tuple) == 2:
                    response_data, status_code = response_tuple
                else:
                    # Assume success if not a tuple (or handle as error)
                    response_data = response_tuple
                    status_code = 200

                # --- Create Flask Response ---
                if isinstance(response_data, Response): # If function returned a Flask Response directly
                    return add_cors_headers(response_data, origin)
                if isinstance(response_data, str): # Handle plain string/HTML
                    html_response = Response(response_data, status=status_code, mimetype='text/html')
                    return add_cors_headers(html_response, origin)
                # Default to JSON
                json_response = Response(json.dumps(response_data), status=status_code, mimetype='application/json')
                return add_cors_headers(json_response, origin)
            else:
                error_data = {"error": "Function not found"}
                status_code = 404
                error_response = Response(json.dumps(error_data), status=status_code, mimetype='application/json')
                return add_cors_headers(error_response, origin)

        elif request.method == 'POST':
            if function_name in functions:
                module_name = functions[function_name]
                module_name, function_name = module_name.rsplit(".", 1)
                imported_module = __import__(module_name)
                function = getattr(imported_module, function_name)

                # --- Execute the target function and unpack the tuple ---
                response_tuple = function(request)
                if isinstance(response_tuple, tuple) and len(response_tuple) == 2:
                    response_data, status_code = response_tuple
                else:
                    # Handle case where function might not return a tuple (e.g., error occurred before return)
                    # Or if you change functions to only return dicts later
                    response_data = response_tuple # Assume it's the data dictionary
                    status_code = 200 # Assume success if only data is returned

                # --- Log response_data to GCS (if applicable) ---
                if not RUNNING_LOCALLY:
                    try:
                        # --- Initialize GCS client ONLY if needed ---
                        from google.cloud import storage
                        storage_client = storage.Client()
                        bucket_name = "cryptomancer"
                        bucket = storage_client.bucket(bucket_name)
                        # --- End GCS initialization ---

                        folder_name = "responses"
                        filename = f"response_{timestamp}_{short_uuid}.json"
                        blob = bucket.blob(f"{folder_name}/{filename}")
                        # Log the actual data part, not the tuple
                        blob.upload_from_string(json.dumps(response_data, indent=2), content_type='application/json')
                    except Exception as e:
                        logging.error(f"Failed to log response to GCS: {str(e)}")

                # --- Prepare and return the actual HTTP response ---
                json_response = Response(json.dumps(response_data), status=status_code, mimetype='application/json')
                return add_cors_headers(json_response, origin)
            else:
                error_data = {"error": "Function not found"}
                status_code = 404
                error_response = Response(json.dumps(error_data), status=status_code, mimetype='application/json')
                return add_cors_headers(error_response, origin)
        else:
             # Handle other methods like OPTIONS
             return add_cors_headers(Response(status=200), origin)

    except Exception as e:
        logging.exception(f"Error processing request: {str(e)}")
        # --- Return error dictionary and 500 status ---
        error_data = {"error": str(e)}
        status_code = 500
        error_response = Response(json.dumps(error_data), status=status_code, mimetype='application/json')
        return add_cors_headers(error_response, origin)

# For local execution
@app.route('/<path:path>', methods=['GET', 'POST', 'OPTIONS'])
def local_handler(path):
    """
    This route captures all paths when running locally via Flask
    and forwards the request to the hello_http function,
    mimicking the Cloud Functions behavior.
    """
    # The 'request' object is globally available in Flask's context
    # and will be correctly passed to hello_http.
    return hello_http(request._get_current_object())


# For local execution
if __name__ == '__main__':
    # Make sure Flask and request are imported if not already
    # from flask import Flask, request
    # RUNNING_LOCALLY is now set based on env var at the top
    if RUNNING_LOCALLY:
        print("Running locally...")
        try:
            from test import setup_environment
            setup_environment() # Sets GOOGLE_APPLICATION_CREDENTIALS if needed
            print("Local environment setup complete.")
        except ImportError:
            print("Warning: test.py or setup_environment not found. Skipping local setup.")
        except Exception as e:
            print(f"Warning: Error during local setup: {e}")

    app.run(debug=True, port=5000, use_reloader=False)