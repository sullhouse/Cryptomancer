import functions_framework
from flask import Response
from google.cloud import storage
from flask import Response, Flask, request
from flask_cors import CORS
import json
import datetime
import uuid
import logging

app = Flask(__name__)
CORS(app)

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

    # Get a reference to the GCS bucket and folder
    bucket_name = "cryptomancer"  # Replace with your bucket name
    folder_name = "requests"
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)

    # Generate a timestamp for the filename
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    short_uuid = str(uuid.uuid4())[:8]

    # Create a filename with the timestamp
    filename = f"request_{timestamp}_{short_uuid}.json"

    # Construct the full path within the bucket
    blob = bucket.blob(f"{folder_name}/{filename}")

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
        "model_run_simulation": "model_run.run_simulation"
    }

    # Log the request details to the GCS bucket as early as possible
    try:
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
            request_data["json"] = request.get_json()

        # Save the request data to the GCS bucket
        blob.upload_from_string(
            data=json.dumps(request_data),
            content_type='application/json'
        )
    except Exception as e:
        # Log an error if saving the request fails
        logging.error(f"Failed to log request to GCS: {str(e)}")

    # Proceed with the rest of the function
    try:
        if request.method == 'GET':
            # Handle GET requests
            if function_name in functions:
                module_name = functions[function_name]
                module_name, function_name = module_name.rsplit(".", 1)
                imported_module = __import__(module_name)
                function = getattr(imported_module, function_name)
                response = function(request)

                if isinstance(response, Response):
                    return add_cors_headers(response, origin)
                if isinstance(response, str):
                    html_response = Response(response, status=200, mimetype='text/html')
                    return add_cors_headers(html_response, origin)
                json_response = Response(json.dumps(response), status=200, mimetype='application/json')
                return add_cors_headers(json_response, origin)
            else:
                error_response = Response(json.dumps({"error": "Function not found"}), status=404, mimetype='application/json')
                return add_cors_headers(error_response, origin)
        else:
            # Handle POST requests
            if function_name in functions:
                module_name = functions[function_name]
                module_name, function_name = module_name.rsplit(".", 1)
                imported_module = __import__(module_name)
                function = getattr(imported_module, function_name)
                response = function(request)

                folder_name = "responses"
                try:
                    filename = f"response_{timestamp}_{short_uuid}.json"
                    blob = bucket.blob(f"{folder_name}/{filename}")
                    blob.upload_from_string(json.dumps(response, indent=2))
                except Exception as e:
                    error_response = Response(json.dumps({"error": str(e)}), status=500, mimetype='application/json')
                    return add_cors_headers(error_response, origin)

                json_response = Response(json.dumps(response), status=200, mimetype='application/json')
                return add_cors_headers(json_response, origin)
            else:
                error_response = Response(json.dumps({"error": "Function not found"}), status=404, mimetype='application/json')
                return add_cors_headers(error_response, origin)
    except Exception as e:
        error_response = Response(json.dumps({"error": str(e)}), status=500, mimetype='application/json')
        return add_cors_headers(error_response, origin)