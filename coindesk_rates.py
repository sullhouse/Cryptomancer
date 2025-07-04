import requests
import csv
from datetime import datetime, timezone, timedelta
from google.cloud import bigquery
from discord_messenger import send_ingestion_log, send_error_log
import json
import time # Add time import for potential delays

def fetch_historical_exchange_rates(
    market, instrument, limit, aggregate, fill, apply_mapping, response_format, api_key
):
    """
    Fetch historical exchange rates from the Coindesk API.

    Args:
        market (str): The market to query (e.g., "cadli").
        instrument (str): The trading pair (e.g., "ETH-BTC").
        limit (int): The total number of data points to fetch.
        aggregate (int): The aggregation interval (e.g., 1 for minute-level data).
        fill (bool): Whether to fill missing data points.
        apply_mapping (bool): Whether to apply mapping to the data.
        response_format (str): The response format (e.g., "JSON").
        api_key (str): Your Coindesk API key.

    Returns:
        list: A list of dictionaries containing the historical exchange rates.
    """
    base_url = "https://data-api.coindesk.com/index/cc/v1/historical/minutes"
    all_rates = []  # To store all the results across multiple queries
    to_ts = None  # Start with no `to_ts` parameter

    try:
        while limit > 0:
            # Determine the number of results to fetch in this query
            query_limit = min(limit, 2000)

            # Prepare the query parameters
            params = {
                "market": market,
                "instrument": instrument,
                "limit": query_limit,
                "aggregate": aggregate,
                "fill": str(fill).lower(),
                "apply_mapping": str(apply_mapping).lower(),
                "response_format": response_format,
                "api_key": api_key,
            }
            if to_ts:
                params["to_ts"] = to_ts

            # Make the GET request to the Coindesk API
            response = requests.get(base_url, params=params)
            response.raise_for_status()  # Raise an exception for HTTP errors

            # Parse the JSON response
            data = response.json()
            rates = data.get("Data", [])  # Correctly reference the "Data" array

            if not rates:
                print("No more data available.")
                break

            # Append the results to the all_rates list
            all_rates.extend(rates)

            # Update the `to_ts` for the next query
            earliest_timestamp = min(rate["TIMESTAMP"] for rate in rates)  # Find the earliest timestamp
            to_ts = earliest_timestamp - 60  # Subtract 60 seconds to avoid overlap

            # Decrease the remaining limit
            limit -= query_limit
            print(f"Fetched {len(rates)} records, {limit} remaining.")

        return all_rates

    except requests.exceptions.RequestException as e:
        error_message = f"Error fetching data from Coindesk API: {e}"
        print(error_message)
        send_error_log("fetch_historical_exchange_rates", error_message)
        return []
    except Exception as e:
        error_message = f"An unexpected error occurred: {e}"
        print(error_message)
        send_error_log("fetch_historical_exchange_rates", error_message)
        return []

def fetch_historical_range(
    market, instrument, start_ts, end_ts, aggregate, fill, apply_mapping, response_format, api_key
):
    """
    Fetch historical exchange rates from the Coindesk API for a specific date range.

    Args:
        market (str): The market to query.
        instrument (str): The trading pair.
        start_ts (int): Start timestamp (Unix seconds).
        end_ts (int): End timestamp (Unix seconds).
        aggregate (int): Aggregation interval (minutes).
        fill (bool): Whether to fill missing data points.
        apply_mapping (bool): Whether to apply mapping.
        response_format (str): Response format.
        api_key (str): Coindesk API key.

    Returns:
        list: A list of dictionaries containing the historical exchange rates within the range.
    """
    base_url = "https://data-api.coindesk.com/index/cc/v1/historical/minutes"
    all_rates = []
    current_ts = start_ts
    max_limit_per_call = 2000 # Coindesk API limit per request

    print(f"Starting fetch for {instrument} from {datetime.utcfromtimestamp(start_ts)} to {datetime.utcfromtimestamp(end_ts)}")

    try:
        while current_ts < end_ts:
            # Calculate the end timestamp for this specific API call batch
            # Fetch up to max_limit_per_call minutes ahead, but don't exceed the overall end_ts
            batch_end_ts = min(current_ts + (max_limit_per_call * aggregate * 60) - (aggregate * 60), end_ts)

            # Prepare the query parameters for fetching forward
            params = {
                "market": market,
                "instrument": instrument,
                "from_ts": current_ts,
                "to_ts": batch_end_ts, # Fetch up to this timestamp
                "limit": max_limit_per_call, # Request the max allowed
                "aggregate": aggregate,
                "fill": str(fill).lower(),
                "apply_mapping": str(apply_mapping).lower(),
                "response_format": response_format,
                "api_key": api_key,
            }

            print(f"Fetching batch for {instrument}: {datetime.utcfromtimestamp(current_ts)} to {datetime.utcfromtimestamp(batch_end_ts)} (Limit: {max_limit_per_call})")

            # Make the GET request
            response = requests.get(base_url, params=params)
            response.raise_for_status()

            data = response.json()
            rates = data.get("Data", [])

            if not rates:
                print(f"No more data received for {instrument} in this batch, or reached end_ts.")
                # If no data, advance current_ts past batch_end_ts to avoid infinite loop if there's a gap
                current_ts = batch_end_ts + (aggregate * 60)
                # Optional: Add a small delay if hitting API limits frequently
                # time.sleep(1)
                continue # Continue to next potential batch

            # Filter rates to ensure they are strictly within the requested overall range [start_ts, end_ts]
            # and sort them by timestamp to process in order
            rates_in_range = sorted(
                [rate for rate in rates if start_ts <= rate.get("TIMESTAMP", 0) <= end_ts],
                key=lambda x: x.get("TIMESTAMP", 0)
            )

            if not rates_in_range:
                 print(f"No data within the desired range [{start_ts}, {end_ts}] in this batch for {instrument}.")
                 # Advance past this batch even if results were outside the final range
                 current_ts = batch_end_ts + (aggregate * 60)
                 continue

            all_rates.extend(rates_in_range)

            # Update current_ts to the timestamp of the *last* record received in this batch + interval
            # This prepares 'from_ts' for the next iteration
            last_timestamp_in_batch = rates_in_range[-1].get("TIMESTAMP", 0)
            current_ts = last_timestamp_in_batch + (aggregate * 60)

            print(f"Fetched {len(rates_in_range)} records for {instrument}. Last timestamp: {datetime.utcfromtimestamp(last_timestamp_in_batch)}. Next fetch starts from: {datetime.utcfromtimestamp(current_ts)}")

            # Optional: Add a small delay between API calls to respect rate limits
            # time.sleep(0.5) # Adjust as needed

        print(f"Finished fetching for {instrument}. Total records fetched: {len(all_rates)}")
        # Remove potential duplicates just in case API overlap occurs (though pagination logic tries to avoid it)
        unique_rates = list({rate['TIMESTAMP']: rate for rate in all_rates}.values())
        unique_rates.sort(key=lambda x: x.get("TIMESTAMP", 0)) # Sort again after deduplication
        print(f"Total unique records for {instrument}: {len(unique_rates)}")
        return unique_rates

    except requests.exceptions.RequestException as e:
        error_message = f"Error fetching data range from Coindesk API for {instrument}: {e}"
        print(error_message)
        send_error_log("fetch_historical_range", error_message)
        return [] # Return empty list on error, consider partial results if needed
    except Exception as e:
        error_message = f"An unexpected error occurred during range fetch for {instrument}: {e}"
        print(error_message)
        send_error_log("fetch_historical_range", error_message)
        return []

def save_rates_to_csv(rates, output_file):
    """
    Save historical exchange rates to a CSV file.

    Args:
        rates (list): A list of dictionaries containing the historical exchange rates.
        output_file (str): The name of the output CSV file.

    Returns:
        None
    """
    try:
        with open(output_file, mode="w", newline="") as csv_file:
            writer = csv.writer(csv_file)

            # Write the header row
            header = [
                "UNIT", "TIMESTAMP", "TYPE", "MARKET", "INSTRUMENT", "OPEN", "HIGH", "LOW", "CLOSE",
                "FIRST_MESSAGE_TIMESTAMP", "LAST_MESSAGE_TIMESTAMP", "FIRST_MESSAGE_VALUE",
                "HIGH_MESSAGE_VALUE", "HIGH_MESSAGE_TIMESTAMP", "LOW_MESSAGE_VALUE",
                "LOW_MESSAGE_TIMESTAMP", "LAST_MESSAGE_VALUE", "TOTAL_INDEX_UPDATES", "VOLUME",
                "QUOTE_VOLUME", "VOLUME_TOP_TIER", "QUOTE_VOLUME_TOP_TIER", "VOLUME_DIRECT",
                "QUOTE_VOLUME_DIRECT", "VOLUME_TOP_TIER_DIRECT", "QUOTE_VOLUME_TOP_TIER_DIRECT"
            ]
            writer.writerow(header)

            # Write each entry to the CSV
            for entry in rates:
                writer.writerow([
                    entry.get("UNIT", ""),
                    datetime.utcfromtimestamp(entry.get("TIMESTAMP", 0)).strftime("%Y-%m-%d %H:%M:%S UTC") if entry.get("TIMESTAMP") else "",
                    entry.get("TYPE", ""),
                    entry.get("MARKET", ""),
                    entry.get("INSTRUMENT", ""),
                    entry.get("OPEN", ""),
                    entry.get("HIGH", ""),
                    entry.get("LOW", ""),
                    entry.get("CLOSE", ""),
                    datetime.utcfromtimestamp(entry.get("FIRST_MESSAGE_TIMESTAMP", 0)).strftime("%Y-%m-%d %H:%M:%S UTC") if entry.get("FIRST_MESSAGE_TIMESTAMP") else "",
                    datetime.utcfromtimestamp(entry.get("LAST_MESSAGE_TIMESTAMP", 0)).strftime("%Y-%m-%d %H:%M:%S UTC") if entry.get("LAST_MESSAGE_TIMESTAMP") else "",
                    entry.get("FIRST_MESSAGE_VALUE", ""),
                    entry.get("HIGH_MESSAGE_VALUE", ""),
                    datetime.utcfromtimestamp(entry.get("HIGH_MESSAGE_TIMESTAMP", 0)).strftime("%Y-%m-%d %H:%M:%S UTC") if entry.get("HIGH_MESSAGE_TIMESTAMP") else "",
                    entry.get("LOW_MESSAGE_VALUE", ""),
                    datetime.utcfromtimestamp(entry.get("LOW_MESSAGE_TIMESTAMP", 0)).strftime("%Y-%m-%d %H:%M:%S UTC") if entry.get("LOW_MESSAGE_TIMESTAMP") else "",
                    entry.get("LAST_MESSAGE_VALUE", ""),
                    entry.get("TOTAL_INDEX_UPDATES", ""),
                    entry.get("VOLUME", ""),
                    entry.get("QUOTE_VOLUME", ""),
                    entry.get("VOLUME_TOP_TIER", ""),
                    entry.get("QUOTE_VOLUME_TOP_TIER", ""),
                    entry.get("VOLUME_DIRECT", ""),
                    entry.get("QUOTE_VOLUME_DIRECT", ""),
                    entry.get("VOLUME_TOP_TIER_DIRECT", ""),
                    entry.get("QUOTE_VOLUME_TOP_TIER_DIRECT", "")
                ])
        print(f"Historical exchange rates successfully saved to {output_file}")
    except Exception as e:
        error_message = f"Error saving rates to CSV: {e}"
        print(error_message)
        send_error_log("save_rates_to_csv", error_message)

def save_rates_to_bigquery(rates, project_id, dataset_id, table_id):
    """
    Save historical exchange rates to a BigQuery table using a MERGE statement to prevent duplicates.

    Args:
        rates (list): A list of dictionaries containing the historical exchange rates.
        project_id (str): The GCP project ID.
        dataset_id (str): The BigQuery dataset ID.
        table_id (str): The BigQuery table ID.

    Returns:
        None
    """
    try:
        client = bigquery.Client(project=project_id)
        table_ref = f"{project_id}.{dataset_id}.{table_id}"

        # Define the schema explicitly to match the target table
        schema = [
            bigquery.SchemaField("UNIT", "STRING"),
            bigquery.SchemaField("TIMESTAMP", "TIMESTAMP"),
            bigquery.SchemaField("TYPE", "INTEGER"),
            bigquery.SchemaField("MARKET", "STRING"),
            bigquery.SchemaField("INSTRUMENT", "STRING"),
            bigquery.SchemaField("OPEN", "FLOAT"),
            bigquery.SchemaField("HIGH", "FLOAT"),
            bigquery.SchemaField("LOW", "FLOAT"),
            bigquery.SchemaField("CLOSE", "FLOAT"),
            bigquery.SchemaField("FIRST_MESSAGE_TIMESTAMP", "TIMESTAMP"),
            bigquery.SchemaField("LAST_MESSAGE_TIMESTAMP", "TIMESTAMP"),
            bigquery.SchemaField("FIRST_MESSAGE_VALUE", "FLOAT"),
            bigquery.SchemaField("HIGH_MESSAGE_VALUE", "FLOAT"),
            bigquery.SchemaField("HIGH_MESSAGE_TIMESTAMP", "TIMESTAMP"),
            bigquery.SchemaField("LOW_MESSAGE_VALUE", "FLOAT"),
            bigquery.SchemaField("LOW_MESSAGE_TIMESTAMP", "TIMESTAMP"),
            bigquery.SchemaField("LAST_MESSAGE_VALUE", "FLOAT"),
            bigquery.SchemaField("TOTAL_INDEX_UPDATES", "INTEGER"),
            bigquery.SchemaField("VOLUME", "FLOAT"),
            bigquery.SchemaField("QUOTE_VOLUME", "FLOAT"),
            bigquery.SchemaField("VOLUME_TOP_TIER", "FLOAT"),
            bigquery.SchemaField("QUOTE_VOLUME_TOP_TIER", "FLOAT"),
            bigquery.SchemaField("VOLUME_DIRECT", "INTEGER"),
            bigquery.SchemaField("QUOTE_VOLUME_DIRECT", "INTEGER"),
            bigquery.SchemaField("VOLUME_TOP_TIER_DIRECT", "INTEGER"),
            bigquery.SchemaField("QUOTE_VOLUME_TOP_TIER_DIRECT", "INTEGER"),
            bigquery.SchemaField("PCT_CHANGE", "FLOAT"),
            bigquery.SchemaField("VOLATILITY", "FLOAT"),
            bigquery.SchemaField("SPREAD_CLOSE_OPEN", "FLOAT"),
            bigquery.SchemaField("VOLUME_RATIO", "FLOAT"),
            bigquery.SchemaField("VOLUME_TOP_TIER_RATIO", "FLOAT"),
        ]

        # Convert rates to BigQuery-compatible rows
        rows_to_insert = []
        for entry in rates:
            open_price = entry.get("OPEN", 0) or 0
            close_price = entry.get("CLOSE", 0) or 0
            high_price = entry.get("HIGH", 0) or 0
            low_price = entry.get("LOW", 0) or 0
            quote_volume = entry.get("QUOTE_VOLUME", 0) or 0
            volume = entry.get("VOLUME", 0) or 0
            quote_volume_top = entry.get("QUOTE_VOLUME_TOP_TIER", 0) or 0
            volume_top = entry.get("VOLUME_TOP_TIER", 0) or 0
            rows_to_insert.append({
                "UNIT": str(entry.get("UNIT", "")),
                "TIMESTAMP": datetime.utcfromtimestamp(entry.get("TIMESTAMP", 0)).strftime("%Y-%m-%d %H:%M:%S UTC") if entry.get("TIMESTAMP") else None,
                "TYPE": entry.get("TYPE", ""),
                "MARKET": entry.get("MARKET", ""),
                "INSTRUMENT": entry.get("INSTRUMENT", ""),
                "OPEN": entry.get("OPEN", ""),
                "HIGH": entry.get("HIGH", ""),
                "LOW": entry.get("LOW", ""),
                "CLOSE": entry.get("CLOSE", ""),
                "FIRST_MESSAGE_TIMESTAMP": datetime.utcfromtimestamp(entry.get("FIRST_MESSAGE_TIMESTAMP", 0)).strftime("%Y-%m-%d %H:%M:%S UTC") if entry.get("FIRST_MESSAGE_TIMESTAMP") else None,
                "LAST_MESSAGE_TIMESTAMP": datetime.utcfromtimestamp(entry.get("LAST_MESSAGE_TIMESTAMP", 0)).strftime("%Y-%m-%d %H:%M:%S UTC") if entry.get("LAST_MESSAGE_TIMESTAMP") else None,
                "FIRST_MESSAGE_VALUE": entry.get("FIRST_MESSAGE_VALUE", ""),
                "HIGH_MESSAGE_VALUE": entry.get("HIGH_MESSAGE_VALUE", ""),
                "HIGH_MESSAGE_TIMESTAMP": datetime.utcfromtimestamp(entry.get("HIGH_MESSAGE_TIMESTAMP", 0)).strftime("%Y-%m-%d %H:%M:%S UTC") if entry.get("HIGH_MESSAGE_TIMESTAMP") else None,
                "LOW_MESSAGE_VALUE": entry.get("LOW_MESSAGE_VALUE", ""),
                "LOW_MESSAGE_TIMESTAMP": datetime.utcfromtimestamp(entry.get("LOW_MESSAGE_TIMESTAMP", 0)).strftime("%Y-%m-%d %H:%M:%S UTC") if entry.get("LOW_MESSAGE_TIMESTAMP") else None,
                "LAST_MESSAGE_VALUE": entry.get("LAST_MESSAGE_VALUE", ""),
                "TOTAL_INDEX_UPDATES": entry.get("TOTAL_INDEX_UPDATES", 0),
                "VOLUME": entry.get("VOLUME", ""),
                "QUOTE_VOLUME": entry.get("QUOTE_VOLUME", ""),
                "VOLUME_TOP_TIER": entry.get("VOLUME_TOP_TIER", ""),
                "QUOTE_VOLUME_TOP_TIER": entry.get("QUOTE_VOLUME_TOP_TIER", ""),
                "VOLUME_DIRECT": entry.get("VOLUME_DIRECT", ""),
                "QUOTE_VOLUME_DIRECT": entry.get("QUOTE_VOLUME_DIRECT", ""),
                "VOLUME_TOP_TIER_DIRECT": entry.get("VOLUME_TOP_TIER_DIRECT", ""),
                "QUOTE_VOLUME_TOP_TIER_DIRECT": entry.get("QUOTE_VOLUME_TOP_TIER_DIRECT", ""),
                "PCT_CHANGE": (close_price - open_price) / open_price if open_price else None,
                "VOLATILITY": high_price - low_price,
                "SPREAD_CLOSE_OPEN": close_price - open_price,
                "VOLUME_RATIO": quote_volume / volume if volume else None,
                "VOLUME_TOP_TIER_RATIO": quote_volume_top / volume_top if volume_top else None
            })

        # Load the data into a temporary table with the explicit schema
        temp_table_id = f"{dataset_id}.temp_{table_id}"
        job_config = bigquery.LoadJobConfig(
            schema=schema,
            write_disposition="WRITE_TRUNCATE"
        )
        job = client.load_table_from_json(rows_to_insert, temp_table_id, job_config=job_config)
        job.result()  # Wait for the load job to complete

        # Use a MERGE statement to insert only new rows
        merge_query = f"""
            MERGE `{table_ref}` T
            USING `{temp_table_id}` S
            ON T.TIMESTAMP = S.TIMESTAMP
            WHEN NOT MATCHED THEN
              INSERT ROW
        """
        query_job = client.query(merge_query)
        query_job.result()  # Wait for the query to complete

        # Clean up the temporary table
        client.delete_table(temp_table_id, not_found_ok=True)

        print(f"Historical exchange rates successfully saved to BigQuery table {dataset_id}.{table_id}")
    except Exception as e:
        error_message = f"Error saving rates to BigQuery: {e}"
        print(error_message)
        send_error_log("save_rates_to_bigquery", error_message)

def update_bigquery_table(project_id, dataset_id, instrument, market, aggregate, fill, apply_mapping, response_format, api_key):
    """
    Update a BigQuery table with up-to-date exchange rates for a given instrument.

    Args:
        project_id (str): The GCP project ID.
        dataset_id (str): The BigQuery dataset ID.
        instrument (str): The trading pair (e.g., "ETH-BTC").
        market (str): The market to query (e.g., "cadli").
        aggregate (int): The aggregation interval (e.g., 1 for minute-level data).
        fill (bool): Whether to fill missing data points.
        apply_mapping (bool): Whether to apply mapping to the data.
        response_format (str): The response format (e.g., "JSON").
        api_key (str): Your Coindesk API key.

    Returns:
        None
    """
    try:
        client = bigquery.Client(project=project_id)
        table_id = f"{project_id}.{dataset_id}.{instrument.lower()}"  # BigQuery table name

        # Define the schema explicitly to match the target table
        schema = [
            bigquery.SchemaField("UNIT", "STRING"),
            bigquery.SchemaField("TIMESTAMP", "TIMESTAMP"),
            bigquery.SchemaField("TYPE", "INTEGER"),
            bigquery.SchemaField("MARKET", "STRING"),
            bigquery.SchemaField("INSTRUMENT", "STRING"),
            bigquery.SchemaField("OPEN", "FLOAT"),
            bigquery.SchemaField("HIGH", "FLOAT"),
            bigquery.SchemaField("LOW", "FLOAT"),
            bigquery.SchemaField("CLOSE", "FLOAT"),
            bigquery.SchemaField("FIRST_MESSAGE_TIMESTAMP", "TIMESTAMP"),
            bigquery.SchemaField("LAST_MESSAGE_TIMESTAMP", "TIMESTAMP"),
            bigquery.SchemaField("FIRST_MESSAGE_VALUE", "FLOAT"),
            bigquery.SchemaField("HIGH_MESSAGE_VALUE", "FLOAT"),
            bigquery.SchemaField("HIGH_MESSAGE_TIMESTAMP", "TIMESTAMP"),
            bigquery.SchemaField("LOW_MESSAGE_VALUE", "FLOAT"),
            bigquery.SchemaField("LOW_MESSAGE_TIMESTAMP", "TIMESTAMP"),
            bigquery.SchemaField("LAST_MESSAGE_VALUE", "FLOAT"),
            bigquery.SchemaField("TOTAL_INDEX_UPDATES", "INTEGER"),
            bigquery.SchemaField("VOLUME", "FLOAT"),
            bigquery.SchemaField("QUOTE_VOLUME", "FLOAT"),
            bigquery.SchemaField("VOLUME_TOP_TIER", "FLOAT"),
            bigquery.SchemaField("QUOTE_VOLUME_TOP_TIER", "FLOAT"),
            bigquery.SchemaField("VOLUME_DIRECT", "INTEGER"),
            bigquery.SchemaField("QUOTE_VOLUME_DIRECT", "INTEGER"),
            bigquery.SchemaField("VOLUME_TOP_TIER_DIRECT", "INTEGER"),
            bigquery.SchemaField("QUOTE_VOLUME_TOP_TIER_DIRECT", "INTEGER"),
            bigquery.SchemaField("PCT_CHANGE", "FLOAT"),
            bigquery.SchemaField("VOLATILITY", "FLOAT"),
            bigquery.SchemaField("SPREAD_CLOSE_OPEN", "FLOAT"),
            bigquery.SchemaField("VOLUME_RATIO", "FLOAT"),
            bigquery.SchemaField("VOLUME_TOP_TIER_RATIO", "FLOAT"),
        ]

        # Check if the table exists
        try:
            client.get_table(table_id)
            print(f"Table {table_id} exists.")
        except Exception:
            # If the table doesn't exist, create it
            print(f"Table {table_id} does not exist. Creating it...")
            table = bigquery.Table(table_id, schema=schema)
            client.create_table(table)
            print(f"Table {table_id} created successfully.")

        # Query the BigQuery table to find the most recent TIMESTAMP
        query = f"""
            SELECT MAX(TIMESTAMP) AS latest_timestamp
            FROM `{table_id}`
        """
        query_job = client.query(query)
        result = query_job.result()
        latest_timestamp_row = next(result, None)

        # Determine the most recent timestamp in the table
        if latest_timestamp_row and latest_timestamp_row["latest_timestamp"]:
            latest_timestamp = latest_timestamp_row["latest_timestamp"]
        else:
            # If the table is empty, start from the beginning of the year
            now = datetime.now(timezone.utc)
            start_of_year = datetime(year=now.year, month=1, day=1, tzinfo=timezone.utc)
            latest_timestamp = start_of_year

        # Calculate the number of missing records (minutes) between the latest timestamp and now
        now = datetime.now(timezone.utc)
        delta = now - latest_timestamp
        limit = int(delta.total_seconds() // 60)  # Convert seconds to minutes

        if limit <= 0:
            print("The table is already up-to-date.")
            return

        print(f"Fetching {limit} missing records for {instrument}...")

        # Fetch the missing records from the Coindesk API
        rates = fetch_historical_exchange_rates(
            market=market,
            instrument=instrument,
            limit=limit,
            aggregate=aggregate,
            fill=fill,
            apply_mapping=apply_mapping,
            response_format=response_format,
            api_key=api_key
        )

        if not rates:
            print(f"No new data available for {instrument}.")
            return

        # Insert the new records into the BigQuery table
        save_rates_to_bigquery(
            rates=rates,
            project_id=project_id,
            dataset_id=dataset_id,
            table_id=instrument.lower()
        )

        print(f"Successfully updated BigQuery table {table_id} with {len(rates)} new records.")
        send_ingestion_log(instrument=instrument, count=len(rates))
    except Exception as e:
        error_message = f"Error updating BigQuery table: {e}"
        print(error_message)
        send_error_log("update_bigquery_table", error_message)

def batch_update_bigquery_tables(request):
    """
    Batch update BigQuery tables for multiple token pairs based on a JSON configuration from a Flask request.

    Args:
        request (flask.Request): The Flask request object containing JSON data.

    Returns:
        dict: A response object containing the status and details of the updates.
    """
    try:
        # Parse the JSON body from the request
        instructions = request.get_json()

        if not isinstance(instructions, list):
            return {"status": "error", "message": "Invalid input format. Expected a JSON array."}

        results = []

        # Iterate through each instruction and call update_bigquery_table
        for instruction in instructions:
            try:
                update_bigquery_table(
                    project_id=instruction["project_id"],
                    dataset_id=instruction["dataset_id"],
                    instrument=instruction["instrument"],
                    market=instruction["market"],
                    aggregate=instruction["aggregate"],
                    fill=instruction["fill"],
                    apply_mapping=instruction["apply_mapping"],
                    response_format=instruction["response_format"],
                    api_key=instruction["api_key"]
                )
                results.append({
                    "instrument": instruction["instrument"],
                    "status": "success",
                    "message": f"Successfully updated table for {instruction['instrument']}."
                })
            except KeyError as e:
                results.append({
                    "instrument": instruction.get("instrument", "unknown"),
                    "status": "error",
                    "message": f"Missing required key: {str(e)}"
                })
            except Exception as e:
                results.append({
                    "instrument": instruction.get("instrument", "unknown"),
                    "status": "error",
                    "message": f"An error occurred: {str(e)}"
                })

        return {"status": "completed", "results": results}

    except Exception as e:
        return {"status": "error", "message": f"Failed to process request: {str(e)}"}

def batch_backfill_bigquery_tables(request):
    """
    Batch backfill BigQuery tables for multiple token pairs for a specified date range,
    based on a JSON configuration from a Flask request.

    Expects JSON body like:
    [
      {
        "project_id": "your-gcp-project",
        "dataset_id": "your_dataset",
        "instrument": "ETH-BTC",
        "market": "cadli",
        "aggregate": 1,
        "fill": false,
        "apply_mapping": false,
        "response_format": "JSON",
        "api_key": "YOUR_API_KEY",
        "start_date": "2024-07-01", // Inclusive
        "end_date": "2024-12-31"    // Inclusive
      },
      { ... other instruments ... }
    ]

    Args:
        request (flask.Request): The Flask request object containing JSON data.

    Returns:
        dict: A response object containing the status and details of the backfill operations.
    """
    try:
        instructions = request.get_json()

        if not isinstance(instructions, list):
            return {"status": "error", "message": "Invalid input format. Expected a JSON array."}

        results = []

        # Iterate through each instruction
        for instruction in instructions:
            instrument = instruction.get("instrument", "unknown")
            try:
                # --- Parameter Extraction and Validation ---
                project_id = instruction["project_id"]
                dataset_id = instruction["dataset_id"]
                instrument = instruction["instrument"]
                market = instruction["market"]
                aggregate = instruction["aggregate"]
                fill = instruction["fill"]
                apply_mapping = instruction["apply_mapping"]
                response_format = instruction["response_format"]
                api_key = instruction["api_key"]
                start_date_str = instruction["start_date"]
                end_date_str = instruction["end_date"]

                # --- Date Conversion ---
                # Convert start_date string to UTC timestamp (beginning of the day)
                start_dt = datetime.strptime(start_date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
                start_ts = int(start_dt.timestamp())

                # Convert end_date string to UTC timestamp (end of the day)
                end_dt = datetime.strptime(end_date_str, "%Y-%m-%d").replace(hour=23, minute=59, second=59, tzinfo=timezone.utc)
                end_ts = int(end_dt.timestamp())

                if start_ts >= end_ts:
                     raise ValueError("Start date must be before end date.")

                print(f"\n--- Processing Backfill for: {instrument} ---")
                print(f"Range: {start_date_str} to {end_date_str}")

                # --- Fetch Data ---
                rates = fetch_historical_range(
                    market=market,
                    instrument=instrument,
                    start_ts=start_ts,
                    end_ts=end_ts,
                    aggregate=aggregate,
                    fill=fill,
                    apply_mapping=apply_mapping,
                    response_format=response_format,
                    api_key=api_key
                )

                if not rates:
                    print(f"No data fetched for {instrument} in the specified range.")
                    results.append({
                        "instrument": instrument,
                        "status": "skipped",
                        "message": "No data fetched for the specified range."
                    })
                    continue # Move to the next instruction

                # --- Save Data to BigQuery ---
                # Ensure the target table exists (create if not) - reuse logic from update_bigquery_table maybe?
                # For simplicity here, we assume save_rates_to_bigquery handles the MERGE correctly
                # and the table structure is compatible or already exists.
                # A more robust approach might explicitly check/create the table first.
                table_id = instrument.lower() # Use lowercase instrument name as table ID convention
                print(f"Saving {len(rates)} records for {instrument} to BigQuery table {dataset_id}.{table_id}...")
                save_rates_to_bigquery(
                    rates=rates,
                    project_id=project_id,
                    dataset_id=dataset_id,
                    table_id=table_id
                )

                results.append({
                    "instrument": instrument,
                    "status": "success",
                    "records_saved": len(rates),
                    "message": f"Successfully backfilled table for {instrument}."
                })
                print(f"--- Finished Backfill for: {instrument} ---")

            except KeyError as e:
                results.append({
                    "instrument": instrument,
                    "status": "error",
                    "message": f"Missing required key in instruction: {str(e)}"
                })
                send_error_log(f"batch_backfill_bigquery_tables ({instrument})", f"Missing key: {str(e)}")
            except ValueError as e:
                 results.append({
                    "instrument": instrument,
                    "status": "error",
                    "message": f"Data validation error: {str(e)}"
                })
                 send_error_log(f"batch_backfill_bigquery_tables ({instrument})", f"Data validation error: {str(e)}")
            except Exception as e:
                error_message = f"An error occurred during backfill for {instrument}: {str(e)}"
                results.append({
                    "instrument": instrument,
                    "status": "error",
                    "message": error_message
                })
                send_error_log(f"batch_backfill_bigquery_tables ({instrument})", error_message)

        return {"status": "completed", "results": results}

    except Exception as e:
        # Error parsing the overall request or other unexpected issue
        error_message = f"Failed to process batch backfill request: {str(e)}"
        print(error_message)
        send_error_log("batch_backfill_bigquery_tables (Request Level)", error_message)
        # Ensure Flask returns a JSON response even on top-level errors
        import flask # Assuming Flask context, might need adjustment if run differently
        return flask.jsonify({"status": "error", "message": error_message}), 500

def fetch_historical_code_repository(
    asset, groups, limit, aggregate, fill, response_format, api_key, asset_lookup_priority="SYMBOL", to_ts=None
):
    """
    Fetch historical code repository data from the Coindesk API.

    Args:
        asset (str): The digital asset symbol (e.g., "ETH", "BTC").
        groups (str): Comma-separated list of data groups to include.
        limit (int): The number of data points to fetch.
        aggregate (int): The aggregation interval in days.
        fill (bool): Whether to fill missing data points.
        response_format (str): The response format (e.g., "JSON").
        api_key (str): Your Coindesk API key.
        asset_lookup_priority (str): Priority for asset lookup (SYMBOL or ID).
        to_ts (int, optional): Get data before this timestamp (Unix seconds).

    Returns:
        list: A list of dictionaries containing the historical code repository data.
    """
    base_url = "https://data-api.coindesk.com/asset/v1/historical/code-repository/days"
    all_data = []  # To store all the results across multiple queries
    current_to_ts = to_ts  # Start with provided to_ts parameter or None
    
    try:
        while limit > 0:
            # Determine the number of results to fetch in this query
            query_limit = min(limit, 2000)  # API max is 2000
            
            # Prepare the query parameters
            params = {
                "asset": asset,
                "groups": groups,
                "limit": query_limit,
                "aggregate": aggregate,
                "fill": str(fill).lower(),
                "response_format": response_format,
                "asset_lookup_priority": asset_lookup_priority,
                "api_key": api_key
            }
            
            # Add to_ts if specified
            if current_to_ts:
                params["to_ts"] = current_to_ts
                
            # Make the GET request to the Coindesk API
            print(f"Fetching code repository data for {asset} with limit {query_limit}...")
            response = requests.get(
                base_url, 
                params=params,
                headers={"Content-type": "application/json; charset=UTF-8"}
            )
            response.raise_for_status()  # Raise an exception for HTTP errors
            
            # Parse the JSON response
            data = response.json()
            entries = data.get("Data", [])
            
            if not entries:
                print(f"No more code repository data available for {asset}.")
                break
                
            # Append the results to the all_data list
            all_data.extend(entries)
            
            # Update the to_ts for the next query
            earliest_timestamp = min(entry["TIMESTAMP"] for entry in entries)  # Find the earliest timestamp
            current_to_ts = earliest_timestamp - (24 * 60 * 60)  # Subtract 1 day to avoid overlap
            
            # Decrease the remaining limit
            limit -= query_limit
            print(f"Fetched {len(entries)} records for {asset}, {limit} remaining.")
            
        return all_data
    
    except requests.exceptions.RequestException as e:
        error_message = f"Error fetching code repository data from Coindesk API: {e}"
        print(error_message)
        send_error_log("fetch_historical_code_repository", error_message)
        return []
    except Exception as e:
        error_message = f"An unexpected error occurred: {e}"
        print(error_message)
        send_error_log("fetch_historical_code_repository", error_message)
        return []

def fetch_code_repository_range(
    asset, groups, start_ts, end_ts, aggregate, fill, response_format, api_key, asset_lookup_priority="SYMBOL"
):
    """
    Fetch historical code repository data from the Coindesk API for a specific date range.

    Args:
        asset (str): The digital asset symbol (e.g., "ETH", "BTC").
        groups (str): Comma-separated list of data groups to include.
        start_ts (int): Start timestamp (Unix seconds).
        end_ts (int): End timestamp (Unix seconds).
        aggregate (int): Aggregation interval (days).
        fill (bool): Whether to fill missing data points.
        response_format (str): Response format.
        api_key (str): Coindesk API key.
        asset_lookup_priority (str): Priority for asset lookup (SYMBOL or ID).

    Returns:
        list: A list of dictionaries containing the historical code repository data within the range.
    """
    base_url = "https://data-api.coindesk.com/asset/v1/historical/code-repository/days"
    all_data = []
    current_ts = end_ts  # Start from end_ts and work backwards
    max_limit_per_call = 2000  # Coindesk API limit per request
    
    print(f"Starting fetch for {asset} code repository data from {datetime.utcfromtimestamp(start_ts)} to {datetime.utcfromtimestamp(end_ts)}")
    
    try:
        while current_ts > start_ts:
            # Prepare the query parameters
            params = {
                "asset": asset,
                "groups": groups,
                "limit": max_limit_per_call,
                "aggregate": aggregate,
                "fill": str(fill).lower(),
                "response_format": response_format,
                "asset_lookup_priority": asset_lookup_priority,
                "api_key": api_key,
                "to_ts": current_ts
            }
            
            print(f"Fetching batch for {asset} code repository data to {datetime.utcfromtimestamp(current_ts)} (Limit: {max_limit_per_call})")
            
            # Make the GET request
            response = requests.get(
                base_url, 
                params=params,
                headers={"Content-type": "application/json; charset=UTF-8"}
            )
            response.raise_for_status()
            
            data = response.json()
            entries = data.get("Data", [])
            
            if not entries:
                print(f"No more data received for {asset} in this batch.")
                break
                
            # Filter entries to ensure they are within the requested range
            entries_in_range = [entry for entry in entries if start_ts <= entry.get("TIMESTAMP", 0) <= end_ts]
            
            if not entries_in_range:
                print(f"No entries within desired range for {asset}.")
                # Find the earliest timestamp to continue pagination
                earliest_timestamp = min(entry.get("TIMESTAMP", 0) for entry in entries)
                current_ts = earliest_timestamp - (24 * 60 * 60)  # Move back one day
                continue
                
            all_data.extend(entries_in_range)
            
            # Update current_ts for the next iteration
            earliest_timestamp = min(entry.get("TIMESTAMP", 0) for entry in entries)
            current_ts = earliest_timestamp - (24 * 60 * 60)  # Move back one day
            
            print(f"Fetched {len(entries_in_range)} records for {asset}. Next fetch from: {datetime.utcfromtimestamp(current_ts)}")
            
        print(f"Finished fetching for {asset}. Total records: {len(all_data)}")
        
        # Remove potential duplicates based on timestamp
        unique_data = list({entry['TIMESTAMP']: entry for entry in all_data}.values())
        unique_data.sort(key=lambda x: x.get("TIMESTAMP", 0))
        print(f"Total unique records for {asset}: {len(unique_data)}")
        
        return unique_data
        
    except requests.exceptions.RequestException as e:
        error_message = f"Error fetching code repository data from Coindesk API for {asset}: {e}"
        print(error_message)
        send_error_log("fetch_code_repository_range", error_message)
        return []
    except Exception as e:
        error_message = f"An unexpected error occurred during code repository range fetch for {asset}: {e}"
        print(error_message)
        send_error_log("fetch_code_repository_range", error_message)
        return []

def save_code_repository_to_bigquery(data, project_id, dataset_id, asset):
    """
    Save historical code repository data to a BigQuery table using a MERGE statement to prevent duplicates.

    Args:
        data (list): A list of dictionaries containing the historical code repository data.
        project_id (str): The GCP project ID.
        dataset_id (str): The BigQuery dataset ID.
        asset (str): The digital asset symbol (e.g., "ETH", "BTC").

    Returns:
        None
    """
    try:
        client = bigquery.Client(project=project_id)
        table_id = f"{project_id}.{dataset_id}.{asset.lower()}_code_repository"
        
        # Define the schema explicitly to match the target table
        schema = [
            bigquery.SchemaField("time", "TIMESTAMP"),
            bigquery.SchemaField("asset_id", "INTEGER"),
            bigquery.SchemaField("asset_symbol", "STRING"),
            bigquery.SchemaField("asset_name", "STRING"),
            bigquery.SchemaField("lines_added", "INTEGER"),
            bigquery.SchemaField("lines_removed", "INTEGER"),
            bigquery.SchemaField("commits", "INTEGER"),
            bigquery.SchemaField("code_changes", "INTEGER"),
            bigquery.SchemaField("repository", "STRING"),
            bigquery.SchemaField("forks", "INTEGER"),
            bigquery.SchemaField("stars", "INTEGER"),
            bigquery.SchemaField("subscribers", "INTEGER"),
            bigquery.SchemaField("open_pull_requests", "INTEGER"),
            bigquery.SchemaField("closed_pull_requests", "INTEGER"),
            bigquery.SchemaField("open_issues", "INTEGER"),
            bigquery.SchemaField("closed_issues", "INTEGER")
        ]
        
        # Convert data to BigQuery-compatible rows
        rows_to_insert = []
        for entry in data:
            rows_to_insert.append({
                "time": datetime.utcfromtimestamp(entry.get("TIMESTAMP", 0)).strftime("%Y-%m-%d %H:%M:%S UTC") if entry.get("TIMESTAMP") else None,
                "asset_id": entry.get("ASSET_ID"),
                "asset_symbol": entry.get("ASSET_SYMBOL"),
                "asset_name": entry.get("ASSET_NAME", entry.get("ASSET_SYMBOL")),
                "lines_added": entry.get("LINES_ADDED", 0),
                "lines_removed": entry.get("LINES_REMOVED", 0),
                "commits": entry.get("COMMITS", 0),
                "code_changes": entry.get("CODE_CHANGES", 0),
                "repository": entry.get("REPOSITORY", ""),
                "forks": entry.get("TOTAL_FORKS", 0),
                "stars": entry.get("TOTAL_STARS", 0),
                "subscribers": entry.get("TOTAL_SUBSCRIBERS", 0),
                "open_pull_requests": entry.get("TOTAL_OPEN_PULL_REQUESTS", 0),
                "closed_pull_requests": entry.get("TOTAL_CLOSED_PULL_REQUESTS", 0),
                "open_issues": entry.get("TOTAL_OPEN_ISSUES", 0),
                "closed_issues": entry.get("TOTAL_CLOSED_ISSUES", 0)
            })
        
        # Load the data into a temporary table with the explicit schema
        temp_table_id = f"{dataset_id}.temp_{asset.lower()}_code_repository"
        job_config = bigquery.LoadJobConfig(
            schema=schema,
            write_disposition="WRITE_TRUNCATE"
        )
        job = client.load_table_from_json(rows_to_insert, temp_table_id, job_config=job_config)
        job.result()  # Wait for the load job to complete
        
        # Use a MERGE statement to insert only new rows
        merge_query = f"""
            MERGE `{table_id}` T
            USING `{temp_table_id}` S
            ON T.time = S.time
            WHEN NOT MATCHED THEN
              INSERT ROW
        """
        query_job = client.query(merge_query)
        query_job.result()  # Wait for the query to complete
        
        # Clean up the temporary table
        client.delete_table(temp_table_id, not_found_ok=True)
        
        print(f"Code repository data successfully saved to BigQuery table {dataset_id}.{asset.lower()}_code_repository")
    except Exception as e:
        error_message = f"Error saving code repository data to BigQuery: {e}"
        print(error_message)
        send_error_log("save_code_repository_to_bigquery", error_message)

def update_code_repository_table(project_id, dataset_id, asset, groups, aggregate, fill, response_format, api_key, asset_lookup_priority="SYMBOL"):
    """
    Update a BigQuery table with up-to-date code repository data for a given asset.

    Args:
        project_id (str): The GCP project ID.
        dataset_id (str): The BigQuery dataset ID.
        asset (str): The digital asset symbol (e.g., "ETH", "BTC").
        groups (str): Comma-separated list of data groups to include.
        aggregate (int): The aggregation interval in days.
        fill (bool): Whether to fill missing data points.
        response_format (str): The response format (e.g., "JSON").
        api_key (str): Your Coindesk API key.
        asset_lookup_priority (str): Priority for asset lookup (SYMBOL or ID).

    Returns:
        None
    """
    try:
        client = bigquery.Client(project=project_id)
        table_id = f"{project_id}.{dataset_id}.{asset.lower()}_code_repository"
        
        # Define the schema for the table
        schema = [
            bigquery.SchemaField("time", "TIMESTAMP"),
            bigquery.SchemaField("asset_id", "INTEGER"),
            bigquery.SchemaField("asset_symbol", "STRING"),
            bigquery.SchemaField("asset_name", "STRING"),
            bigquery.SchemaField("lines_added", "INTEGER"),
            bigquery.SchemaField("lines_removed", "INTEGER"),
            bigquery.SchemaField("commits", "INTEGER"),
            bigquery.SchemaField("code_changes", "INTEGER"),
            bigquery.SchemaField("repository", "STRING"),
            bigquery.SchemaField("forks", "INTEGER"),
            bigquery.SchemaField("stars", "INTEGER"),
            bigquery.SchemaField("subscribers", "INTEGER"),
            bigquery.SchemaField("open_pull_requests", "INTEGER"),
            bigquery.SchemaField("closed_pull_requests", "INTEGER"),
            bigquery.SchemaField("open_issues", "INTEGER"),
            bigquery.SchemaField("closed_issues", "INTEGER")
        ]
        
        # Check if the table exists
        try:
            client.get_table(table_id)
            print(f"Table {table_id} exists.")
        except Exception:
            # If the table doesn't exist, create it
            print(f"Table {table_id} does not exist. Creating it...")
            table = bigquery.Table(table_id, schema=schema)
            client.create_table(table)
            print(f"Table {table_id} created successfully.")
            
        # Query the BigQuery table to find the most recent timestamp
        query = f"""
            SELECT MAX(time) AS latest_timestamp
            FROM `{table_id}`
        """
        query_job = client.query(query)
        result = query_job.result()
        latest_timestamp_row = next(result, None)
        
        # Determine the most recent timestamp in the table
        if latest_timestamp_row and latest_timestamp_row["latest_timestamp"]:
            latest_timestamp = latest_timestamp_row["latest_timestamp"]
            # Convert to Unix timestamp for the API
            latest_ts = int(latest_timestamp.timestamp())
        else:
            # If the table is empty, start from 30 days ago
            now = datetime.now(timezone.utc)
            thirty_days_ago = now - timedelta(days=30)
            latest_ts = int(thirty_days_ago.timestamp())
            
        # Calculate the number of days to fetch
        now = datetime.now(timezone.utc)
        now_ts = int(now.timestamp())
        days_diff = (now_ts - latest_ts) // (24 * 60 * 60)
        
        if days_diff <= 0:
            print(f"The code repository table for {asset} is already up-to-date.")
            return
            
        limit = days_diff + 5  # Add a few extra days to ensure coverage
        print(f"Fetching {limit} days of code repository data for {asset}...")
        
        # Fetch the new data
        data = fetch_historical_code_repository(
            asset=asset,
            groups=groups,
            limit=limit,
            aggregate=aggregate,
            fill=fill,
            response_format=response_format,
            api_key=api_key,
            asset_lookup_priority=asset_lookup_priority
        )
        
        if not data:
            print(f"No new code repository data available for {asset}.")
            return
            
        # Save the data to BigQuery
        save_code_repository_to_bigquery(
            data=data,
            project_id=project_id,
            dataset_id=dataset_id,
            asset=asset
        )
        
        print(f"Successfully updated code repository table for {asset} with {len(data)} new records.")
        send_ingestion_log(instrument=f"{asset}_code_repository", count=len(data))
    except Exception as e:
        error_message = f"Error updating code repository table: {e}"
        print(error_message)
        send_error_log("update_code_repository_table", error_message)

def batch_update_code_repository_tables(request):
    """
    Batch update BigQuery tables with code repository data for multiple assets based on a JSON configuration.

    Args:
        request (flask.Request): The Flask request object containing JSON data.

    Returns:
        dict: A response object containing the status and details of the updates.
    """
    try:
        # Parse the JSON body from the request
        instructions = request.get_json()

        if not isinstance(instructions, list):
            return {"status": "error", "message": "Invalid input format. Expected a JSON array."}

        results = []

        # Iterate through each instruction
        for instruction in instructions:
            try:
                update_code_repository_table(
                    project_id=instruction["project_id"],
                    dataset_id=instruction["dataset_id"],
                    asset=instruction["asset"],
                    groups=instruction["groups"],
                    aggregate=instruction["aggregate"],
                    fill=instruction["fill"],
                    response_format=instruction["response_format"],
                    api_key=instruction["api_key"],
                    asset_lookup_priority=instruction.get("asset_lookup_priority", "SYMBOL")
                )
                results.append({
                    "asset": instruction["asset"],
                    "status": "success",
                    "message": f"Successfully updated code repository table for {instruction['asset']}."
                })
            except KeyError as e:
                results.append({
                    "asset": instruction.get("asset", "unknown"),
                    "status": "error",
                    "message": f"Missing required key: {str(e)}"
                })
            except Exception as e:
                results.append({
                    "asset": instruction.get("asset", "unknown"),
                    "status": "error",
                    "message": f"An error occurred: {str(e)}"
                })

        return {"status": "completed", "results": results}

    except Exception as e:
        return {"status": "error", "message": f"Failed to process request: {str(e)}"}

def batch_backfill_code_repository_tables(request):
    """
    Batch backfill BigQuery tables with code repository data for a specified date range.

    Args:
        request (flask.Request): The Flask request object containing JSON data.

    Returns:
        dict: A response object containing the status and details of the backfill operations.
    """
    try:
        instructions = request.get_json()

        if not isinstance(instructions, list):
            return {"status": "error", "message": "Invalid input format. Expected a JSON array."}

        results = []

        for instruction in instructions:
            asset = instruction.get("asset", "unknown")
            try:
                # Parameter extraction and validation
                project_id = instruction["project_id"]
                dataset_id = instruction["dataset_id"]
                asset = instruction["asset"]
                groups = instruction["groups"]
                aggregate = instruction["aggregate"]
                fill = instruction["fill"]
                response_format = instruction["response_format"]
                api_key = instruction["api_key"]
                asset_lookup_priority = instruction.get("asset_lookup_priority", "SYMBOL")
                start_date_str = instruction["start_date"]
                end_date_str = instruction["end_date"]

                # Date conversion
                start_dt = datetime.strptime(start_date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
                start_ts = int(start_dt.timestamp())

                end_dt = datetime.strptime(end_date_str, "%Y-%m-%d").replace(hour=23, minute=59, second=59, tzinfo=timezone.utc)
                end_ts = int(end_dt.timestamp())

                if start_ts >= end_ts:
                    raise ValueError("Start date must be before end date.")

                print(f"\n--- Processing Code Repository Backfill for: {asset} ---")
                print(f"Range: {start_date_str} to {end_date_str}")

                # Fetch data
                data = fetch_code_repository_range(
                    asset=asset,
                    groups=groups,
                    start_ts=start_ts,
                    end_ts=end_ts,
                    aggregate=aggregate,
                    fill=fill,
                    response_format=response_format,
                    api_key=api_key,
                    asset_lookup_priority=asset_lookup_priority
                )

                if not data:
                    print(f"No code repository data fetched for {asset} in the specified range.")
                    results.append({
                        "asset": asset,
                        "status": "skipped",
                        "message": "No data fetched for the specified range."
                    })
                    continue

                # Save data to BigQuery
                save_code_repository_to_bigquery(
                    data=data,
                    project_id=project_id,
                    dataset_id=dataset_id,
                    asset=asset
                )

                results.append({
                    "asset": asset,
                    "status": "success",
                    "records_saved": len(data),
                    "message": f"Successfully backfilled code repository table for {asset}."
                })
                print(f"--- Finished Code Repository Backfill for: {asset} ---")

            except KeyError as e:
                results.append({
                    "asset": asset,
                    "status": "error",
                    "message": f"Missing required key in instruction: {str(e)}"
                })
                send_error_log(f"batch_backfill_code_repository_tables ({asset})", f"Missing key: {str(e)}")
            except ValueError as e:
                results.append({
                    "asset": asset,
                    "status": "error",
                    "message": f"Data validation error: {str(e)}"
                })
                send_error_log(f"batch_backfill_code_repository_tables ({asset})", f"Data validation error: {str(e)}")
            except Exception as e:
                error_message = f"An error occurred during code repository backfill for {asset}: {str(e)}"
                results.append({
                    "asset": asset,
                    "status": "error",
                    "message": error_message
                })
                send_error_log(f"batch_backfill_code_repository_tables ({asset})", error_message)

        return {"status": "completed", "results": results}

    except Exception as e:
        error_message = f"Failed to process batch code repository backfill request: {str(e)}"
        print(error_message)
        send_error_log("batch_backfill_code_repository_tables (Request Level)", error_message)
        import flask
        return flask.jsonify({"status": "error", "message": error_message}), 500