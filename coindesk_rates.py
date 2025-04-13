import requests
import csv
from datetime import datetime, timezone
from google.cloud import bigquery
import json

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
        print(f"Error fetching data from Coindesk API: {e}")
        return []
    except Exception as e:
        print(f"An error occurred: {e}")
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