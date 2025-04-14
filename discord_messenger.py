import requests
import os
import json
from datetime import datetime
import pytz  # Add this import for timezone handling

# Discord webhook URL (set in environment variables for security)
DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL")

def send_discord_message(content: str, username="Cryptomancer Bot", avatar_url=None):
    """
    Sends a simple message to a Discord channel via webhook.

    Args:
        content (str): The message content to send.
        username (str): The name displayed as the message sender in Discord.
        avatar_url (str, optional): An image URL to use as the avatar for the message sender.
    """
    if not DISCORD_WEBHOOK_URL:
        print("[Discord] Webhook URL not set in environment.")
        return

    payload = {
        "username": username,
        "content": content
    }

    if avatar_url:
        payload["avatar_url"] = avatar_url

    try:
        response = requests.post(DISCORD_WEBHOOK_URL, data=json.dumps(payload), headers={"Content-Type": "application/json"})
        if response.status_code != 204:
            print(f"[Discord] Failed to send message: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"[Discord] Exception occurred while sending message: {e}")

def send_ingestion_log(instrument: str, count: int, timestamp=None):
    """
    Sends a structured message about data ingestion.

    Args:
        instrument (str): The trading pair (e.g., ETH-MATIC).
        count (int): Number of records ingested.
        timestamp (str): Optional UTC timestamp. If None, current time is used.
    """
    # Convert timestamp to New York timezone
    ny_tz = pytz.timezone("America/New_York")
    if timestamp:
        utc_time = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S UTC").replace(tzinfo=pytz.utc)
        local_time = utc_time.astimezone(ny_tz)
    else:
        local_time = datetime.now(pytz.utc).astimezone(ny_tz)

    formatted_time = local_time.strftime("%Y-%m-%d %I:%M:%S %p %Z")
    message = f"📊 Data ingestion completed for **{instrument}**\n" \
              f"✅ {count} new records ingested\n" \
              f"🕒 {formatted_time}"

    send_discord_message(message)

def send_error_log(context: str, error: str):
    """
    Sends an error message to Discord.

    Args:
        context (str): What part of the app triggered the error.
        error (str): Description or exception message.
    """
    # Use New York timezone for error logs
    ny_tz = pytz.timezone("America/New_York")
    local_time = datetime.now(pytz.utc).astimezone(ny_tz)
    formatted_time = local_time.strftime("%Y-%m-%d %I:%M:%S %p %Z")

    message = f"🚨 Error in **{context}**\n" \
              f"❌ {error}\n" \
              f"🕒 {formatted_time}"
    send_discord_message(message)
