import os
import json
import re
import logging
import dotenv
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
import boto3

# Load environment variables
dotenv.load_dotenv()

# Slack API tokens
SLACK_BOT_TOKEN = os.getenv("SLACK_BOT_TOKEN")
SLACK_APP_TOKEN = os.getenv("SLACK_APP_TOKEN")

# AWS Bedrock configuration
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
MODEL_ID = os.getenv("AWS_MODEL_ID", "anthropic.claude-v2")

# Emoji to listen for
SPECIFIC_EMOJI = os.getenv("SPECIFIC_EMOJI", "bulb")  # Default is "bulb"

# Logging setup
logging.basicConfig(level=logging.DEBUG)

# Initialize Slack app
app = App(token=SLACK_BOT_TOKEN)

# Initialize AWS Bedrock client
boto3_session = boto3.Session(region_name=AWS_REGION)
bedrock = boto3_session.client(service_name="bedrock-runtime")


def anonymize_users(messages):
    """Replace Slack usernames with anonymous placeholders like 'User 1', 'User 2'."""
    user_mapping = {}  # Store user ID to anonymous name mapping
    user_counter = 1

    cleaned_messages = []
    for msg in messages:
        user_id = msg.get("user", "Unknown")  # Get user ID
        text = msg.get("text", "").strip()

        if user_id not in user_mapping:
            user_mapping[user_id] = f"User {user_counter}"
            user_counter += 1

        anonymized_user = user_mapping[user_id]
        text = re.sub(r"<@U\w+>", "", text)  # Remove explicit Slack mentions

        # Format as a conversation
        cleaned_messages.append(f"{anonymized_user}: {text}")

    return "\n".join(cleaned_messages)


def summarize_with_bedrock(text):
    """Sends text to AWS Bedrock LLM and retrieves a plain-text summary using the Messages API."""
    messages = [
        {"role": "user", "content": "You are a technical assistant bot. Summarize the following Slack conversation in a structured format, focusing on: \n"
        "- What the issue was \n"
        "- How the issue was identified \n"
        "- Steps taken to resolve it \n"
        "- Final resolution \n\n" }
        {"role": "user", "content": f"Conversation:\n{text}"},
    ]

    try:
        response = bedrock.invoke_model(
            modelId=MODEL_ID,  
            body=json.dumps({
                "messages": messages,
                "max_tokens": 500,  # Corrected key
                "anthropic_version": "bedrock-2023-05-31"  # Required version
            })
        )

        result = json.loads(response["body"].read())
        return result.get("content", "No summary available.")  # Ensure correct key extraction
    except Exception as e:
        logging.error(f"Error calling AWS Bedrock: {e}")
        return "Error generating summary."


@app.event("reaction_added")
def handle_reaction(event):
    """Handles reactions, fetches anonymized conversation, summarizes, and saves the result."""
    logging.info(f"Received Event: {event}")  # Debugging
    reaction = event.get("reaction")

    if reaction != SPECIFIC_EMOJI:
        logging.info(f"Ignoring reaction: {reaction}")
        return

    channel_id = event["item"]["channel"]
    message_ts = event["item"]["ts"]  # Timestamp of the message where reaction was added

    logging.info(f"Triggered on emoji {reaction} in channel {channel_id}")

    # Fetch thread replies instead of the whole channel history
    response = app.client.conversations_replies(channel=channel_id, ts=message_ts, limit=50)
    messages = response.get("messages", [])

    if not messages:
        logging.info("No messages found in the thread to summarize.")
        return

    # Anonymize conversation
    anonymized_text = anonymize_users(messages)

    # Generate summary using AWS Bedrock
    summary = summarize_with_bedrock(anonymized_text)

    # Save summary to file
    with open("conversation_summary.txt", "w") as f:
        f.write(f"Original Conversation:\n{anonymized_text}\n\nSummary:\n{summary}")

    logging.info("Anonymized conversation and plain-text summary have been saved to a file.")


if __name__ == "__main__":
    handler = SocketModeHandler(app, SLACK_APP_TOKEN)
    handler.start()
