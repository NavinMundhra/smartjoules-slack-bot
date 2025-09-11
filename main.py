import os
import hmac
import hashlib
import time
import json
import requests
import pandas as pd
import gspread
from fastapi import FastAPI, Request, HTTPException
from google.oauth2.service_account import Credentials
from starlette.responses import JSONResponse
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# -------------------------------
# Config
# -------------------------------
SLACK_BOT_TOKEN = os.getenv("SLACK_BOT_TOKEN")
SLACK_SIGNING_SECRET = os.getenv("SLACK_SIGNING_SECRET")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
SERVICE_ACCOUNT_FILE = os.getenv("SERVICE_ACCOUNT_FILE")
SHEET_URL = os.getenv("SHEET_URL")

# -------------------------------
# Google Sheets Loader
# -------------------------------
def read_all_sheets(service_account_file: str, spreadsheet_url: str) -> dict:
    SCOPES = [
        "https://www.googleapis.com/auth/spreadsheets.readonly",
        "https://www.googleapis.com/auth/drive.readonly",
    ]
    creds = Credentials.from_service_account_file(service_account_file, scopes=SCOPES)
    client = gspread.authorize(creds)
    spreadsheet = client.open_by_url(spreadsheet_url)

    sheet_data = {}
    for worksheet in spreadsheet.worksheets():
        headers = worksheet.row_values(1)
        rows = worksheet.get_all_records(expected_headers=headers)
        df = pd.DataFrame(rows)
        sheet_data[worksheet.title] = df
    return sheet_data

# -------------------------------
# AI Q&A via OpenRouter
# -------------------------------
def ask_sales_insights_openrouter(df: pd.DataFrame, question: str) -> str:
    data_text = df.head(50).to_csv(index=False)

    prompt = f"""
    You are a helpful sales analyst.
    Dataset (CSV, first 50 rows):
    {data_text}

    Question: {question}

    Provide a clear, business-style answer with insights.
    """

    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": "google/gemini-pro-1.5",  # ✅ use a model your key supports
        "messages": [{"role": "user", "content": prompt}],
    }

    response = requests.post(url, headers=headers, json=payload)
    if response.status_code != 200:
        return f"⚠️ Error from OpenRouter: {response.text}"

    data = response.json()
    return data["choices"][0]["message"]["content"]

# -------------------------------
# Slack Helpers
# -------------------------------
def verify_slack_signature(request: Request, body: bytes):
    if os.getenv("ENV") == "dev":   # ✅ skip in dev mode
        return
    timestamp = request.headers.get("X-Slack-Request-Timestamp")
    slack_signature = request.headers.get("X-Slack-Signature")
    if not timestamp or not slack_signature:
        raise HTTPException(status_code=400, detail="Missing Slack headers")

    if abs(time.time() - int(timestamp)) > 60 * 5:
        raise HTTPException(status_code=400, detail="Invalid timestamp")

    sig_basestring = f"v0:{timestamp}:{body.decode('utf-8')}"
    my_signature = "v0=" + hmac.new(
        SLACK_SIGNING_SECRET.encode(), sig_basestring.encode(), hashlib.sha256
    ).hexdigest()

    if not hmac.compare_digest(my_signature, slack_signature):
        raise HTTPException(status_code=400, detail="Invalid Slack signature")

def post_message(channel: str, text: str):
    url = "https://slack.com/api/chat.postMessage"
    headers = {"Authorization": f"Bearer {SLACK_BOT_TOKEN}"}
    payload = {"channel": channel, "text": text}
    requests.post(url, headers=headers, json=payload)

# -------------------------------
# FastAPI App
# -------------------------------
app = FastAPI()

@app.post("/slack/events")
async def slack_events(request: Request):
    body = await request.body()
    data = json.loads(body)

    # Slack challenge verification
    if "challenge" in data:
        return JSONResponse(content={"challenge": data["challenge"]})

    # Verify signature
    verify_slack_signature(request, body)

    # Handle events
    event = data.get("event", {})
    if event.get("type") == "app_mention":
        user_question = event.get("text", "")
        channel = event.get("channel")

        all_sheets = read_all_sheets(SERVICE_ACCOUNT_FILE, SHEET_URL)
        df = all_sheets.get("Pipeline")

        if df is not None:
            answer = ask_sales_insights_openrouter(df, user_question)
        else:
            answer = "⚠️ Could not find 'Pipeline' sheet."

        post_message(channel, answer)

    return {"ok": True}

if __name__ == "__main__":
    fake_event = {
        "event": {
            "type": "app_mention",
            "text": "What is the pipeline?",
            "channel": "C12345"
        }
    }
    from fastapi.testclient import TestClient
    client = TestClient(app)
    response = client.post("/slack/events", json=fake_event)
    print(response.json())
