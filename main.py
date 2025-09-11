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
from openai import OpenAI

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
MODEL = "google/gemini-pro-1.5"  # Default model

# -------------------------------
# Google Sheets Loader
# -------------------------------
def read_all_sheets(service_account_file: str, spreadsheet_url: str) -> dict:
    try:
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
    except Exception as e:
        print(f"Error reading sheets: {e}")
        return {}

# -------------------------------
# AI Q&A via OpenRouter
# -------------------------------
def ask_sales_insights_openrouter(df: pd.DataFrame, question: str) -> str:
    try:
        # Convert DataFrame to CSV string for the prompt
        data_text = df.to_csv(index=False)

        prompt = f"""
You are an expert business data analyst working for a sales operations team.
Your responsibilities:
- Interpret data accurately, focusing on trends, anomalies, and pipeline progress.
- Provide **actionable recommendations**, not just summaries.
- Use structured, business-oriented language.
- Where helpful, suggest specific charts (time-series, bar charts, funnel plots).
- Keep answers concise but insightful, like an executive summary.

Dataset (CSV, all rows):
{data_text}

User Question: {question}

Respond in the following format:
1. **Key Findings** (3 to 5 bullet points)
2. **Notable Anomalies**
3. **Recommendations**.
"""
        
        # Method 1: Using OpenAI client with OpenRouter (recommended)
        try:
            client = OpenAI(
                base_url="https://openrouter.ai/api/v1", 
                api_key=OPENROUTER_API_KEY
            )
            
            completion = client.chat.completions.create(
                extra_headers={
                    "HTTP-Referer": "https://your-app.com",  # Optional: replace with your site
                    "X-Title": "Slack Sales Bot",  # Optional: your app name
                },
                model=MODEL,  # More reliable and cost-effective model
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            return completion.choices[0].message.content
            
        except Exception as client_error:
            print(f"OpenAI client method failed: {client_error}")
            
            # Method 2: Fallback to direct requests
            url = "https://openrouter.ai/api/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://your-app.com",  # Optional: replace with your site
                "X-Title": "Slack Sales Bot",  # Optional: your app name
            }
            
            payload = {
                "model": "openai/gpt-4o-mini",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 1000,
                "temperature": 0.7
            }

            response = requests.post(url, headers=headers, json=payload)
            
            if response.status_code == 200:
                data = response.json()
                return data["choices"][0]["message"]["content"]
            else:
                # Try with a different model if the first fails
                payload["model"] = "meta-llama/llama-3.1-8b-instruct:free"
                response = requests.post(url, headers=headers, json=payload)
                
                if response.status_code == 200:
                    data = response.json()
                    return data["choices"][0]["message"]["content"]
                else:
                    return f"⚠️ Error from OpenRouter: {response.status_code} - {response.text}"
        
    except Exception as e:
        return f"⚠️ Error getting AI response: {str(e)}"

# -------------------------------
# Slack Helpers
# -------------------------------
def verify_slack_signature(request: Request, body: bytes):
    if os.getenv("ENV") == "dev":   # Skip verification in dev mode
        return
        
    timestamp = request.headers.get("X-Slack-Request-Timestamp")
    slack_signature = request.headers.get("X-Slack-Signature")
    
    if not timestamp or not slack_signature:
        raise HTTPException(status_code=400, detail="Missing Slack headers")

    if abs(time.time() - int(timestamp)) > 60 * 5:
        raise HTTPException(status_code=400, detail="Invalid timestamp")

    sig_basestring = f"v0:{timestamp}:{body.decode('utf-8')}"
    my_signature = "v0=" + hmac.new(
        SLACK_SIGNING_SECRET.encode(), 
        sig_basestring.encode(), 
        hashlib.sha256
    ).hexdigest()

    if not hmac.compare_digest(my_signature, slack_signature):
        raise HTTPException(status_code=400, detail="Invalid Slack signature")

def post_message(channel: str, text: str):
    try:
        url = "https://slack.com/api/chat.postMessage"
        headers = {"Authorization": f"Bearer {SLACK_BOT_TOKEN}"}
        payload = {"channel": channel, "text": text}
        response = requests.post(url, headers=headers, json=payload)
        
        if response.status_code != 200:
            print(f"Error posting to Slack: {response.text}")
        
        return response.json()
    except Exception as e:
        print(f"Error posting message: {e}")

# -------------------------------
# FastAPI App
# -------------------------------
app = FastAPI()

@app.post("/slack/events")
async def slack_events(request: Request):
    try:
        body = await request.body()
        data = json.loads(body)

        # Slack challenge verification
        if "challenge" in data:
            return JSONResponse(content={"challenge": data["challenge"]})

        # Verify signature (skip in dev mode)
        verify_slack_signature(request, body)

        # Handle events
        event = data.get("event", {})
        if event.get("type") == "app_mention":
            user_question = event.get("text", "")
            channel = event.get("channel")

            # Load Google Sheets data
            all_sheets = read_all_sheets(SERVICE_ACCOUNT_FILE, SHEET_URL)
            df = all_sheets.get("Pipeline")

            if df is not None and not df.empty:
                answer = ask_sales_insights_openrouter(df, user_question)
            else:
                answer = "⚠️ Could not find 'Pipeline' sheet or sheet is empty."

            # Post response to Slack
            post_message(channel, answer)

        return {"ok": True}
        
    except Exception as e:
        print(f"Error in slack_events: {e}")
        return {"ok": False, "error": str(e)}

# Health check endpoint
@app.get("/")
async def health_check():
    return {"status": "healthy", "message": "Slack Sales Bot is running"}

# Test endpoint for development
@app.post("/test")
async def test_endpoint():
    """Test endpoint to verify the bot functionality without Slack"""
    try:
        # Test Google Sheets connection
        all_sheets = read_all_sheets(SERVICE_ACCOUNT_FILE, SHEET_URL)
        df = all_sheets.get("Pipeline")
        
        if df is not None and not df.empty:
            # Test AI response
            test_question = "What is the current pipeline status?"
            answer = ask_sales_insights_openrouter(df, test_question)
            return {
                "status": "success",
                "sheets_loaded": list(all_sheets.keys()),
                "pipeline_rows": len(df),
                "test_response": answer
            }
        else:
            return {
                "status": "error",
                "message": "Pipeline sheet not found or empty",
                "sheets_available": list(all_sheets.keys()) if all_sheets else []
            }
    except Exception as e:
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    # Test the application
    print("Testing Slack Sales Bot...")
    
    # Test 1: Check environment variables
    required_vars = ["SLACK_BOT_TOKEN", "SLACK_SIGNING_SECRET", "OPENROUTER_API_KEY", "SERVICE_ACCOUNT_FILE", "SHEET_URL"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        print(f"❌ Missing environment variables: {missing_vars}")
    else:
        print("✅ All environment variables loaded")
    
    # Test 2: Test Google Sheets connection
    try:
        all_sheets = read_all_sheets(SERVICE_ACCOUNT_FILE, SHEET_URL)
        if all_sheets:
            print(f"✅ Successfully loaded {len(all_sheets)} sheets: {list(all_sheets.keys())}")
            
            # Test Pipeline sheet specifically
            if "Pipeline" in all_sheets:
                df = all_sheets["Pipeline"]
                print(f"✅ Pipeline sheet loaded with {len(df)} rows")
                
                # Test AI response
                test_question = "What is the pipeline summary?"
                answer = ask_sales_insights_openrouter(df, test_question)
                print(f"✅ AI Response: {answer[:100]}...")
            else:
                print("❌ 'Pipeline' sheet not found")
        else:
            print("❌ No sheets loaded")
    except Exception as e:
        print(f"❌ Error testing: {e}")
    
    # To run the server, uncomment the lines below:
    # import uvicorn
    # uvicorn.run(app, host="0.0.0.0", port=8000)