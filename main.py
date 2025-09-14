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


USER_ACCESS_CONTROL = {
    "U08LUFKQ03D": {        # Arjun (CEO)
        "name": "Kaustubh",
        "role": "CTO", 
        "permissions": ["all"]  # Access to everything
    },
    "U456HR": {         # Ameya (HR Head)
        "name": "Ameya",
        "role": "HR Head",
        "permissions": ["hr", "employees", "payroll", "recruitment"]
    }
    # ... more users
}

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


def filter_sheets_by_permission(all_sheets: dict, user_permissions: list) -> dict:
    # 1. CEO Check - "all" permission gives access to everything
    if "all" in user_permissions:
        return all_sheets
    
    # 2. No Permission Check - empty list means no access
    if not user_permissions:
        return {}
    
    # 3. Filter sheets based on permission matching
    filtered_sheets = {}
    for sheet_name, sheet_data in all_sheets.items():
        for permission in user_permissions:
            if (permission.lower() in sheet_name.lower() or sheet_name.lower() in permission.lower() 
                or sheet_name.lower().startswith(permission.lower()) 
                or permission.lower().startswith(sheet_name.lower())):

                filtered_sheets[sheet_name] = sheet_data
                break
    
    return filtered_sheets

def get_user_permissions(user_id: str) -> dict:
    # Looks up user in USER_ACCESS_CONTROL dictionary
    user_info = USER_ACCESS_CONTROL.get(user_id)
    
    if user_info:
        # User found - return their permissions
        return user_info
    else:
        # User not found - return guest with no permissions
        return {
            "name": "Guest User",
            "role": "Guest", 
            "permissions": []
        }
    
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
        data_text = df.head(50).to_csv(index=False)

        prompt = f"""
You are a helpful sales analyst.
Dataset (CSV, first 50 rows):
{data_text}

Question: {question}

Provide a clear, business-style answer with insights.
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
                model="openai/gpt-4o-mini",  # More reliable and cost-effective model
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

def post_message(channel: str, text: str, thread_ts: str = None):
    try:
        url = "https://slack.com/api/chat.postMessage"
        headers = {"Authorization": f"Bearer {SLACK_BOT_TOKEN}"}
        payload = {
            "channel": channel, 
            "text": text,
            "unfurl_links": False,  # Prevent auto-unfurling
            "unfurl_media": False   # Prevent media unfurling
        }
        
        # Add thread_ts if provided to reply in thread
        if thread_ts:
            payload["thread_ts"] = thread_ts
            
        response = requests.post(url, headers=headers, json=payload)
        
        if response.status_code == 200:
            result = response.json()
            if result.get("ok"):
                print(f"✅ Successfully posted message to {channel}")
                return result
            else:
                print(f"❌ Slack API error: {result.get('error')}")
                return None
        else:
            print(f"❌ HTTP Error posting to Slack: {response.status_code} - {response.text}")
            return None
        
    except Exception as e:
        print(f"❌ Exception posting message: {e}")
        return None

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
            user_id = event.get("user")
            event_ts = event.get("ts")  # Original message timestamp
            thread_ts = event.get("thread_ts")  # If already in a thread
            
            # Skip if this is our own message (bot responding to itself)
            if event.get("bot_id"):
                print("🤖 Skipping bot message to avoid loops")
                return {"ok": True}
            
            # Skip if message is empty or too short
            if len(user_question.strip()) < 3:
                print("📝 Skipping too short message")
                return {"ok": True}
            
            # Get user permissions
            user_info = get_user_permissions(user_id)
            print(f"👤 User: {user_info['name']} ({user_info['role']}) - Permissions: {user_info['permissions']}")
            
            # Check if user has any permissions
            if not user_info['permissions']:
                unauthorized_msg = f"🚫 Sorry, you don't have permission to access company data. Please contact your administrator."
                # Use thread_ts if already in thread, otherwise event_ts for new thread
                reply_ts = thread_ts or event_ts
                post_message(channel, unauthorized_msg, thread_ts=reply_ts)
                return {"ok": True}
            
            # Create message key for deduplication (but allow multiple questions in same thread)
            message_key = f"{channel}_{user_id}_{event_ts}"
            
            # Simple in-memory cache to track recent messages
            if not hasattr(app, '_processed_messages'):
                app._processed_messages = set()
            
            # Clean old messages if cache gets too large
            if len(app._processed_messages) > 100:
                app._processed_messages.clear()
            
            # Check if we've already processed this exact message
            if message_key in app._processed_messages:
                print(f"🔄 Already processed message: {message_key}")
                return {"ok": True}
            
            # Add to processed messages
            app._processed_messages.add(message_key)
            
            print(f"📨 Processing message from {user_info['name']} in channel {channel}")

            # Load Google Sheets data
            all_sheets = read_all_sheets(SERVICE_ACCOUNT_FILE, SHEET_URL)
            
            # Filter sheets based on user permissions
            available_sheets = filter_sheets_by_permission(all_sheets, user_info['permissions'])
            
            if not available_sheets:
                answer = f"⚠️ Sorry {user_info['name']}, you don't have access to any data sheets that are currently available."
            else:
                # Determine main sheet based on user role
                main_sheet = None
                if "pipeline" in [p.lower() for p in user_info['permissions']] or "all" in user_info['permissions']:
                    main_sheet = "Pipeline"
                else:
                    # Use first available sheet as main sheet
                    main_sheet = list(available_sheets.keys())[0]
                
                print(f"📊 Available sheets for {user_info['name']}: {list(available_sheets.keys())}")
                print(f"🎯 Main sheet: {main_sheet}")
                
                # Get AI response with filtered data
                answer = ask_sales_insights_openrouter(available_sheets, main_sheet, user_question, user_info)

            # Post response to Slack - continue in same thread if it exists
            reply_ts = thread_ts or event_ts  # Use existing thread or create new one
            post_message(channel, answer, thread_ts=reply_ts)

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