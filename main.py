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
from typing import Dict, List, Optional
from datetime import datetime, timedelta

# Load environment variables
load_dotenv()

# -------------------------------
# User Access Control
# -------------------------------
USER_ACCESS_CONTROL = {
    "U08LUFKQ03D": {       
        "name": "Kaustubh",
        "role": "CTO", 
        "permissions": ["all"]  # Access to everything
    },
    "U099EAAA947": {        
        "name": "Navin",
        "role": "Analytics Manager",
        "permissions": ["pipeline"]
    }
    # Add more users as needed
}

# -------------------------------
# Config
# -------------------------------
SLACK_BOT_TOKEN = os.getenv("SLACK_BOT_TOKEN")
SLACK_SIGNING_SECRET = os.getenv("SLACK_SIGNING_SECRET")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
SERVICE_ACCOUNT_FILE = os.getenv("SERVICE_ACCOUNT_FILE")
SHEET_URL = os.getenv("SHEET_URL")

# -------------------------------
# Message Deduplication Cache
# -------------------------------
class MessageCache:
    def __init__(self, max_size: int = 100, ttl_seconds: int = 300):
        self.cache = {}
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
    
    def add(self, key: str) -> bool:
        """Add a message to cache. Returns True if new, False if duplicate."""
        current_time = datetime.now()
        
        # Clean expired entries
        self._clean_expired(current_time)
        
        # Check if already processed
        if key in self.cache:
            return False
        
        # Clean if cache is too large
        if len(self.cache) >= self.max_size:
            self._clean_oldest()
        
        # Add new entry
        self.cache[key] = current_time
        return True
    
    def _clean_expired(self, current_time: datetime):
        """Remove entries older than TTL."""
        expired_keys = [
            k for k, v in self.cache.items() 
            if (current_time - v).total_seconds() > self.ttl_seconds
        ]
        for key in expired_keys:
            del self.cache[key]
    
    def _clean_oldest(self):
        """Remove oldest 20% of entries when cache is full."""
        if not self.cache:
            return
        sorted_items = sorted(self.cache.items(), key=lambda x: x[1])
        remove_count = max(1, len(sorted_items) // 5)  # Remove 20%
        for key, _ in sorted_items[:remove_count]:
            del self.cache[key]

# Initialize message cache
message_cache = MessageCache()

# -------------------------------
# Google Sheets Functions
# -------------------------------
def filter_sheets_by_permission(all_sheets: Dict, user_permissions: List[str]) -> Dict:
    """Filter sheets based on user permissions."""
    # CEO Check - "all" permission gives access to everything
    if "all" in user_permissions:
        return all_sheets
    
    # No Permission Check - empty list means no access
    if not user_permissions:
        return {}
    
    # Filter sheets based on permission matching
    filtered_sheets = {}
    for sheet_name, sheet_data in all_sheets.items():
        for permission in user_permissions:
            # Case-insensitive matching with various patterns
            permission_lower = permission.lower()
            sheet_name_lower = sheet_name.lower()
            
            if (permission_lower in sheet_name_lower or 
                sheet_name_lower in permission_lower or
                sheet_name_lower.startswith(permission_lower) or 
                permission_lower.startswith(sheet_name_lower)):
                filtered_sheets[sheet_name] = sheet_data
                break
    
    return filtered_sheets

def get_user_permissions(user_id: str) -> Dict:
    """Looks up user in USER_ACCESS_CONTROL dictionary."""
    user_info = USER_ACCESS_CONTROL.get(user_id)
    
    if user_info:
        return user_info
    else:
        # User not found - return guest with no permissions
        return {
            "name": "Guest User",
            "role": "Guest", 
            "permissions": []
        }

def read_all_sheets(service_account_file: str, spreadsheet_url: str) -> Dict:
    """Read all sheets from Google Spreadsheet."""
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
            try:
                headers = worksheet.row_values(1)
                if not headers:  # Skip empty sheets
                    continue
                rows = worksheet.get_all_records(expected_headers=headers)
                df = pd.DataFrame(rows)
                sheet_data[worksheet.title] = df
                print(f"✅ Loaded sheet '{worksheet.title}' with {len(df)} rows")
            except Exception as e:
                print(f"⚠️ Error loading sheet '{worksheet.title}': {e}")
                continue
        
        return sheet_data
    except Exception as e:
        print(f"❌ Error reading sheets: {e}")
        return {}

# -------------------------------
# AI Q&A via OpenRouter
# -------------------------------
def ask_sales_insights_openrouter(
    available_sheets: Dict[str, pd.DataFrame], 
    main_sheet_name: str,
    question: str,
    user_info: Dict
) -> str:
    """Get AI insights from the data."""
    try:
        # Get the main sheet
        main_df = available_sheets.get(main_sheet_name)
        if main_df is None or main_df.empty:
            return f"⚠️ No data available in {main_sheet_name} sheet."
        
        # Prepare data context
        data_context = f"Sheet: {main_sheet_name}\n"
        data_context += f"Columns: {', '.join(main_df.columns.tolist())}\n"
        data_context += f"Total rows: {len(main_df)}\n\n"
        
        # Include sample data (limited to prevent token overflow)
        sample_size = min(50, len(main_df))
        data_text = main_df.head(sample_size).to_csv(index=False)
        
        # Include information about other available sheets
        other_sheets = [s for s in available_sheets.keys() if s != main_sheet_name]
        if other_sheets:
            data_context += f"Other available sheets: {', '.join(other_sheets)}\n"

        prompt = f"""You are a helpful data analyst assistant.

User: {user_info['name']} ({user_info['role']})
Available data permissions: {', '.join(user_info['permissions'])}

{data_context}

Dataset (CSV, first {sample_size} rows of {main_sheet_name}):
{data_text}

Question: {question}

Provide a clear, business-style answer with specific insights from the data.
If the question relates to data not in the current sheet but available in other sheets, mention that.
Be concise but informative."""

        # Use OpenAI client with OpenRouter
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1", 
            api_key=OPENROUTER_API_KEY
        )
        
        completion = client.chat.completions.create(
            extra_headers={
                "HTTP-Referer": "https://your-app.com",
                "X-Title": "Slack Sales Bot",
            },
            model="openai/gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful data analyst. Provide clear, actionable insights based on the data provided."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1000,
            temperature=0.7
        )
        
        return completion.choices[0].message.content
            
    except Exception as e:
        print(f"❌ Error in ask_sales_insights_openrouter: {e}")
        
        # Fallback to direct API call
        try:
            return fallback_openrouter_request(data_text, question, user_info)
        except Exception as fallback_error:
            return f"⚠️ Error getting AI response: {str(fallback_error)}"

def fallback_openrouter_request(data_text: str, question: str, user_info: Dict) -> str:
    """Fallback method using direct HTTP requests."""
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://your-app.com",
        "X-Title": "Slack Sales Bot",
    }
    
    prompt = f"""You are a helpful data analyst.
User: {user_info['name']} ({user_info['role']})

Dataset (CSV):
{data_text}

Question: {question}

Provide a clear, business-style answer with insights."""
    
    payload = {
        "model": "openai/gpt-4o-mini",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 1000,
        "temperature": 0.7
    }

    response = requests.post(url, headers=headers, json=payload, timeout=30)
    
    if response.status_code == 200:
        data = response.json()
        return data["choices"][0]["message"]["content"]
    else:
        # Try with free model as last resort
        payload["model"] = "meta-llama/llama-3.1-8b-instruct:free"
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            return data["choices"][0]["message"]["content"]
        else:
            raise Exception(f"OpenRouter API error: {response.status_code} - {response.text}")

# -------------------------------
# Slack Functions
# -------------------------------
def verify_slack_signature(request: Request, body: bytes):
    """Verify the request is from Slack."""
    if os.getenv("ENV") == "dev":   # Skip verification in dev mode
        return
        
    timestamp = request.headers.get("X-Slack-Request-Timestamp")
    slack_signature = request.headers.get("X-Slack-Signature")
    
    if not timestamp or not slack_signature:
        raise HTTPException(status_code=400, detail="Missing Slack headers")

    # Check timestamp is recent (within 5 minutes)
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

def post_message(channel: str, text: str, thread_ts: Optional[str] = None) -> Optional[Dict]:
    """Post a message to Slack."""
    try:
        url = "https://slack.com/api/chat.postMessage"
        headers = {"Authorization": f"Bearer {SLACK_BOT_TOKEN}"}
        payload = {
            "channel": channel, 
            "text": text,
            "unfurl_links": False,
            "unfurl_media": False
        }
        
        # Add thread_ts if provided to reply in thread
        if thread_ts:
            payload["thread_ts"] = thread_ts
            
        response = requests.post(url, headers=headers, json=payload, timeout=10)
        
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

def extract_clean_question(text: str) -> str:
    """Extract the actual question from the mention text."""
    # Remove bot mention (e.g., <@U123456>)
    import re
    cleaned = re.sub(r'<@[A-Z0-9]+>', '', text).strip()
    return cleaned if cleaned else text

# -------------------------------
# FastAPI App
# -------------------------------
app = FastAPI(title="Slack Sales Bot")

@app.post("/slack/events")
async def slack_events(request: Request):
    """Handle Slack events."""
    try:
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
            # Extract event details
            raw_text = event.get("text", "")
            channel = event.get("channel")
            user_id = event.get("user")
            event_ts = event.get("ts")
            thread_ts = event.get("thread_ts")
            
            # Skip bot messages
            if event.get("bot_id"):
                print("🤖 Skipping bot message")
                return {"ok": True}
            
            # Clean the question text
            user_question = extract_clean_question(raw_text)
            
            # Skip empty messages
            if len(user_question.strip()) < 3:
                print("📝 Skipping too short message")
                return {"ok": True}
            
            # Check for duplicate processing
            message_key = f"{channel}_{user_id}_{event_ts}"
            if not message_cache.add(message_key):
                print(f"🔄 Already processed message: {message_key}")
                return {"ok": True}
            
            # Get user permissions
            user_info = get_user_permissions(user_id)
            print(f"👤 User: {user_info['name']} ({user_info['role']}) - Permissions: {user_info['permissions']}")
            print(f"💬 Question: {user_question}")
            
            # Determine reply thread
            reply_ts = thread_ts or event_ts
            
            # Check permissions
            if not user_info['permissions']:
                unauthorized_msg = (
                    f"🚫 Sorry <@{user_id}>, you don't have permission to access company data.\n"
                    "Please contact your administrator for access."
                )
                post_message(channel, unauthorized_msg, thread_ts=reply_ts)
                return {"ok": True}
            
            # Send typing indicator (acknowledge receipt)
            post_message(channel, "🔍 Analyzing your question...", thread_ts=reply_ts)
            
            # Load and filter data
            all_sheets = read_all_sheets(SERVICE_ACCOUNT_FILE, SHEET_URL)
            
            if not all_sheets:
                error_msg = "⚠️ Unable to load data sheets. Please try again later or contact support."
                post_message(channel, error_msg, thread_ts=reply_ts)
                return {"ok": True}
            
            # Filter sheets based on permissions
            available_sheets = filter_sheets_by_permission(all_sheets, user_info['permissions'])
            
            if not available_sheets:
                no_access_msg = (
                    f"⚠️ Sorry {user_info['name']}, you don't have access to any data sheets "
                    "that match your current permissions."
                )
                post_message(channel, no_access_msg, thread_ts=reply_ts)
                return {"ok": True}
            
            # Determine main sheet
            main_sheet = determine_main_sheet(available_sheets, user_info['permissions'])
            
            print(f"📊 Available sheets: {list(available_sheets.keys())}")
            print(f"🎯 Main sheet: {main_sheet}")
            
            # Get AI response
            answer = ask_sales_insights_openrouter(
                available_sheets, 
                main_sheet, 
                user_question, 
                user_info
            )
            
            # Format response with user context
            formatted_answer = f"📊 *Analysis for {user_info['name']}*\n\n{answer}"
            
            # Post response
            post_message(channel, formatted_answer, thread_ts=reply_ts)

        return {"ok": True}
        
    except HTTPException as he:
        print(f"❌ HTTP Exception: {he.detail}")
        raise he
    except Exception as e:
        print(f"❌ Unexpected error in slack_events: {e}")
        import traceback
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"ok": False, "error": "Internal server error"}
        )

def determine_main_sheet(available_sheets: Dict, permissions: List[str]) -> str:
    """Determine the main sheet to use based on permissions."""
    # Priority order for main sheets
    priority_sheets = ["Pipeline", "Sales", "Revenue", "Customers"]
    
    # Check for priority sheets
    for sheet in priority_sheets:
        if sheet in available_sheets:
            return sheet
    
    # Check permissions for hints
    for permission in permissions:
        for sheet_name in available_sheets.keys():
            if permission.lower() in sheet_name.lower():
                return sheet_name
    
    # Default to first available sheet
    return list(available_sheets.keys())[0]

# -------------------------------
# Additional Endpoints
# -------------------------------
@app.get("/")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "message": "Slack Sales Bot is running",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/config/status")
async def config_status():
    """Check configuration status."""
    config_checks = {
        "slack_bot_token": bool(SLACK_BOT_TOKEN),
        "slack_signing_secret": bool(SLACK_SIGNING_SECRET),
        "openrouter_api_key": bool(OPENROUTER_API_KEY),
        "service_account_file": bool(SERVICE_ACCOUNT_FILE),
        "sheet_url": bool(SHEET_URL),
        "service_account_exists": os.path.exists(SERVICE_ACCOUNT_FILE) if SERVICE_ACCOUNT_FILE else False
    }
    
    all_configured = all(config_checks.values())
    
    return {
        "configured": all_configured,
        "checks": config_checks,
        "users_configured": len(USER_ACCESS_CONTROL)
    }

@app.post("/test")
async def test_endpoint():
    """Test endpoint to verify functionality without Slack."""
    try:
        # Test Google Sheets connection
        all_sheets = read_all_sheets(SERVICE_ACCOUNT_FILE, SHEET_URL)
        
        if not all_sheets:
            return {
                "status": "error",
                "message": "No sheets could be loaded",
                "suggestion": "Check SERVICE_ACCOUNT_FILE and SHEET_URL configuration"
            }
        
        # Get a test user
        test_user_id = list(USER_ACCESS_CONTROL.keys())[0] if USER_ACCESS_CONTROL else None
        test_user_info = get_user_permissions(test_user_id) if test_user_id else {"name": "Test", "role": "Test", "permissions": ["all"]}
        
        # Filter sheets for test user
        available_sheets = filter_sheets_by_permission(all_sheets, test_user_info['permissions'])
        
        if available_sheets:
            main_sheet = determine_main_sheet(available_sheets, test_user_info['permissions'])
            test_question = "What is the current status summary?"
            
            # Test AI response
            answer = ask_sales_insights_openrouter(
                available_sheets,
                main_sheet,
                test_question,
                test_user_info
            )
            
            return {
                "status": "success",
                "all_sheets": list(all_sheets.keys()),
                "user_accessible_sheets": list(available_sheets.keys()),
                "main_sheet": main_sheet,
                "test_user": test_user_info,
                "test_response": answer[:500] + "..." if len(answer) > 500 else answer
            }
        else:
            return {
                "status": "warning",
                "message": "Sheets loaded but no accessible sheets for test user",
                "all_sheets": list(all_sheets.keys()),
                "test_user": test_user_info
            }
            
    except Exception as e:
        return {
            "status": "error",
            "message": str(e),
            "type": type(e).__name__
        }

# -------------------------------
# Main Execution
# -------------------------------
if __name__ == "__main__":
    print("=" * 50)
    print("🤖 Slack Sales Bot - System Check")
    print("=" * 50)
    
    # Check environment variables
    required_vars = ["SLACK_BOT_TOKEN", "SLACK_SIGNING_SECRET", "OPENROUTER_API_KEY", "SERVICE_ACCOUNT_FILE", "SHEET_URL"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        print(f"❌ Missing environment variables: {missing_vars}")
        print("   Please check your .env file")
    else:
        print("✅ All environment variables loaded")
    
    # Check service account file
    if SERVICE_ACCOUNT_FILE:
        if os.path.exists(SERVICE_ACCOUNT_FILE):
            print(f"✅ Service account file found: {SERVICE_ACCOUNT_FILE}")
        else:
            print(f"❌ Service account file not found: {SERVICE_ACCOUNT_FILE}")
    
    # Test Google Sheets connection
    if not missing_vars:
        try:
            print("\n📊 Testing Google Sheets connection...")
            all_sheets = read_all_sheets(SERVICE_ACCOUNT_FILE, SHEET_URL)
            
            if all_sheets:
                print(f"✅ Successfully loaded {len(all_sheets)} sheets:")
                for sheet_name, df in all_sheets.items():
                    print(f"   - {sheet_name}: {len(df)} rows, {len(df.columns)} columns")
            else:
                print("❌ No sheets could be loaded")
        except Exception as e:
            print(f"❌ Error testing Google Sheets: {e}")
    
    print("\n" + "=" * 50)
    print("To run the server:")
    print("  uvicorn main:app --reload --port 8000")
    print("=" * 50)