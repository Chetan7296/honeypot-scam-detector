"""
PERFECT VERSION - What Should Have Been Deployed
- Handles any request format (no 422 errors)
- Always returns JSON (even for errors)
- Has proper rate limiting
- Graceful error handling
"""

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Dict, Optional
import google.generativeai as genai
import os
import json
from datetime import datetime, date
import httpx
from contextlib import asynccontextmanager

# Configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
API_KEY = os.getenv("API_KEY", "your-secret-api-key-here")
GUVI_CALLBACK_URL = "https://hackathon.guvi.in/api/updateHoneyPotFinalResult"

MAX_DAILY_REQUESTS = int(os.getenv("MAX_DAILY_REQUESTS", "300"))
MAX_SESSION_MESSAGES = int(os.getenv("MAX_SESSION_MESSAGES", "20"))

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

sessions: Dict[str, Dict] = {}
usage_tracker = {"date": str(date.today()), "count": 0}
session_message_counts: Dict[str, int] = {}

# Models
class AgentResponse(BaseModel):
    status: str
    reply: str

class ExtractedIntelligence(BaseModel):
    bankAccounts: List[str] = []
    upiIds: List[str] = []
    phishingLinks: List[str] = []
    phoneNumbers: List[str] = []
    suspiciousKeywords: List[str] = []

# Budget Controls - CRITICAL!
def check_daily_budget() -> tuple[bool, str]:
    """Check if daily request limit reached"""
    today = str(date.today())
    if usage_tracker["date"] != today:
        usage_tracker["date"] = today
        usage_tracker["count"] = 0
    
    if usage_tracker["count"] >= MAX_DAILY_REQUESTS:
        return False, "Daily request limit reached"
    
    return True, "OK"

def check_session_limit(session_id: str) -> tuple[bool, str]:
    """Check if session message limit reached"""
    if session_id not in session_message_counts:
        session_message_counts[session_id] = 0
    
    if session_message_counts[session_id] >= MAX_SESSION_MESSAGES:
        return False, "Session limit reached"
    
    return True, "OK"

def increment_usage():
    usage_tracker["count"] += 1

def increment_session_count(session_id: str):
    session_message_counts[session_id] = session_message_counts.get(session_id, 0) + 1

# Gemini Interaction
async def analyze_with_gemini(message_text: str) -> Dict:
    try:
        model = genai.GenerativeModel(model_name="gemini-2.5-flash")
        
        prompt = f"""Analyze for scam. Respond ONLY with valid JSON (no markdown).

Message: "{message_text}"

Format: {{"scamDetected": true/false, "reply": "1-2 sentence response", "extractedIntelligence": {{"phoneNumbers": [], "upiIds": [], "bankAccounts": [], "phishingLinks": [], "suspiciousKeywords": []}}, "confidence": 0.9, "agentNotes": "brief analysis"}}

If scam: Act curious victim, ask natural questions."""
        
        response = model.generate_content(
            [prompt],
            generation_config=genai.GenerationConfig(
                temperature=0.7,
                max_output_tokens=512,
            )
        )
        
        response_text = response.text.strip()
        
        # Clean markdown
        if response_text.startswith("```json"):
            response_text = response_text[7:]
        if response_text.startswith("```"):
            response_text = response_text[3:]
        if response_text.endswith("```"):
            response_text = response_text[:-3]
        
        response_text = response_text.strip()
        result = json.loads(response_text)
        return result
        
    except Exception as e:
        # ALWAYS return valid structure
        return {
            "scamDetected": True,
            "reply": "Could you tell me more?",
            "extractedIntelligence": {},
            "confidence": 0.5,
            "agentNotes": f"Error: {str(e)}"
        }

# Session Management
def get_or_create_session(session_id: str) -> Dict:
    if session_id not in sessions:
        sessions[session_id] = {
            "scamDetected": False,
            "totalMessages": 0,
            "allIntelligence": ExtractedIntelligence(),
            "agentNotes": []
        }
    return sessions[session_id]

def merge_intelligence(existing: ExtractedIntelligence, new_data: Dict) -> ExtractedIntelligence:
    intel = new_data.get("extractedIntelligence", {})
    
    existing.bankAccounts = list(set(existing.bankAccounts + intel.get("bankAccounts", [])))
    existing.upiIds = list(set(existing.upiIds + intel.get("upiIds", [])))
    existing.phishingLinks = list(set(existing.phishingLinks + intel.get("phishingLinks", [])))
    existing.phoneNumbers = list(set(existing.phoneNumbers + intel.get("phoneNumbers", [])))
    existing.suspiciousKeywords = list(set(existing.suspiciousKeywords + intel.get("suspiciousKeywords", [])))
    
    return existing

async def send_final_callback(session_id: str):
    session = sessions.get(session_id)
    if not session:
        return
    
    payload = {
        "sessionId": session_id,
        "scamDetected": session["scamDetected"],
        "totalMessagesExchanged": session["totalMessages"],
        "extractedIntelligence": {
            "bankAccounts": session["allIntelligence"].bankAccounts,
            "upiIds": session["allIntelligence"].upiIds,
            "phishingLinks": session["allIntelligence"].phishingLinks,
            "phoneNumbers": session["allIntelligence"].phoneNumbers,
            "suspiciousKeywords": session["allIntelligence"].suspiciousKeywords
        },
        "agentNotes": " | ".join(session["agentNotes"][-5:])
    }
    
    try:
        async with httpx.AsyncClient() as client:
            await client.post(GUVI_CALLBACK_URL, json=payload, timeout=10.0)
    except:
        pass  # Silent failure for callback

# FastAPI App
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🚀 Honeypot API Starting")
    yield

app = FastAPI(title="Honeypot API", version="2.0", lifespan=lifespan)

# CRITICAL: Custom exception handler to ALWAYS return JSON
@app.exception_handler(Exception)
async def universal_exception_handler(request: Request, exc: Exception):
    """Catch ALL exceptions and return proper JSON"""
    return JSONResponse(
        status_code=200,  # Always return 200 to avoid client errors
        content={
            "status": "error",
            "reply": "An error occurred. Please try again."
        }
    )

@app.get("/")
async def root():
    return {
        "status": "running",
        "service": "Agentic Honeypot for Scam Detection",
        "version": "2.0",
        "endpoint": "/api/detect-scam"
    }

@app.post("/api/detect-scam")
async def detect_scam(request: Request):
    """
    Main endpoint - ALWAYS returns JSON with {status, reply}
    """
    
    try:
        # Get raw body (flexible parsing)
        try:
            body = await request.json()
        except:
            body = {}
        
        # Extract message (multiple format support)
        message_text = None
        session_id = "default-session"
        
        if isinstance(body, dict):
            if "message" in body and isinstance(body["message"], dict):
                message_text = body["message"].get("text", "")
                session_id = body.get("sessionId", session_id)
            elif "text" in body:
                message_text = body["text"]
                session_id = body.get("sessionId", session_id)
        
        # Test request handling
        if not message_text or not message_text.strip():
            return JSONResponse(
                status_code=200,
                content={
                    "status": "success",
                    "reply": "API operational. Send scam message to test detection."
                }
            )
        
        # CRITICAL: Check rate limits BEFORE processing
        daily_allowed, daily_msg = check_daily_budget()
        if not daily_allowed:
            return JSONResponse(
                status_code=200,  # Return 200, not 429!
                content={
                    "status": "error",
                    "reply": "Daily request limit reached. Please try again tomorrow."
                }
            )
        
        session_allowed, session_msg = check_session_limit(session_id)
        if not session_allowed:
            return JSONResponse(
                status_code=200,
                content={
                    "status": "success",
                    "reply": "Thank you for your time. Session limit reached."
                }
            )
        
        # Process message
        session = get_or_create_session(session_id)
        
        # Call Gemini
        ai_response = await analyze_with_gemini(message_text)
        
        # Update session
        if ai_response.get("scamDetected"):
            session["scamDetected"] = True
        
        session["allIntelligence"] = merge_intelligence(session["allIntelligence"], ai_response)
        
        if "agentNotes" in ai_response:
            session["agentNotes"].append(ai_response["agentNotes"])
        
        session["totalMessages"] += 2
        
        # Increment counters AFTER successful processing
        increment_usage()
        increment_session_count(session_id)
        
        # Callback check
        should_callback = (
            session["scamDetected"] and
            session["totalMessages"] >= 8 and (
                len(session["allIntelligence"].phoneNumbers) > 0 or
                len(session["allIntelligence"].upiIds) > 0 or
                session["totalMessages"] >= 10
            )
        )
        
        if should_callback:
            await send_final_callback(session_id)
        
        # ALWAYS return JSON with proper structure
        return JSONResponse(
            status_code=200,
            content={
                "status": "success",
                "reply": ai_response.get("reply", "Tell me more.")
            }
        )
    
    except Exception as e:
        # Catch-all: ALWAYS return JSON
        print(f"Error: {e}")
        return JSONResponse(
            status_code=200,
            content={
                "status": "error",
                "reply": "Unable to process request. Please try again."
            }
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
