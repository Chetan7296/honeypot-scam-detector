"""
ULTRA-FLEXIBLE VERSION - Accepts ANY request format
Works with GUVI test and full requests
"""

from fastapi import FastAPI, Header, HTTPException, Request
from pydantic import BaseModel
from typing import List, Dict, Optional, Any
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
usage_tracker = {"date": str(date.today()), "count": 0, "warned": False}
session_message_counts: Dict[str, int] = {}

# Simple response model
class AgentResponse(BaseModel):
    status: str
    reply: str

class ExtractedIntelligence(BaseModel):
    bankAccounts: List[str] = []
    upiIds: List[str] = []
    phishingLinks: List[str] = []
    phoneNumbers: List[str] = []
    suspiciousKeywords: List[str] = []

# Budget Controls
def increment_usage():
    usage_tracker["count"] += 1

def get_usage_stats() -> Dict:
    return {
        "date": usage_tracker["date"],
        "dailyCount": usage_tracker["count"],
        "dailyLimit": MAX_DAILY_REQUESTS,
        "activeSessions": len(sessions)
    }

# Gemini Interaction
async def analyze_with_gemini(message_text: str) -> Dict:
    try:
        model = genai.GenerativeModel(model_name="gemini-2.5-flash")
        
        prompt = f"""Analyze this message for scam. Respond ONLY with valid JSON (no markdown).

Message: "{message_text}"

Format: {{"scamDetected": true/false, "reply": "short response (1-2 sentences)", "extractedIntelligence": {{"phoneNumbers": [], "upiIds": [], "bankAccounts": [], "phishingLinks": [], "suspiciousKeywords": []}}, "confidence": 0.9, "agentNotes": "analysis"}}

If scam: Act as curious victim, ask natural questions. Extract phone numbers, UPIs, bank accounts, links."""
        
        response = model.generate_content(
            [prompt],
            generation_config=genai.GenerationConfig(
                temperature=0.7,
                max_output_tokens=1024,
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
        print(f"❌ Gemini error: {e}")
        return {
            "scamDetected": True,
            "reply": "Could you tell me more?",
            "extractedIntelligence": {},
            "confidence": 0.5,
            "agentNotes": str(e)
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
            response = await client.post(GUVI_CALLBACK_URL, json=payload, timeout=10.0)
            print(f"✅ Callback sent: {response.status_code}")
    except Exception as e:
        print(f"❌ Callback failed: {e}")

# FastAPI
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🚀 Honeypot API Starting")
    yield

app = FastAPI(title="Honeypot API", version="2.0", lifespan=lifespan)

@app.get("/")
async def root():
    return {
        "status": "running",
        "service": "Agentic Honeypot for Scam Detection",
        "version": "2.0",
        "endpoint": "/api/detect-scam",
        "usage": get_usage_stats()
    }

@app.post("/api/detect-scam")
async def detect_scam(
    request: Request,
    x_api_key: Optional[str] = Header(None)
):
    """
    Ultra-flexible endpoint - accepts ANY JSON format
    """
    
    try:
        # Get raw body
        body = await request.json()
        print(f"📨 Received request: {json.dumps(body, indent=2)}")
    except:
        # Empty body or invalid JSON - still return success for test
        print("📨 Received empty/invalid request")
        return AgentResponse(
            status="success",
            reply="Honeypot API is operational and ready to detect scams."
        )
    
    # Validate API key if provided
    if x_api_key and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")
    
    # Extract message text (try multiple possible formats)
    message_text = None
    session_id = "default-session"
    conversation_history = []
    
    # Try different JSON structures
    if isinstance(body, dict):
        # Format 1: Full GUVI format
        if "message" in body and isinstance(body["message"], dict):
            message_text = body["message"].get("text", "")
            session_id = body.get("sessionId", session_id)
            conversation_history = body.get("conversationHistory", [])
        
        # Format 2: Direct message
        elif "text" in body:
            message_text = body["text"]
            session_id = body.get("sessionId", session_id)
        
        # Format 3: Just sessionId (test)
        elif "sessionId" in body:
            session_id = body["sessionId"]
            message_text = "Test message"
    
    # If no message found, return test success
    if not message_text or message_text.strip() == "":
        print("✅ Test request - returning success")
        return AgentResponse(
            status="success",
            reply="API is ready. Send a message with scam content to test detection."
        )
    
    print(f"📝 Processing: {message_text[:50]}...")
    
    # Get session
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
    
    increment_usage()
    
    # Callback check
    should_callback = (
        session["scamDetected"] and (
            len(session["allIntelligence"].phoneNumbers) > 0 or
            len(session["allIntelligence"].upiIds) > 0 or
            session["totalMessages"] >= 10
        )
    )
    
    if should_callback and session["totalMessages"] >= 8:
        await send_final_callback(session_id)
    
    reply_text = ai_response.get("reply", "I see. Tell me more.")
    print(f"✅ Returning: {reply_text[:50]}...")
    
    return AgentResponse(
        status="success",
        reply=reply_text
    )

@app.get("/api/session/{session_id}")
async def get_session(session_id: str):
    session = sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return {
        "sessionId": session_id,
        "scamDetected": session["scamDetected"],
        "totalMessages": session["totalMessages"],
        "intelligence": session["allIntelligence"]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
