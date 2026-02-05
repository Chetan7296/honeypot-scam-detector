"""
FINAL WORKING VERSION - Agentic Honeypot
Uses gemini-pro (the most stable, widely available model)
"""

from fastapi import FastAPI, Header, HTTPException
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
WARNING_THRESHOLD = int(os.getenv("WARNING_THRESHOLD", "250"))

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

sessions: Dict[str, Dict] = {}
usage_tracker = {"date": str(date.today()), "count": 0, "warned": False}
session_message_counts: Dict[str, int] = {}

# Data Models
class Message(BaseModel):
    sender: str
    text: str
    timestamp: str

class Metadata(BaseModel):
    channel: Optional[str] = "SMS"
    language: Optional[str] = "English"
    locale: Optional[str] = "IN"

class IncomingRequest(BaseModel):
    sessionId: str
    message: Message
    conversationHistory: List[Message] = []
    metadata: Optional[Metadata] = None

class ExtractedIntelligence(BaseModel):
    bankAccounts: List[str] = []
    upiIds: List[str] = []
    phishingLinks: List[str] = []
    phoneNumbers: List[str] = []
    suspiciousKeywords: List[str] = []

class AgentResponse(BaseModel):
    status: str
    reply: str

# Budget Controls
def check_daily_budget() -> tuple[bool, str]:
    today = str(date.today())
    if usage_tracker["date"] != today:
        usage_tracker["date"] = today
        usage_tracker["count"] = 0
        usage_tracker["warned"] = False
    
    if usage_tracker["count"] >= MAX_DAILY_REQUESTS:
        return False, f"Daily quota exceeded"
    
    if usage_tracker["count"] >= WARNING_THRESHOLD and not usage_tracker["warned"]:
        print(f"⚠️ WARNING: {usage_tracker['count']}/{MAX_DAILY_REQUESTS} used!")
        usage_tracker["warned"] = True
    
    return True, f"OK"

def increment_usage():
    usage_tracker["count"] += 1
    print(f"📊 Usage: {usage_tracker['count']}/{MAX_DAILY_REQUESTS}")

def check_session_limit(session_id: str) -> tuple[bool, str]:
    if session_id not in session_message_counts:
        session_message_counts[session_id] = 0
    
    if session_message_counts[session_id] >= MAX_SESSION_MESSAGES:
        return False, "Session limit reached"
    
    return True, "OK"

def increment_session_count(session_id: str):
    session_message_counts[session_id] = session_message_counts.get(session_id, 0) + 1

def get_usage_stats() -> Dict:
    return {
        "date": usage_tracker["date"],
        "dailyCount": usage_tracker["count"],
        "dailyLimit": MAX_DAILY_REQUESTS,
        "percentageUsed": round((usage_tracker["count"] / MAX_DAILY_REQUESTS) * 100, 2),
        "activeSessions": len(sessions)
    }

# Gemini Interaction
def create_conversation_context(conversation_history: List[Message], current_message: Message) -> str:
    context = "You are a scam detection agent. Analyze this message and respond ONLY with valid JSON.\n\n"
    context += "Response format:\n"
    context += '{"scamDetected": true/false, "reply": "your conversational response", '
    context += '"extractedIntelligence": {"phoneNumbers": [], "upiIds": [], "bankAccounts": [], '
    context += '"phishingLinks": [], "suspiciousKeywords": []}, "confidence": 0.95, "agentNotes": "analysis"}\n\n'
    
    context += "Rules:\n"
    context += "- Act as a curious victim if scam detected\n"
    context += "- Keep reply SHORT (1-3 sentences)\n"
    context += "- Extract phone numbers, UPI IDs, bank accounts, links\n"
    context += "- NEVER reveal you're an AI\n\n"
    
    if conversation_history:
        context += "Previous conversation:\n"
        for msg in conversation_history:
            sender_label = "SCAMMER" if msg.sender == "scammer" else "YOU"
            context += f"{sender_label}: {msg.text}\n"
    
    context += f"\nNew message from scammer:\n{current_message.text}\n\n"
    context += "Respond with JSON only:"
    
    return context

async def analyze_with_gemini(conversation_history: List[Message], current_message: Message) -> Dict:
    try:
        print("🤖 Calling Gemini...")
        
        # USE GEMINI-PRO - MOST STABLE MODEL
        model = genai.GenerativeModel(model_name="gemini-pro")

        prompt = create_conversation_context(conversation_history, current_message)

        response = model.generate_content(
            [prompt],
            generation_config=genai.GenerationConfig(
                temperature=0.7,
                max_output_tokens=1024,
            )
        )

        
        response_text = response.text.strip()
        print(f"📥 Response: {response_text[:100]}...")
        
        # Clean markdown
        if response_text.startswith("```json"):
            response_text = response_text[7:]
        if response_text.startswith("```"):
            response_text = response_text[3:]
        if response_text.endswith("```"):
            response_text = response_text[:-3]
        
        response_text = response_text.strip()
        
        result = json.loads(response_text)
        print(f"✅ Scam detected: {result.get('scamDetected')}")
        return result
        
    except json.JSONDecodeError as e:
        print(f"❌ JSON error: {e}")
        return {
            "scamDetected": False,
            "reply": "Could you explain more?",
            "extractedIntelligence": {},
            "confidence": 0.3,
            "agentNotes": "Parse error"
        }
    except Exception as e:
        print(f"❌ Error: {e}")
        return {
            "scamDetected": False,
            "reply": "Sorry, can you repeat that?",
            "extractedIntelligence": {},
            "confidence": 0.0,
            "agentNotes": str(e)
        }

# Session Management
def get_or_create_session(session_id: str) -> Dict:
    if session_id not in sessions:
        sessions[session_id] = {
            "messages": [],
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
    print(f"✅ Gemini configured: {bool(GEMINI_API_KEY)}")
    yield
    print("👋 Shutting down")

app = FastAPI(title="Honeypot API", version="2.0", lifespan=lifespan)

@app.get("/")
async def root():
    return {
        "status": "running",
        "service": "Agentic Honeypot",
        "version": "2.0",
        "usage": get_usage_stats()
    }

@app.post("/api/detect-scam")
async def detect_scam(request: IncomingRequest, x_api_key: str = Header(...)):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")
    
    # Budget checks
    daily_allowed, _ = check_daily_budget()
    if not daily_allowed:
        raise HTTPException(status_code=429, detail="Daily quota exceeded")
    
    session_allowed, _ = check_session_limit(request.sessionId)
    if not session_allowed:
        return AgentResponse(status="success", reply="Thank you for your time.")
    
    # Process
    session = get_or_create_session(request.sessionId)
    session["messages"].append(request.message)
    session["totalMessages"] += 1
    
    # Call Gemini
    ai_response = await analyze_with_gemini(request.conversationHistory, request.message)
    
    # Update session
    if ai_response.get("scamDetected"):
        session["scamDetected"] = True
    
    session["allIntelligence"] = merge_intelligence(session["allIntelligence"], ai_response)
    
    if "agentNotes" in ai_response:
        session["agentNotes"].append(ai_response["agentNotes"])
    
    agent_message = Message(
        sender="user",
        text=ai_response.get("reply", "Tell me more."),
        timestamp=datetime.utcnow().isoformat() + "Z"
    )
    session["messages"].append(agent_message)
    session["totalMessages"] += 1
    
    increment_usage()
    increment_session_count(request.sessionId)
    
    # Callback check
    should_callback = (
        session["scamDetected"] and (
            len(session["allIntelligence"].phoneNumbers) > 0 or
            len(session["allIntelligence"].upiIds) > 0 or
            len(session["allIntelligence"].bankAccounts) > 0 or
            session["totalMessages"] >= 15
        )
    )
    
    if should_callback and session["totalMessages"] >= 8:
        await send_final_callback(request.sessionId)
    
    return AgentResponse(
        status="success",
        reply=ai_response.get("reply", "I see.")
    )

@app.get("/api/session/{session_id}")
async def get_session(session_id: str, x_api_key: str = Header(...)):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")
    
    session = sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return {
        "sessionId": session_id,
        "scamDetected": session["scamDetected"],
        "totalMessages": session["totalMessages"],
        "intelligence": session["allIntelligence"],
        "agentNotes": session["agentNotes"]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
