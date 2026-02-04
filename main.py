"""
Agentic Honeypot for Scam Detection - PRODUCTION VERSION
Solution B: FastAPI + Google Gemini API with Budget Controls

GUVI AI Impact Competition 2026 - Problem Statement 2
Version: 2.0 (with budget controls)
"""

from fastapi import FastAPI, Header, HTTPException, Request
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
import google.generativeai as genai
import os
import json
from datetime import datetime, date
import httpx
from contextlib import asynccontextmanager

# ============================================================================
# CONFIGURATION
# ============================================================================

# API Keys
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
API_KEY = os.getenv("API_KEY", "your-secret-api-key-here")
GUVI_CALLBACK_URL = "https://hackathon.guvi.in/api/updateHoneyPotFinalResult"

# Budget Controls (Set these in Render.com environment variables)
MAX_DAILY_REQUESTS = int(os.getenv("MAX_DAILY_REQUESTS", "300"))
MAX_SESSION_MESSAGES = int(os.getenv("MAX_SESSION_MESSAGES", "20"))
WARNING_THRESHOLD = int(os.getenv("WARNING_THRESHOLD", "250"))  # Warn at 250/300

# Configure Gemini
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

# In-memory session storage
sessions: Dict[str, Dict] = {}

# Usage tracking
usage_tracker = {
    "date": str(date.today()),
    "count": 0,
    "warned": False
}

session_message_counts: Dict[str, int] = {}

# ============================================================================
# DATA MODELS (GUVI Required Format)
# ============================================================================

class Message(BaseModel):
    sender: str  # "scammer" or "user"
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

class FinalResultPayload(BaseModel):
    sessionId: str
    scamDetected: bool
    totalMessagesExchanged: int
    extractedIntelligence: ExtractedIntelligence
    agentNotes: str

# ============================================================================
# BUDGET CONTROL FUNCTIONS
# ============================================================================

def check_daily_budget() -> tuple[bool, str]:
    """
    Check if daily request limit has been reached
    Returns: (allowed: bool, message: str)
    """
    today = str(date.today())
    
    # Reset counter if new day
    if usage_tracker["date"] != today:
        usage_tracker["date"] = today
        usage_tracker["count"] = 0
        usage_tracker["warned"] = False
        print(f"🔄 Daily counter reset for {today}")
    
    # Check if limit reached
    if usage_tracker["count"] >= MAX_DAILY_REQUESTS:
        return False, f"Daily quota of {MAX_DAILY_REQUESTS} requests exceeded. Try again tomorrow."
    
    # Warning at threshold (e.g., 250/300)
    if usage_tracker["count"] >= WARNING_THRESHOLD and not usage_tracker["warned"]:
        print(f"⚠️ WARNING: {usage_tracker['count']}/{MAX_DAILY_REQUESTS} requests used today!")
        usage_tracker["warned"] = True
    
    return True, f"Request {usage_tracker['count'] + 1}/{MAX_DAILY_REQUESTS}"

def increment_usage():
    """Increment daily usage counter"""
    usage_tracker["count"] += 1
    print(f"📊 Daily Usage: {usage_tracker['count']}/{MAX_DAILY_REQUESTS} requests")

def check_session_limit(session_id: str) -> tuple[bool, str]:
    """
    Check if session message limit has been reached
    Returns: (allowed: bool, message: str)
    """
    # Initialize counter for new sessions
    if session_id not in session_message_counts:
        session_message_counts[session_id] = 0
    
    current_count = session_message_counts[session_id]
    
    # Check limit
    if current_count >= MAX_SESSION_MESSAGES:
        return False, f"Session limit of {MAX_SESSION_MESSAGES} messages reached"
    
    return True, f"Session message {current_count + 1}/{MAX_SESSION_MESSAGES}"

def increment_session_count(session_id: str):
    """Increment session message counter"""
    session_message_counts[session_id] = session_message_counts.get(session_id, 0) + 1
    print(f"📝 Session {session_id}: {session_message_counts[session_id]}/{MAX_SESSION_MESSAGES} messages")

def get_usage_stats() -> Dict:
    """Get current usage statistics"""
    return {
        "date": usage_tracker["date"],
        "dailyCount": usage_tracker["count"],
        "dailyLimit": MAX_DAILY_REQUESTS,
        "percentageUsed": round((usage_tracker["count"] / MAX_DAILY_REQUESTS) * 100, 2),
        "activeSessions": len(sessions),
        "totalSessionMessages": sum(session_message_counts.values())
    }

# ============================================================================
# SYSTEM PROMPT FOR GEMINI
# ============================================================================

SYSTEM_PROMPT = """You are an intelligent scam detection and honeypot agent. Your mission is to detect scams and extract intelligence.

CRITICAL INSTRUCTIONS:

1. SCAM DETECTION
   - Analyze messages for: urgency tactics, payment requests, threats, phishing links, impersonation
   - Common patterns: bank fraud, UPI fraud, fake refunds, account suspension threats, lottery scams
   - Look for: urgent language, requests for personal info, suspicious links, payment demands

2. ENGAGEMENT STRATEGY (Only if scam detected)
   - Act as a believable, slightly naive victim
   - Ask natural follow-up questions
   - Show concern but also curiosity
   - NEVER reveal you're an AI or that you detected a scam
   - Keep responses SHORT (1-3 sentences, conversational)
   - Match the tone of the channel (SMS = casual, formal if needed)

3. INTELLIGENCE EXTRACTION
   Extract and accumulate:
   - Bank account numbers (format: XXXX-XXXX-XXXX or account numbers)
   - UPI IDs (format: name@bank or phone@upi)
   - Phone numbers (any format)
   - URLs/Links (http://, https://, bit.ly, shortened links)
   - Payment app names (PhonePe, Paytm, GooglePay, etc.)
   - Suspicious keywords (urgent, verify, blocked, suspended, refund, prize, lottery)
   - Names/aliases used by scammer

4. RESPONSE FORMAT
   You MUST return ONLY a valid JSON object. No markdown, no explanation, just JSON:

   {
     "scamDetected": true,
     "reply": "Your conversational response",
     "extractedIntelligence": {
       "bankAccounts": ["found accounts"],
       "upiIds": ["found UPIs"],
       "phishingLinks": ["found URLs"],
       "phoneNumbers": ["found numbers"],
       "suspiciousKeywords": ["keywords"]
     },
     "confidence": 0.95,
     "agentNotes": "Brief analysis"
   }

5. CONVERSATION EXAMPLES

Example 1:
Scammer: "Your bank account will be blocked. Call 9876543210 now."
Your response:
{
  "scamDetected": true,
  "reply": "Why would it be blocked? Which bank are you from?",
  "extractedIntelligence": {
    "phoneNumbers": ["9876543210"],
    "suspiciousKeywords": ["blocked", "call now"]
  },
  "confidence": 0.95,
  "agentNotes": "Classic urgency tactic with phone number"
}

Example 2:
Scammer: "Send UPI ID to receive refund."
Your response:
{
  "scamDetected": true,
  "reply": "Refund for what? I don't remember any transaction.",
  "extractedIntelligence": {
    "suspiciousKeywords": ["UPI ID", "refund"]
  },
  "confidence": 0.88,
  "agentNotes": "Refund scam, requesting payment details"
}

Example 3 (Legitimate):
User: "Hey, still on for dinner tonight?"
Your response:
{
  "scamDetected": false,
  "reply": "Yes, see you at 7!",
  "extractedIntelligence": {},
  "confidence": 0.05,
  "agentNotes": "Normal conversation"
}

CRITICAL RULES:
- Always return valid JSON, nothing else
- Be conversational and natural
- Extract information progressively
- Update extractedIntelligence with cumulative data from entire conversation
- Never break character
- If unsure, ask clarifying questions to gather more info
"""

# ============================================================================
# GEMINI INTERACTION
# ============================================================================

def create_conversation_context(conversation_history: List[Message], current_message: Message) -> str:
    """Build conversation context for Gemini"""
    context = "CONVERSATION HISTORY:\n\n"
    
    for msg in conversation_history:
        sender_label = "SCAMMER" if msg.sender == "scammer" else "YOU (as victim)"
        context += f"{sender_label}: {msg.text}\n"
    
    context += f"\nNEW MESSAGE FROM SCAMMER:\n{current_message.text}\n\n"
    context += "Analyze this message and respond with the JSON format specified."
    
    return context

async def analyze_with_gemini(conversation_history: List[Message], current_message: Message) -> Dict:
    """Send conversation to Gemini and get response"""
    
    try:
        # Initialize Gemini model
        model = genai.GenerativeModel(
            model_name="gemini-2.0-flash-exp",
            system_instruction=SYSTEM_PROMPT
        )
        
        # Build conversation context
        prompt = create_conversation_context(conversation_history, current_message)
        
        # Generate response
        response = model.generate_content(
            prompt,
            generation_config=genai.GenerationConfig(
                temperature=0.7,
                top_p=0.9,
                top_k=40,
                max_output_tokens=1024,
            )
        )
        
        # Extract JSON from response
        response_text = response.text.strip()
        
        # Remove markdown code blocks if present
        if response_text.startswith("```json"):
            response_text = response_text[7:]
        if response_text.startswith("```"):
            response_text = response_text[3:]
        if response_text.endswith("```"):
            response_text = response_text[:-3]
        
        response_text = response_text.strip()
        
        # Parse JSON
        result = json.loads(response_text)
        
        print(f"✅ Gemini response received successfully")
        return result
        
    except json.JSONDecodeError as e:
        print(f"❌ JSON Parse Error: {e}")
        print(f"Response text: {response_text}")
        # Fallback response
        return {
            "scamDetected": False,
            "reply": "I didn't quite understand. Could you explain again?",
            "extractedIntelligence": {},
            "confidence": 0.3,
            "agentNotes": "Failed to parse AI response"
        }
    except Exception as e:
        print(f"❌ Gemini API Error: {e}")
        return {
            "scamDetected": False,
            "reply": "Sorry, can you repeat that?",
            "extractedIntelligence": {},
            "confidence": 0.0,
            "agentNotes": f"Error: {str(e)}"
        }

# ============================================================================
# SESSION MANAGEMENT
# ============================================================================

def get_or_create_session(session_id: str) -> Dict:
    """Get existing session or create new one"""
    if session_id not in sessions:
        sessions[session_id] = {
            "messages": [],
            "scamDetected": False,
            "totalMessages": 0,
            "allIntelligence": ExtractedIntelligence(),
            "agentNotes": []
        }
        print(f"🆕 New session created: {session_id}")
    return sessions[session_id]

def merge_intelligence(existing: ExtractedIntelligence, new_data: Dict) -> ExtractedIntelligence:
    """Merge new intelligence with existing, avoiding duplicates"""
    
    intel = new_data.get("extractedIntelligence", {})
    
    # Merge each field, removing duplicates
    existing.bankAccounts = list(set(existing.bankAccounts + intel.get("bankAccounts", [])))
    existing.upiIds = list(set(existing.upiIds + intel.get("upiIds", [])))
    existing.phishingLinks = list(set(existing.phishingLinks + intel.get("phishingLinks", [])))
    existing.phoneNumbers = list(set(existing.phoneNumbers + intel.get("phoneNumbers", [])))
    existing.suspiciousKeywords = list(set(existing.suspiciousKeywords + intel.get("suspiciousKeywords", [])))
    
    return existing

async def send_final_callback(session_id: str):
    """Send final results to GUVI callback endpoint"""
    
    session = sessions.get(session_id)
    if not session:
        print(f"❌ Session {session_id} not found for callback")
        return
    
    # Prepare payload
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
        "agentNotes": " | ".join(session["agentNotes"][-5:])  # Last 5 notes
    }
    
    # Send to GUVI
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                GUVI_CALLBACK_URL,
                json=payload,
                timeout=10.0
            )
            print(f"✅ Callback sent for session {session_id}: {response.status_code}")
            print(f"📤 Response: {response.text}")
    except Exception as e:
        print(f"❌ Failed to send callback: {e}")

# ============================================================================
# FASTAPI APPLICATION
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events"""
    print("=" * 60)
    print("🚀 Agentic Honeypot API Starting...")
    print("=" * 60)
    print(f"✅ Gemini API configured: {bool(GEMINI_API_KEY)}")
    print(f"📊 Daily request limit: {MAX_DAILY_REQUESTS}")
    print(f"📝 Session message limit: {MAX_SESSION_MESSAGES}")
    print(f"⚠️  Warning threshold: {WARNING_THRESHOLD}")
    print("=" * 60)
    yield
    print("=" * 60)
    print("👋 Honeypot API shutting down...")
    print(f"📊 Final stats: {usage_tracker['count']} requests today")
    print("=" * 60)

app = FastAPI(
    title="Agentic Honeypot API",
    description="AI-powered scam detection and intelligence extraction with budget controls",
    version="2.0.0",
    lifespan=lifespan
)

# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.get("/")
async def root():
    """Health check endpoint with usage stats"""
    stats = get_usage_stats()
    return {
        "status": "running",
        "service": "Agentic Honeypot for Scam Detection",
        "version": "2.0.0",
        "features": ["budget-controls", "usage-tracking"],
        "usage": stats
    }

@app.get("/api/usage")
async def get_usage(x_api_key: str = Header(...)):
    """Get detailed usage statistics"""
    
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")
    
    stats = get_usage_stats()
    stats["sessionDetails"] = {
        session_id: {
            "messageCount": session_message_counts.get(session_id, 0),
            "scamDetected": sessions[session_id]["scamDetected"],
            "totalMessages": sessions[session_id]["totalMessages"]
        }
        for session_id in sessions.keys()
    }
    
    return stats

@app.post("/api/detect-scam")
async def detect_scam(
    request: IncomingRequest,
    x_api_key: str = Header(..., description="API Key for authentication")
):
    """
    Main endpoint for scam detection and engagement
    
    This endpoint:
    1. Validates budget limits (daily + session)
    2. Receives incoming message from GUVI platform
    3. Analyzes for scam intent
    4. Engages scammer if detected
    5. Extracts intelligence
    6. Returns conversational reply
    7. Sends final callback when conversation concludes
    """
    
    # Validate API key
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")
    
    # ========== BUDGET CONTROLS ==========
    
    # Check daily limit
    daily_allowed, daily_msg = check_daily_budget()
    if not daily_allowed:
        print(f"🚫 Daily quota exceeded")
        raise HTTPException(status_code=429, detail=daily_msg)
    
    # Check session limit
    session_allowed, session_msg = check_session_limit(request.sessionId)
    if not session_allowed:
        print(f"🚫 Session limit reached for {request.sessionId}")
        # End conversation gracefully
        return AgentResponse(
            status="success",
            reply="Thank you for your time. Have a good day."
        )
    
    # ========== NORMAL PROCESSING ==========
    
    # Get or create session
    session = get_or_create_session(request.sessionId)
    
    # Add current message to session
    session["messages"].append(request.message)
    session["totalMessages"] += 1
    
    # Analyze with Gemini
    print(f"🤖 Analyzing message from session {request.sessionId}")
    ai_response = await analyze_with_gemini(
        request.conversationHistory,
        request.message
    )
    
    # Update session with results
    if ai_response.get("scamDetected", False):
        session["scamDetected"] = True
        print(f"🚨 Scam detected in session {request.sessionId}")
    
    # Merge intelligence
    session["allIntelligence"] = merge_intelligence(
        session["allIntelligence"],
        ai_response
    )
    
    # Add agent notes
    if "agentNotes" in ai_response:
        session["agentNotes"].append(ai_response["agentNotes"])
    
    # Add agent's reply to session
    agent_message = Message(
        sender="user",
        text=ai_response.get("reply", "Can you tell me more?"),
        timestamp=datetime.utcnow().isoformat() + "Z"
    )
    session["messages"].append(agent_message)
    session["totalMessages"] += 1
    
    # Increment counters AFTER successful processing
    increment_usage()
    increment_session_count(request.sessionId)
    
    # Check if we should send final callback
    should_callback = (
        session["scamDetected"] and (
            len(session["allIntelligence"].phoneNumbers) > 0 or
            len(session["allIntelligence"].upiIds) > 0 or
            len(session["allIntelligence"].bankAccounts) > 0 or
            session["totalMessages"] >= 15  # After 15 messages, conclude
        )
    )
    
    if should_callback and session["totalMessages"] >= 8:  # Minimum engagement
        print(f"📞 Sending final callback for session {request.sessionId}")
        await send_final_callback(request.sessionId)
    
    # Return response in GUVI format
    return AgentResponse(
        status="success",
        reply=ai_response.get("reply", "I see. Tell me more.")
    )

@app.get("/api/session/{session_id}")
async def get_session(session_id: str, x_api_key: str = Header(...)):
    """Get session details (for debugging)"""
    
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
        "conversationLength": len(session["messages"]),
        "messageCount": session_message_counts.get(session_id, 0),
        "agentNotes": session["agentNotes"]
    }

@app.post("/api/reset-usage")
async def reset_usage(x_api_key: str = Header(...)):
    """Reset usage counters (for testing only)"""
    
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")
    
    usage_tracker["count"] = 0
    usage_tracker["warned"] = False
    session_message_counts.clear()
    
    print("🔄 Usage counters reset")
    
    return {
        "status": "success",
        "message": "Usage counters reset",
        "currentUsage": get_usage_stats()
    }

# ============================================================================
# MAIN (for local testing)
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting Honeypot API on http://localhost:8000")
    print("📖 API docs available at http://localhost:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000)
