# 🛡️ Agentic Honeypot for Scam Detection

AI-powered scam detection system for GUVI AI Impact Competition - Problem Statement 2

## 🎯 What This Does

This API:
- ✅ Detects scam messages in real-time
- 🤖 Engages scammers using AI agent
- 🔍 Extracts intelligence (bank accounts, UPI IDs, phone numbers, etc.)
- 📊 Reports results to GUVI platform
- 🌐 Supports SMS, WhatsApp, Email channels
- 🗣️ Multi-language support (English, Hindi, Tamil, etc.)

## 🚀 Quick Start

### 1. Get API Key
- Visit [Google AI Studio](https://aistudio.google.com/app/apikey)
- Create API key
- Save it securely

### 2. Deploy to Render.com
- Fork this repository
- Sign up at [Render.com](https://render.com)
- Connect your repository
- Add environment variables:
  - `GEMINI_API_KEY`: Your Gemini API key
  - `API_KEY`: Your custom secret key

### 3. Test Your API

```bash
curl https://your-app.onrender.com/
```

## 📖 Full Documentation

See [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md) for complete step-by-step instructions.

## 🏗️ Architecture

```
GUVI Platform → FastAPI → Gemini AI → Intelligence Extraction → Callback
```

## 🛠️ Tech Stack

- **Framework**: FastAPI (Python)
- **AI Model**: Google Gemini 2.0 Flash
- **Hosting**: Render.com (Free tier)
- **Storage**: In-memory (no database needed for MVP)

## 💡 Features

### Scam Detection
- Urgency tactics
- Payment requests
- Account threats
- Phishing links
- Impersonation

### Intelligence Extraction
- Bank account numbers
- UPI IDs
- Phone numbers
- URLs/Links
- Suspicious keywords

### Conversational AI
- Natural language responses
- Context awareness
- Multi-turn conversations
- Persona maintenance

## 🧪 Testing

Run test suite:
```bash
chmod +x test_requests.sh
./test_requests.sh
```

Or test manually:
```bash
curl -X POST https://your-app.onrender.com/api/detect-scam \
-H "Content-Type: application/json" \
-H "x-api-key: YOUR_API_KEY" \
-d '{
  "sessionId": "test-001",
  "message": {
    "sender": "scammer",
    "text": "Your account will be blocked. Call 9876543210",
    "timestamp": "2026-02-03T10:00:00Z"
  },
  "conversationHistory": []
}'
```

## 📊 API Endpoints

### `GET /`
Health check

### `POST /api/detect-scam`
Main scam detection endpoint

**Headers:**
- `Content-Type: application/json`
- `x-api-key: YOUR_API_KEY`

**Body:**
```json
{
  "sessionId": "unique-session-id",
  "message": {
    "sender": "scammer",
    "text": "Message content",
    "timestamp": "ISO-8601"
  },
  "conversationHistory": [],
  "metadata": {
    "channel": "SMS",
    "language": "English",
    "locale": "IN"
  }
}
```

**Response:**
```json
{
  "status": "success",
  "reply": "Conversational response"
}
```

### `GET /api/session/{sessionId}`
Get session details (for debugging)

## 💰 Cost

**FREE** for competition:
- Gemini API: 1,500 requests/day free
- Render.com: 750 hours/month free
- Total: $0

## 📝 Competition Requirements

✅ REST API with JSON format  
✅ API key authentication  
✅ Multi-turn conversation support  
✅ Intelligence extraction  
✅ GUVI callback integration  
✅ Scam detection  
✅ Ethical behavior  

## 🏆 Built For

GUVI AI Impact Competition - Problem Statement 2  
**Challenge**: Agentic Honey-Pot for Scam Detection & Intelligence Extraction

## 📄 License

MIT License - Built for educational purposes

## 🤝 Support

For issues or questions, see the [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md)

---

**Made with ❤️ for GUVI AI Impact Competition 2026**
