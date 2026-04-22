# AutoStream – Social-to-Lead Agentic Workflow
### Machine Learning Intern Assignment · ServiceHive / Inflx

---

## Overview

A production-ready conversational AI agent that converts social media conversations into qualified leads for **AutoStream** — a fictional SaaS video editing platform.

The agent can:
- 🗣️ Greet users and answer product/pricing questions using RAG
- 🎯 Detect user intent (greeting / inquiry / high-intent)
- 📋 Collect lead info (name, email, platform) conversationally
- ⚡ Fire a mock CRM lead-capture tool when all details are collected

---

## Project Structure

```
autostream-agent/
├── main.py                        # CLI entry point
├── requirements.txt
├── .env.example
│
├── agent/
│   └── graph.py                   # LangGraph state machine + nodes
│
├── tools/
│   └── lead_capture.py            # mock_lead_capture() tool
│
├── utils/
│   └── rag.py                     # Local RAG retrieval
│
└── knowledge_base/
    ├── autostream_kb.md           # Human-readable KB (chunked for RAG)
    └── autostream_kb.json         # Structured KB (pricing, policies)
```

---

## How to Run Locally

### 1. Clone & enter the repo
```bash
git clone https://github.com/your-username/autostream-agent.git
cd autostream-agent
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. API Key Configured
The API key has already been added to the `.env` file, so no further setup is required.

### 4. Run the agent
```bash
python main.py
```

### 5. Run as a website/server
```bash
uvicorn server:app --reload --host 0.0.0.0 --port 8000
```

Then open `http://localhost:8000` in your browser.

API endpoints:
- `POST /api/chat` with JSON `{ "message": "...", "session_id": "optional" }`
- `POST /api/reset/{session_id}` to clear one chat session
- `GET /health` for a basic health check

### 6. Submission readiness check
Before demo/submission, verify the server is responding properly.

Quick API smoke test:
```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message":"Hi there"}'
```

### Example Session
```
You: Hi there!
Alex: Hey! 👋 Welcome to AutoStream! I'm Alex ...

You: What's your pricing?
Alex: Great question! Here's a quick breakdown:
  • Basic Plan – $29/month: 10 videos, 720p, email support
  • Pro Plan   – $79/month: Unlimited videos, 4K, AI captions, 24/7 support
  ...

You: That sounds great, I want to try the Pro plan for my YouTube channel.
Alex: Awesome, let's get you set up! 🎬 What's your name?

You: Jamie Rivera
Alex: Great to meet you, Jamie! What's your email address?

You: jamie@example.com
Alex: Almost there! Which platform do you primarily create on?

You: YouTube
Alex: 🎉 You're all set, Jamie! ...

═══════════════════════════════════════
  ✅  LEAD CAPTURED SUCCESSFULLY
  Name     : Jamie Rivera
  Email    : jamie@example.com
  Platform : YouTube
  Lead ID  : LEAD-04291
═══════════════════════════════════════
```

---

## Architecture Explanation (~200 words)

### Why LangGraph?

LangGraph was chosen over a simple chain or plain AutoGen because it provides **explicit, inspectable state management** via a typed `AgentState` dict that persists across all conversation turns. Each node reads from and writes to this shared state, making it easy to track `intent`, `stage`, and the three lead fields (`lead_name`, `lead_email`, `lead_platform`) without hidden memory abstractions.

The graph has three nodes:

| Node | Role |
|------|------|
| `classify_node` | Calls the LLM to label intent; escalates `stage` to `qualifying` on high-intent |
| `rag_chat_node` | Runs keyword-based retrieval over the local KB, injects context into the system prompt, and generates a grounded response |
| `qualify_node` | Statefully collects the three lead fields one per turn; fires `mock_lead_capture()` only when all three are present |

A conditional edge after `classify_node` routes to either `rag_chat` or `qualify` depending on `stage`. This ensures the tool is **never triggered prematurely** — it only runs once all required fields are confirmed.

State is retained for the full session (5–6+ turns) because LangGraph's compiled graph accepts and returns the full `AgentState` on every `.invoke()` call.

---

## WhatsApp Deployment via Webhooks

To deploy this agent on WhatsApp:

### 1. Register a WhatsApp Business API account
Use Meta's Cloud API or a provider like Twilio/360dialog.

### 2. Set up a Webhook endpoint
Create a FastAPI (or Flask) server with two routes:

```python
from fastapi import FastAPI, Request
app = FastAPI()

# Verification handshake required by Meta
@app.get("/webhook")
async def verify(hub_mode, hub_challenge, hub_verify_token):
    if hub_verify_token == os.getenv("WA_VERIFY_TOKEN"):
        return int(hub_challenge)

# Incoming messages from WhatsApp
@app.post("/webhook")
async def receive_message(request: Request):
    body = await request.json()
    message = body["entry"][0]["changes"][0]["value"]["messages"][0]
    user_phone = message["from"]
    user_text  = message["text"]["body"]

    # Retrieve or create session state for this phone number
    state = session_store.get(user_phone, INITIAL_STATE.copy())
    state["messages"].append(HumanMessage(content=user_text))
    state = agent.invoke(state)
    session_store[user_phone] = state

    # Send reply back via WhatsApp Cloud API
    reply = get_last_ai_message(state)
    send_whatsapp_message(user_phone, reply)
    return {"status": "ok"}
```

### 3. Session persistence
Replace the in-memory `session_store` dict with **Redis** or a database (e.g., DynamoDB) so sessions survive server restarts.

### 4. Deploy
Host the webhook on any HTTPS endpoint — Railway, Render, AWS Lambda, or GCP Cloud Run all work well.

---

## Evaluation Checklist

| Criterion | Implemented |
|-----------|-------------|
| Intent detection (greeting / inquiry / high-intent) | ✅ LLM classifier in `classify_node` |
| RAG over local knowledge base | ✅ Keyword-scored chunk retrieval in `utils/rag.py` |
| State management across 5–6 turns | ✅ LangGraph `AgentState` TypedDict |
| Tool calling only when all fields collected | ✅ Guard logic in `qualify_node` |
| Clean code structure | ✅ Modular packages: agent / tools / utils / knowledge_base |
| Real-world deployability | ✅ WhatsApp webhook architecture documented above |

---

## Tech Stack

| Layer | Choice | Reason |
|-------|--------|--------|
| Framework | LangGraph | Explicit state machine, inspectable routing |
| LLM | GPT-4o-mini / Gemini 2.0 Flash / Claude Haiku | Cost-effective, fast inference |
| RAG | Local keyword retrieval (no vector DB) | Zero infra overhead for assignment scope |
| State | LangGraph `AgentState` TypedDict | Type-safe, serializable |
| Lead tool | `mock_lead_capture()` | Simulates CRM POST |

---

*Built for ServiceHive · Inflx ML Intern Assignment*
