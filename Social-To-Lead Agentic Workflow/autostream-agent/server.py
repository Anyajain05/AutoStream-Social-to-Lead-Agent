"""
server.py
FastAPI web server for the AutoStream conversational agent.

Run locally:
  uvicorn server:app --reload --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

import os
import sys
import uuid
from typing import Dict

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

# Ensure project root is on the path
sys.path.insert(0, os.path.dirname(__file__))

# Load environment variables before importing the graph
load_dotenv()

from langchain_core.messages import HumanMessage

from agent.graph import agent


app = FastAPI(title="AutoStream Agent API", version="1.0.0")

INITIAL_STATE = {
    "messages": [],
    "intent": "greeting",
    "stage": "chat",
    "lead_name": None,
    "lead_email": None,
    "lead_platform": None,
}

# In-memory sessions for local/dev use.
sessions: Dict[str, dict] = {}


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2000)
    session_id: str | None = None


class ChatResponse(BaseModel):
    session_id: str
    reply: str
    stage: str
    intent: str


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/api/chat", response_model=ChatResponse)
def chat(payload: ChatRequest) -> ChatResponse:
    session_id = payload.session_id or str(uuid.uuid4())
    state = sessions.get(session_id, INITIAL_STATE.copy())

    state["messages"] = state.get("messages", []) + [HumanMessage(content=payload.message.strip())]
    state = agent.invoke(state)
    sessions[session_id] = state

    last_ai = next(
        (m for m in reversed(state["messages"]) if getattr(m, "type", "") == "ai"),
        None,
    )
    if not last_ai:
        raise HTTPException(status_code=500, detail="No AI response produced")

    return ChatResponse(
        session_id=session_id,
        reply=last_ai.content,
        stage=state.get("stage", "chat"),
        intent=state.get("intent", "inquiry"),
    )


@app.post("/api/reset/{session_id}")
def reset_session(session_id: str) -> dict:
    sessions.pop(session_id, None)
    return {"ok": True}


@app.get("/", response_class=HTMLResponse)
def home() -> str:
    return """<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>AutoStream Agent</title>
  <style>
    :root {
      --bg-a: #f4f7ff;
      --bg-b: #fff8ef;
      --ink: #171717;
      --brand: #0f766e;
      --brand-2: #1d4ed8;
      --card: #ffffffee;
      --line: #e5e7eb;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      color: var(--ink);
      font-family: \"Segoe UI\", Tahoma, Geneva, Verdana, sans-serif;
      background: radial-gradient(1200px 500px at 5% 0%, #dbeafe, transparent 60%),
                  radial-gradient(1000px 450px at 95% 100%, #ffe4c7, transparent 60%),
                  linear-gradient(120deg, var(--bg-a), var(--bg-b));
      min-height: 100vh;
      display: grid;
      place-items: center;
      padding: 20px;
    }
    .app {
      width: min(900px, 100%);
      background: var(--card);
      border: 1px solid var(--line);
      border-radius: 18px;
      backdrop-filter: blur(10px);
      box-shadow: 0 10px 40px rgba(0, 0, 0, 0.08);
      overflow: hidden;
      animation: pop 320ms ease-out;
    }
    @keyframes pop {
      from { transform: translateY(8px); opacity: 0; }
      to { transform: translateY(0); opacity: 1; }
    }
    header {
      padding: 18px 20px;
      border-bottom: 1px solid var(--line);
      background: linear-gradient(90deg, rgba(15,118,110,0.08), rgba(29,78,216,0.08));
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
    }
    h1 { margin: 0; font-size: 1.05rem; letter-spacing: 0.2px; }
    #meta { font-size: 0.82rem; color: #525252; }
    #chat {
      height: 58vh;
      min-height: 380px;
      overflow: auto;
      padding: 16px;
      display: flex;
      flex-direction: column;
      gap: 10px;
    }
    .msg {
      max-width: 78%;
      padding: 10px 12px;
      border-radius: 12px;
      line-height: 1.45;
      white-space: pre-wrap;
    }
    .me {
      align-self: flex-end;
      background: #dbeafe;
      border: 1px solid #bfdbfe;
    }
    .bot {
      align-self: flex-start;
      background: #ecfeff;
      border: 1px solid #bae6fd;
    }
    form {
      display: grid;
      grid-template-columns: 1fr auto auto;
      gap: 10px;
      padding: 14px;
      border-top: 1px solid var(--line);
      background: #ffffffcc;
    }
    input {
      border: 1px solid #cbd5e1;
      border-radius: 10px;
      padding: 11px 12px;
      font-size: 0.95rem;
      outline: none;
    }
    input:focus { border-color: var(--brand-2); box-shadow: 0 0 0 3px #dbeafe; }
    button {
      border: none;
      border-radius: 10px;
      padding: 0 14px;
      cursor: pointer;
      font-weight: 600;
      transition: transform 0.12s ease, opacity 0.12s ease;
    }
    button:hover { transform: translateY(-1px); }
    #send { background: var(--brand-2); color: #fff; }
    #reset { background: #f3f4f6; color: #111827; }
    @media (max-width: 700px) {
      #chat { height: 62vh; min-height: 320px; }
      .msg { max-width: 90%; }
      form { grid-template-columns: 1fr auto; }
      #reset { grid-column: 1 / -1; height: 38px; }
    }
  </style>
</head>
<body>
  <section class=\"app\">
    <header>
      <h1>AutoStream AI Sales Assistant</h1>
      <div id=\"meta\">Session: <span id=\"sid\">new</span></div>
    </header>
    <div id=\"chat\"></div>
    <form id=\"form\">
      <input id=\"text\" placeholder=\"Ask about plans or say: I want to sign up\" autocomplete=\"off\" />
      <button id=\"send\" type=\"submit\">Send</button>
      <button id=\"reset\" type=\"button\">Reset Session</button>
    </form>
  </section>
  <script>
    const chat = document.getElementById('chat');
    const form = document.getElementById('form');
    const text = document.getElementById('text');
    const sid = document.getElementById('sid');
    const reset = document.getElementById('reset');
    let sessionId = null;

    const add = (content, who) => {
      const div = document.createElement('div');
      div.className = `msg ${who}`;
      div.textContent = content;
      chat.appendChild(div);
      chat.scrollTop = chat.scrollHeight;
    };

    add('Hi, I am Alex from AutoStream. How can I help today?', 'bot');

    form.addEventListener('submit', async (e) => {
      e.preventDefault();
      const message = text.value.trim();
      if (!message) return;
      add(message, 'me');
      text.value = '';

      try {
        const res = await fetch('/api/chat', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ message, session_id: sessionId })
        });
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const data = await res.json();
        sessionId = data.session_id;
        sid.textContent = sessionId.slice(0, 8);
        add(data.reply, 'bot');
      } catch (err) {
        add('Server error. Check your API key/quota and server logs.', 'bot');
      }
    });

    reset.addEventListener('click', async () => {
      if (sessionId) {
        await fetch(`/api/reset/${sessionId}`, { method: 'POST' });
      }
      sessionId = null;
      sid.textContent = 'new';
      chat.innerHTML = '';
      add('New session started. How can I help?', 'bot');
    });
  </script>
</body>
</html>"""