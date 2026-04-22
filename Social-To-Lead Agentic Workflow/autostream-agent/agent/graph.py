"""
agent/graph.py
LangGraph-based conversational agent for AutoStream (Inflx assignment).

State machine:
  GREETING → ANSWERING → QUALIFYING (collect name/email/platform) → LEAD_CAPTURED
"""
from __future__ import annotations

import os
import re
from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from tools.lead_capture import mock_lead_capture
from utils.rag import get_full_kb_summary, retrieve_context

# ── LLM ──────────────────────────────────────────────────────────────────────
def _build_llm():
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("LLM_API_KEY")
    model   = os.getenv("LLM_MODEL", "gpt-4o-mini")

    if api_key and "gpt" in model:
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=model, temperature=0.3, api_key=api_key)

    # Fallback: Google Gemini
    google_key = os.getenv("GOOGLE_API_KEY")
    if google_key:
        from langchain_google_genai import ChatGoogleGenerativeAI
        google_model = os.getenv("LLM_MODEL", "gemini-2.0-flash")
        return ChatGoogleGenerativeAI(model=google_model, temperature=0.3, google_api_key=google_key)

    # Fallback: Anthropic Claude
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    if anthropic_key:
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(model="claude-haiku-4-5", temperature=0.3, api_key=anthropic_key)

    raise EnvironmentError(
        "No LLM API key found. Set one of: OPENAI_API_KEY, GOOGLE_API_KEY, ANTHROPIC_API_KEY"
    )


llm = _build_llm()

# ── Agent State ───────────────────────────────────────────────────────────────
class AgentState(TypedDict):
    messages:       Annotated[list[BaseMessage], add_messages]
    intent:         str           # greeting | inquiry | high_intent
    stage:          str           # chat | qualifying | done
    lead_name:      str | None
    lead_email:     str | None
    lead_platform:  str | None


# ── System Prompt ─────────────────────────────────────────────────────────────
KB_SUMMARY = get_full_kb_summary()

SYSTEM_PROMPT = f"""You are Alex, a friendly and knowledgeable sales assistant for AutoStream — an AI-powered video editing SaaS for content creators.

KNOWLEDGE BASE:
{KB_SUMMARY}

YOUR RESPONSIBILITIES:
1. Greet users warmly and help with product/pricing questions.
2. Use ONLY the knowledge base above to answer product questions. Never invent pricing or features.
3. Detect user intent:
   - "greeting"     → casual hi/hello, no specific need
   - "inquiry"      → asking about features, pricing, plans, policies
   - "high_intent"  → user says they want to sign up, try, buy, or is clearly ready to convert
4. When intent is high_intent, collect: Name, Email, and Creator Platform (YouTube, Instagram, etc.) — one at a time, conversationally.
5. Do NOT ask for lead info unless intent is high_intent.
6. Be concise, warm, and helpful. No walls of text.

LEAD COLLECTION RULES:
- Ask for Name first, then Email, then Platform — one per message.
- Once all three are collected, confirm and say you'll get them set up.
- Do NOT fabricate lead info. Only use what the user tells you.

INTENT SIGNAL EXAMPLES (high_intent):
- "I want to sign up"
- "I'd like to try the Pro plan"
- "How do I get started?"
- "Let's do it" / "I'm ready"
- "Sign me up"
"""

# ── Intent Detection ──────────────────────────────────────────────────────────
INTENT_PROMPT = """Classify the user's latest message into exactly one intent label.

Labels:
- greeting     : casual hello, how are you, etc.
- inquiry      : asking about product, pricing, features, policies
- high_intent  : ready to sign up, buy, or try the product

Respond with ONLY the label word. Nothing else."""


def detect_intent(user_message: str) -> str:
    result = llm.invoke([
        SystemMessage(content=INTENT_PROMPT),
        HumanMessage(content=user_message),
    ])
    label = result.content.strip().lower()
    if label not in ("greeting", "inquiry", "high_intent"):
        label = "inquiry"
    return label


# ── Helper: extract simple field from message ─────────────────────────────────
def _extract_email(text: str) -> str | None:
    match = re.search(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+", text)
    return match.group(0) if match else None


KNOWN_PLATFORMS = {
    "youtube": "YouTube",
    "instagram": "Instagram",
    "tiktok": "TikTok",
    "facebook": "Facebook",
    "twitter": "Twitter",
    "x": "X",
    "linkedin": "LinkedIn",
    "twitch": "Twitch",
}


def _normalize_name(text: str) -> str:
    cleaned = text.strip().strip(".,!?")
    lowered = cleaned.lower()
    prefixes = ("my name is ", "i am ", "i'm ", "im ", "this is ")
    for prefix in prefixes:
        if lowered.startswith(prefix):
            cleaned = cleaned[len(prefix):].strip()
            break
    return cleaned


def _looks_like_name(text: str) -> bool:
    candidate = _normalize_name(text)
    lowered = candidate.lower()

    if not candidate or "@" in candidate:
        return False
    if len(candidate.split()) > 4:
        return False
    if any(ch.isdigit() for ch in candidate):
        return False

    # Reject obvious intent statements and command-like phrases.
    blocked_tokens = {
        "sign", "signup", "up", "ready", "start", "trial", "plan", "pricing",
        "buy", "purchase", "help", "please", "want", "interested", "pro",
        "basic", "yes", "no", "okay", "ok", "sure", "lets", "let's",
    }
    tokens = set(re.findall(r"[a-zA-Z']+", lowered))
    if tokens & blocked_tokens:
        return False

    # Allow letters, spaces, apostrophes, hyphens and dots.
    return bool(re.fullmatch(r"[A-Za-z][A-Za-z .'-]{1,48}", candidate))


def _extract_platform(text: str) -> str | None:
    lowered = text.lower()
    for key, canonical in KNOWN_PLATFORMS.items():
        if re.search(rf"\b{re.escape(key)}\b", lowered):
            return canonical

    # Accept a short custom platform label (e.g., "Snapchat"), but reject phrases.
    candidate = text.strip().strip(".,!?")
    if not candidate:
        return None
    if len(candidate.split()) > 3:
        return None
    if any(ch.isdigit() for ch in candidate):
        return None

    blocked_tokens = {
        "sign", "signup", "up", "ready", "start", "trial", "plan", "pricing",
        "buy", "purchase", "help", "please", "want", "interested", "email",
        "name", "my", "is", "i", "am", "im", "i'm", "me",
    }
    tokens = set(re.findall(r"[a-zA-Z']+", candidate.lower()))
    if tokens & blocked_tokens:
        return None

    if re.fullmatch(r"[A-Za-z][A-Za-z &+.-]{1,30}", candidate):
        return candidate.title()
    return None


# ── Graph Nodes ───────────────────────────────────────────────────────────────

def classify_node(state: AgentState) -> AgentState:
    """Detect intent from the latest human message."""
    last_human = next(
        (m for m in reversed(state["messages"]) if isinstance(m, HumanMessage)), None
    )
    if last_human:
        intent = detect_intent(last_human.content)
    else:
        intent = "greeting"

    # Escalate to qualifying if high_intent and not already done
    stage = state.get("stage", "chat")
    if intent == "high_intent" and stage == "chat":
        stage = "qualifying"

    return {**state, "intent": intent, "stage": stage}


def rag_chat_node(state: AgentState) -> AgentState:
    """Standard RAG-powered reply for greetings and inquiries."""
    last_human = next(
        (m for m in reversed(state["messages"]) if isinstance(m, HumanMessage)), None
    )
    query = last_human.content if last_human else ""

    # Retrieve relevant KB context
    context = retrieve_context(query)

    augmented_system = (
        SYSTEM_PROMPT
        + f"\n\nRELEVANT CONTEXT FOR THIS QUERY:\n{context}"
    )

    response = llm.invoke(
        [SystemMessage(content=augmented_system)] + state["messages"]
    )
    return {**state, "messages": [response]}


def qualify_node(state: AgentState) -> AgentState:
    """
    Collects lead info one field at a time.
    Checks what we have and asks for the next missing field.
    Once all three collected, calls mock_lead_capture and marks done.
    """
    last_human = next(
        (m for m in reversed(state["messages"]) if isinstance(m, HumanMessage)), None
    )
    user_text = last_human.content if last_human else ""

    name     = state.get("lead_name")
    email    = state.get("lead_email")
    platform = state.get("lead_platform")

    # ── Try to fill missing fields from the latest user message ──────────────
    if not name:
        if _looks_like_name(user_text):
            name = _normalize_name(user_text).title()

    if not email:
        extracted = _extract_email(user_text)
        if extracted:
            email = extracted

    if email and name and not platform:
        platform = _extract_platform(user_text)

    updated_state = {
        **state,
        "lead_name": name,
        "lead_email": email,
        "lead_platform": platform,
    }

    # ── All three collected → fire the tool ──────────────────────────────────
    if name and email and platform:
        result = mock_lead_capture(name, email, platform)
        reply = (
            f"🎉 You're all set, {name}! I've captured your details and our team will reach out to **{email}** shortly to get your AutoStream Pro account activated.\n\n"
            f"Welcome aboard! Your {platform} content is about to level up. 🚀"
        )
        updated_state["stage"] = "done"
        updated_state["messages"] = [AIMessage(content=reply)]
        return updated_state

    # ── Ask for the next missing field ───────────────────────────────────────
    if not name:
        reply = "Awesome, let's get you set up! 🎬 What's your name?"
    elif not email:
        reply = f"Great to meet you, {name}! What's your email address?"
    else:
        reply = f"Almost there! Which platform do you primarily create on? (e.g., YouTube, Instagram, TikTok)"

    updated_state["messages"] = [AIMessage(content=reply)]
    return updated_state


# ── Routing Logic ─────────────────────────────────────────────────────────────

def route(state: AgentState) -> str:
    stage = state.get("stage", "chat")
    if stage in ("qualifying", "done"):
        return "qualify"
    return "rag_chat"


# ── Build the Graph ───────────────────────────────────────────────────────────

def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("classify",  classify_node)
    graph.add_node("rag_chat",  rag_chat_node)
    graph.add_node("qualify",   qualify_node)

    graph.add_edge(START, "classify")
    graph.add_conditional_edges("classify", route, {"rag_chat": "rag_chat", "qualify": "qualify"})
    graph.add_edge("rag_chat", END)
    graph.add_edge("qualify",  END)

    return graph.compile()


agent = build_graph()
