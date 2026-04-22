"""
utils/rag.py
Lightweight RAG (Retrieval-Augmented Generation) over the local knowledge base.
Uses simple keyword/TF-IDF-style chunking – no external vector DB required.
"""
import json
import os
import re
from pathlib import Path

KB_DIR = Path(__file__).parent.parent / "knowledge_base"


def _load_markdown_chunks() -> list[dict]:
    """Split the markdown KB into section chunks."""
    md_path = KB_DIR / "autostream_kb.md"
    text = md_path.read_text(encoding="utf-8")

    # Split on H2 / H3 headings
    sections = re.split(r"\n(?=#{1,3} )", text)
    chunks = []
    for section in sections:
        lines = section.strip().splitlines()
        if not lines:
            continue
        heading = lines[0].lstrip("#").strip()
        body = "\n".join(lines[1:]).strip()
        chunks.append({"heading": heading, "content": body})
    return chunks


def _load_json_kb() -> dict:
    """Load the structured JSON knowledge base."""
    json_path = KB_DIR / "autostream_kb.json"
    return json.loads(json_path.read_text(encoding="utf-8"))


def retrieve_context(query: str, top_k: int = 3) -> str:
    """
    Return the most relevant KB snippets for a given query.

    Strategy:
      1. Keyword overlap scoring against markdown chunks
      2. Always inject the full pricing JSON for pricing/plan queries
    """
    query_tokens = set(re.findall(r"\w+", query.lower()))
    chunks = _load_markdown_chunks()

    scored = []
    for chunk in chunks:
        text = (chunk["heading"] + " " + chunk["content"]).lower()
        tokens = set(re.findall(r"\w+", text))
        overlap = len(query_tokens & tokens)
        scored.append((overlap, chunk))

    scored.sort(key=lambda x: x[0], reverse=True)
    top_chunks = [c for _, c in scored[:top_k]]

    # Always include structured pricing for plan-related queries
    pricing_keywords = {"price", "pricing", "plan", "cost", "basic", "pro", "month", "pay"}
    if query_tokens & pricing_keywords:
        kb = _load_json_kb()
        pricing_summary = _format_pricing(kb)
        top_chunks.insert(0, {"heading": "Pricing (structured)", "content": pricing_summary})

    if not top_chunks:
        return "No relevant information found in the knowledge base."

    result = []
    for chunk in top_chunks:
        result.append(f"### {chunk['heading']}\n{chunk['content']}")

    return "\n\n---\n\n".join(result)


def _format_pricing(kb: dict) -> str:
    lines = []
    for plan in kb["plans"]:
        lines.append(
            f"**{plan['name']} Plan** – ${plan['price_monthly']}/month | "
            f"{plan['videos_per_month']} videos | {plan['resolution']} | "
            f"AI Captions: {'Yes' if plan['ai_captions'] else 'No'} | "
            f"Support: {plan['support']}"
        )
    lines.append(f"\nRefund policy: {kb['policies']['refund']}")
    lines.append(f"Free trial: {kb['policies']['free_trial']}")
    return "\n".join(lines)


def get_full_kb_summary() -> str:
    """Return a compact summary of the entire KB for the system prompt."""
    kb = _load_json_kb()
    return f"""
AUTOSTREAM KNOWLEDGE BASE SUMMARY
===================================
Company: {kb['company']['name']} — {kb['company']['description']}

PLANS:
{_format_pricing(kb)}

SUPPORTED PLATFORMS: {', '.join(kb['platforms_supported'])}

KEY FEATURES (All plans): {', '.join(kb['features']['all_plans'])}
Pro-only features: {', '.join(kb['features']['pro_only'])}
""".strip()
