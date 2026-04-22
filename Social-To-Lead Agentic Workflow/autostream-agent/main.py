"""
main.py
CLI runner for the AutoStream conversational agent.
Run:  python main.py
"""
import os
import sys

# Ensure project root is on the path
sys.path.insert(0, os.path.dirname(__file__))

if sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

from dotenv import load_dotenv

# Load local environment variables before graph import (LLM is initialized on import).
load_dotenv()

from langchain_core.messages import HumanMessage

from agent.graph import agent

BANNER = """
╔══════════════════════════════════════════════════════╗
║        AutoStream – AI Sales Assistant               ║
║        Powered by Inflx · ServiceHive                ║
╚══════════════════════════════════════════════════════╝
Type your message and press Enter.  Type 'quit' to exit.
"""

INITIAL_STATE = {
    "messages":      [],
    "intent":        "greeting",
    "stage":         "chat",
    "lead_name":     None,
    "lead_email":    None,
    "lead_platform": None,
}


def run():
    print(BANNER)
    state = INITIAL_STATE.copy()

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye! 👋")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "bye"):
            print("Agent: Thanks for chatting! Have a great day 🎬")
            break

        # Append the new human message and run the graph
        state["messages"] = state.get("messages", []) + [HumanMessage(content=user_input)]
        state = agent.invoke(state)

        # Print the latest AI response
        last_ai = next(
            (m for m in reversed(state["messages"]) if hasattr(m, "content") and m.type == "ai"),
            None,
        )
        if last_ai:
            print(f"\nAlex: {last_ai.content}\n")

        # Stop accepting input once lead is captured
        if state.get("stage") == "done":
            print("(Session complete — lead successfully captured!)")
            break


if __name__ == "__main__":
    run()
