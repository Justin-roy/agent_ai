"""
A minimal AI agent, built from scratch.

An "agent" is just three things glued together:
  1. An LLM (the brain)            -> served locally by Ollama
  2. Tools (the hands)             -> plain Python functions
  3. A loop (the orchestrator)     -> let the model call tools until it answers

This file talks to Ollama's REST API directly (no heavy framework) so you can
see exactly what's happening on the wire.

This is the "engine" module — it has no UI code in it. app.py (Streamlit)
imports from here to build the chat/PDF interface. You can still run this
file directly for a plain terminal chat.
"""

import os
import ast
import operator
import requests
from datetime import datetime

# ---- Config (overridable via environment variables, used by Docker later) ----
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/chat")
MODEL = os.getenv("MODEL", "qwen2.5:3b")

SYSTEM_PROMPT = (
    "You are a helpful assistant. Use tools when they help. "
    "If the conversation contains text extracted from an uploaded PDF, treat it as "
    "your source of truth for questions about that document, and say clearly if "
    "the answer isn't contained in it."
)


# --------------------------------------------------------------------------- #
# 1. TOOLS  -- ordinary Python functions the model is allowed to call.
# --------------------------------------------------------------------------- #

# A safe arithmetic evaluator (we deliberately do NOT use eval(), which is a
# security risk). We walk the parsed expression and only allow math operators.
_ALLOWED_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.USub: operator.neg,
}


def _safe_eval(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp):
        return _ALLOWED_OPS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp):
        return _ALLOWED_OPS[type(node.op)](_safe_eval(node.operand))
    raise ValueError("unsupported expression")


def calculator(expression: str) -> str:
    """Evaluate a basic arithmetic expression, e.g. '2 * (3 + 4)'."""
    try:
        return str(_safe_eval(ast.parse(expression, mode="eval").body))
    except Exception as e:
        return f"Error: {e}"


def get_time() -> str:
    """Return the current local date and time."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# The Python functions, keyed by the name the model will use.
TOOLS = {
    "calculator": calculator,
    "get_time": get_time,
}

# JSON-Schema descriptions of those tools, sent to the model so it knows
# what it can call and with which arguments.
TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Evaluate a basic arithmetic expression like '2 * (3 + 4)'.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "The arithmetic expression to evaluate.",
                    }
                },
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_time",
            "description": "Get the current local date and time.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]


# --------------------------------------------------------------------------- #
# 2. THE LLM CALL  -- one HTTP request to Ollama.
# --------------------------------------------------------------------------- #
def chat(messages):
    """Send the conversation to Ollama and return the assistant's message dict."""
    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL,
            "messages": messages,
            "tools": TOOL_SCHEMAS,
            "stream": False,
        },
        timeout=300,
    )
    response.raise_for_status()
    return response.json()["message"]


# --------------------------------------------------------------------------- #
# 3. THE AGENT LOOP  -- the part that makes it an "agent".
# --------------------------------------------------------------------------- #
def new_conversation():
    """Start a fresh message history with just the system prompt."""
    return [{"role": "system", "content": SYSTEM_PROMPT}]


def run_agent_step(messages, max_steps: int = 8):
    """
    Run the tool-calling loop on an EXISTING message list (the new user turn
    should already be appended before calling this). This keeps full
    conversation memory across turns, which a chat UI needs.

    Returns (final_answer_text, updated_messages).
    """
    for _ in range(max_steps):
        msg = chat(messages)
        messages.append(msg)

        tool_calls = msg.get("tool_calls")
        if not tool_calls:
            # No tool requested -> the model has its final answer.
            return msg.get("content", ""), messages

        # The model asked to call one or more tools. Run them and feed results back.
        for call in tool_calls:
            fn = call["function"]
            name = fn["name"]
            args = fn.get("arguments", {}) or {}
            tool = TOOLS.get(name)
            result = tool(**args) if tool else f"Unknown tool: {name}"
            print(f"  [tool] {name}({args}) -> {result}")
            messages.append({"role": "tool", "name": name, "content": str(result)})

    final = "Stopped: reached the maximum number of tool steps."
    messages.append({"role": "assistant", "content": final})
    return final, messages


def run_agent(user_input: str, max_steps: int = 8) -> str:
    """Single-shot convenience wrapper (no memory) — kept for backwards compatibility."""
    messages = new_conversation()
    messages.append({"role": "user", "content": user_input})
    answer, _ = run_agent_step(messages, max_steps=max_steps)
    return answer


# --------------------------------------------------------------------------- #
# 4. A tiny command-line chat loop so you can talk to it (now with memory).
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    print(f"Local agent ready (model: {MODEL}). Type 'quit' to exit.")
    messages = new_conversation()
    while True:
        try:
            user = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if user.lower() in {"quit", "exit"}:
            break
        if not user:
            continue
        messages.append({"role": "user", "content": user})
        answer, messages = run_agent_step(messages)
        print("Agent:", answer)
