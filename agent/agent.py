"""Agent conversation loop: runs one question through tool-calling rounds."""

import json
import re

from agent.llm_client import chat
from agent.system_prompt import SYSTEM_PROMPT
from agent.tools import TOOLS, call_tool

def extract_references(text: str) -> tuple[str, dict | None, dict | None]:
    """Extract <chart>JSON</chart> and <table>JSON</table> blocks from agent response.
    Returns (cleaned_text, chart_dict_or_None, table_dict_or_None).
    """
    chart = None
    table = None

    chart_pattern = r'<chart>\s*(\{.*?\})\s*</chart>'
    chart_match = re.search(chart_pattern, text, re.DOTALL)
    if chart_match:
        chart = json.loads(chart_match.group(1))
        text = text[:chart_match.start()].strip()

    table_pattern = r'<table>\s*(\{.*?\})\s*</table>'
    table_match = re.search(table_pattern, text, re.DOTALL)
    if table_match:
        table = json.loads(table_match.group(1))
        text = re.sub(table_pattern, '', text, flags=re.DOTALL).strip()

    return text, chart, table


MAX_TURNS = 8     # max tool-call rounds before forcing a response
MAX_HISTORY = 20  # max prior messages to keep


def run_agent(
    question: str,
    history: list[dict],
    app_context: dict,
    api_key: str,
    provider: str = "anthropic",
    model: str | None = None,
) -> dict:
    """Run one question through the agent loop.

    Returns: {"response": str, "tools_called": list[str], "error": str | None}
    """
    context_prefix = _format_context(app_context)
    full_question = f"{context_prefix}\n\n{question}" if context_prefix else question

    messages = _trim_history(history) + [{"role": "user", "content": full_question}]
    tools_called: list[str] = []
    turns = 0

    try:
        while turns < MAX_TURNS:
            response = chat(
                messages=messages, tools=TOOLS, system=SYSTEM_PROMPT,
                api_key=api_key, provider=provider, model=model,
            )
            message = response.choices[0].message
            tool_calls = getattr(message, "tool_calls", None)

            # No tool calls -> the model has produced its final answer.
            if not tool_calls:
                raw_text = message.content or ""
                clean_text, chart, table = extract_references(raw_text)
                return {
                    "response": clean_text,
                    "tools_called": tools_called,
                    "chart": chart,
                    "table": table,
                    "error": None,
                }

            # Record the assistant turn (with its tool calls) ...
            messages.append({
                "role": "assistant",
                "content": message.content or "",
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.function.name,
                                     "arguments": tc.function.arguments},
                    }
                    for tc in tool_calls
                ],
            })

            # ... then run each tool and feed results back.
            for tc in tool_calls:
                name = tc.function.name
                try:
                    args = json.loads(tc.function.arguments or "{}")
                except json.JSONDecodeError:
                    args = {}
                tools_called.append(name)
                result = call_tool(name, args)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": json.dumps(result, default=str),
                })

            turns += 1

        return {
            "response": "I wasn't able to complete the analysis. Please try rephrasing.",
            "tools_called": tools_called,
            "chart": None,
            "table": None,
            "error": "max_turns_exceeded",
        }

    except Exception as exc:
        return {
            "response": f"Sorry, I hit an error while answering: {exc}",
            "tools_called": tools_called,
            "chart": None,
            "table": None,
            "error": str(exc),
        }


def _format_context(ctx: dict) -> str:
    """Convert app context into a natural-language prefix."""
    if not ctx:
        return ""
    parts = []
    if ctx.get("aoi"):
        parts.append(f"Study area: {ctx['aoi']}")
    if ctx.get("tab"):
        parts.append(f"Current tab: {ctx['tab']}")
    if ctx.get("sensor") and ctx.get("resolution"):
        parts.append(f"Currently viewing: {ctx['sensor']} {ctx['resolution']}m")
    if ctx.get("year"):
        parts.append(f"Selected year: {ctx['year']}")
    return "[App context: " + ", ".join(parts) + "]" if parts else ""


def _trim_history(history: list[dict]) -> list[dict]:
    """Keep only the last MAX_HISTORY messages to control token cost."""
    if not history:
        return []
    return history[-MAX_HISTORY:] if len(history) > MAX_HISTORY else list(history)
