"""LiteLLM multi-provider abstraction.

Handles provider differences transparently. LiteLLM normalises every provider
to the OpenAI response shape, so callers read resp.choices[0].message.
"""

import litellm

SUPPORTED_PROVIDERS = {
    "anthropic": {
        "model_prefix": "anthropic/",
        "default_model": "claude-sonnet-4-20250514",
        "display_name": "Anthropic (Claude)",
    },
    "openai": {
        "model_prefix": "openai/",
        "default_model": "gpt-4o",
        "display_name": "OpenAI (GPT-4o)",
    },
}


def _to_openai_tools(tools: list[dict]) -> list[dict]:
    """Convert Anthropic-ish tool defs (name/description/input_schema) to the
    OpenAI function-tool format LiteLLM expects."""
    converted = []
    for t in tools:
        converted.append({
            "type": "function",
            "function": {
                "name": t["name"],
                "description": " ".join(t["description"].split()),
                "parameters": t.get("input_schema", {"type": "object", "properties": {}}),
            },
        })
    return converted


def chat(
    messages: list[dict],
    tools: list[dict],
    system: str,
    api_key: str,
    provider: str = "anthropic",
    model: str | None = None,
):
    """Call the LLM via LiteLLM. Returns the (OpenAI-normalised) response object.

    Tool use requires capable models — best with Claude Sonnet or GPT-4o.
    """
    if provider not in SUPPORTED_PROVIDERS:
        raise ValueError(f"Unsupported provider: {provider}")

    info = SUPPORTED_PROVIDERS[provider]
    model_name = model or info["default_model"]

    # LiteLLM has no top-level system kwarg — pass it as a leading system message.
    full_messages = [{"role": "system", "content": system}] + messages

    return litellm.completion(
        model=f"{info['model_prefix']}{model_name}",
        messages=full_messages,
        tools=_to_openai_tools(tools),
        api_key=api_key,
        temperature=0,
        max_tokens=2048,
    )
