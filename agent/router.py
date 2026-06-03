"""/agent endpoints: chat + provider discovery."""

import os

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from agent.agent import run_agent
from agent.llm_client import SUPPORTED_PROVIDERS

router = APIRouter(prefix="/agent", tags=["agent"])

ENV_KEYS = {"anthropic": "ANTHROPIC_API_KEY", "openai": "OPENAI_API_KEY"}


class ChatRequest(BaseModel):
    question: str
    history: list[dict] = []       # previous messages this session
    app_context: dict = {}         # what the user is viewing in the Shiny app
    user_api_key: str | None = None
    provider: str = "anthropic"    # anthropic | openai


class ChatResponse(BaseModel):
    response: str
    tools_called: list[str]
    key_source: str                # "server" or "user"
    error: str | None = None


@router.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    if request.provider not in SUPPORTED_PROVIDERS:
        raise HTTPException(status_code=400, detail=f"Unsupported provider: {request.provider}")

    # Key priority: server env var first (SensingClues pays), then user-provided.
    server_key = os.environ.get(ENV_KEYS.get(request.provider, ""))
    if server_key:
        api_key, key_source = server_key, "server"
    elif request.user_api_key:
        api_key, key_source = request.user_api_key, "user"
    else:
        raise HTTPException(
            status_code=401,
            detail=f"No API key available. Set {ENV_KEYS[request.provider]} on the "
                   f"server or provide your own key in the request.",
        )

    result = run_agent(
        question=request.question,
        history=request.history,
        app_context=request.app_context,
        api_key=api_key,
        provider=request.provider,
    )

    return ChatResponse(
        response=result["response"],
        tools_called=result["tools_called"],
        key_source=key_source,
        error=result.get("error"),
    )


@router.get("/providers")
def get_providers():
    """Supported LLM providers and whether a server key is configured for each."""
    out = {}
    for provider, info in SUPPORTED_PROVIDERS.items():
        out[provider] = {
            "display_name": info["display_name"],
            "server_key_configured": bool(os.environ.get(ENV_KEYS[provider])),
            "default_model": info["default_model"],
        }
    return out
