"""Combined ASGI entrypoint: Part-1 data API + Part-2 agent.

Run from repo root:
    uvicorn serve:app --port 8000

This keeps api/main.py unmodified — it imports the existing Part-1 app and
adds the /agent router on top. /api/v1/* endpoints are unchanged.
"""

from agent.router import router as agent_router
from api.main import app

app.include_router(agent_router)
