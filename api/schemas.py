"""Pydantic response models for the data API."""

from typing import Any

from pydantic import BaseModel


class APIResponse(BaseModel):
    """Consistent envelope for all data endpoints (ndvi/*, burned-area/*).

    health, available-data, and geometry/* return their own bespoke shapes.
    """

    data: list[dict[str, Any]]
    metadata: dict[str, Any]
    status: str = "ok"
    error: str | None = None
