"""FastAPI application entrypoint. Run: `uvicorn app.api.main:app --reload`."""
from __future__ import annotations

from fastapi import FastAPI
from fastapi.responses import RedirectResponse

from app.api.routers import conversations, reviews
from app.config import settings

app = FastAPI(title="AgentOps API", version="0.0.0")
app.include_router(conversations.router)
app.include_router(reviews.router)


@app.get("/", include_in_schema=False)
def root() -> RedirectResponse:
    return RedirectResponse(url="/docs")


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "mode": "live" if settings.has_credentials else "demo"}
