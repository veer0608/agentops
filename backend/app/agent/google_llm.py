"""Gemini via Google's OpenAI-compatible endpoint.

Google exposes an OpenAI-compatible surface for the Gemini API, so we reuse the
tested OpenAI adapter (message + tool-call translation) and just point it at
Google's endpoint. Still the Google AI API — Google's models, a Google API key.
"""
from __future__ import annotations

from app.agent.openai_llm import OpenAILLMClient

GEMINI_OPENAI_BASE = "https://generativelanguage.googleapis.com/v1beta/openai/"


class GoogleLLMClient(OpenAILLMClient):
    def __init__(self, api_key: str, model: str) -> None:
        super().__init__(api_key=api_key, model=model, base_url=GEMINI_OPENAI_BASE)
