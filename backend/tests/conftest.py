"""Test fixtures: isolated in-memory SQLite, seeded demo data, FastAPI client."""
from __future__ import annotations

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

import app.models  # noqa: F401  (register tables)
from app.db import Base, get_session
from app.seed import seed_demo


@pytest.fixture()
def engine():
    eng = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
        future=True,
    )
    Base.metadata.create_all(eng)
    return eng


@pytest.fixture()
def session_factory(engine):
    return sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)


@pytest.fixture()
def db_session(session_factory):
    session = session_factory()
    try:
        yield session
    finally:
        session.close()


@pytest.fixture()
def seeded_customer_id(db_session) -> str:
    return seed_demo(db_session)


@pytest.fixture()
def client(session_factory):
    from fastapi.testclient import TestClient

    from app.api.main import app

    def _override_session():
        session = session_factory()
        try:
            yield session
        finally:
            session.close()

    from app.agent import AgentRunner, DemoLLMClient
    from app.api.routers.conversations import get_runner
    from app.tools import StubDocSearch, build_registry

    app.dependency_overrides[get_session] = _override_session
    # Force the offline demo model so ambient API keys can't make tests hit a live API.
    app.dependency_overrides[get_runner] = lambda: AgentRunner(
        DemoLLMClient(), build_registry(), StubDocSearch(), max_steps=8
    )
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()
