"""SQLAlchemy engine/session wiring. SQLite for dev, Postgres later via URL swap."""
from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from app.config import settings


class Base(DeclarativeBase):
    pass


def make_engine(url: str):
    connect_args: dict = {}
    if url.startswith("sqlite"):
        connect_args["check_same_thread"] = False
        # Ensure the parent directory for a file-backed SQLite DB exists.
        if url.startswith("sqlite:///"):
            db_path = Path(url[len("sqlite:///") :])
            if db_path.parent and str(db_path.parent) not in ("", "."):
                db_path.parent.mkdir(parents=True, exist_ok=True)
    return create_engine(url, connect_args=connect_args, future=True)


engine = make_engine(settings.database_url)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)


def get_session() -> Iterator[Session]:
    """FastAPI dependency: yields a session and always closes it."""
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()
