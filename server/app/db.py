from __future__ import annotations

from contextlib import contextmanager

from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from .config import Settings


class Base(DeclarativeBase):
    pass


def build_engine(settings: Settings):
    return create_engine(settings.database_url, future=True, pool_pre_ping=True)


def build_session_factory(settings: Settings):
    engine = build_engine(settings)
    return engine, sessionmaker(bind=engine, autoflush=False, autocommit=False, expire_on_commit=False)


def init_db(engine) -> None:
    from . import models  # noqa: F401

    Base.metadata.create_all(engine)
    _upgrade_message_schema(engine)


def _upgrade_message_schema(engine) -> None:
    inspector = inspect(engine)
    try:
        columns = {column["name"] for column in inspector.get_columns("messages")}
    except Exception:
        return
    if "metadata_json" in columns:
        return
    with engine.begin() as connection:
        connection.execute(text("ALTER TABLE messages ADD COLUMN metadata_json TEXT"))


@contextmanager
def session_scope(session_factory) -> Session:
    session = session_factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
