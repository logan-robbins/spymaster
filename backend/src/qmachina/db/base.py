"""SQLAlchemy 2.x declarative base for control-plane ORM models."""
from __future__ import annotations

from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    """Shared declarative base for all control-plane tables.

    All ORM models inherit from this class. Alembic autogenerate
    discovers tables via ``Base.metadata``.
    """
