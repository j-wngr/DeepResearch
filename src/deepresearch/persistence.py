"""Checkpointer factory."""

import inspect
import sqlite3
from contextlib import closing, contextmanager
from pathlib import Path

from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.sqlite import SqliteSaver


def _model_allowlist() -> list[tuple[str, str]]:
    """Return (module, class_name) tuples for every class in deepresearch.models."""
    import deepresearch.models as _models
    return [
        (_models.__name__, name)
        for name, obj in inspect.getmembers(_models, inspect.isclass)
        if obj.__module__ == _models.__name__
    ]


@contextmanager
def create_checkpointer(state_dir: Path):
    """Create a SQLite-backed LangGraph checkpointer context manager.

    The returned object must be entered with ``with`` (or used as a context
    manager) to obtain the actual ``SqliteSaver``. Callers depend only on this
    factory, so the concrete class can be swapped later.
    """
    db_path = state_dir / "checkpoints.sqlite"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    serde = JsonPlusSerializer(allowed_msgpack_modules=_model_allowlist())
    with closing(
        sqlite3.connect(str(db_path), check_same_thread=False)
    ) as conn:
        yield SqliteSaver(conn, serde=serde)
