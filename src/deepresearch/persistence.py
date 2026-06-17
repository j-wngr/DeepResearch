"""Checkpointer factory."""

from pathlib import Path

from langgraph.checkpoint.sqlite import SqliteSaver


def create_checkpointer(state_dir: Path):
    """Create a SQLite-backed LangGraph checkpointer context manager.

    The returned object must be entered with ``with`` (or used as a context
    manager) to obtain the actual ``SqliteSaver``. Callers depend only on this
    factory, so the concrete class can be swapped later.
    """
    db_path = state_dir / "checkpoints.sqlite"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return SqliteSaver.from_conn_string(str(db_path))
