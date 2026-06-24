"""Checkpointer factory."""

import logging
import re
from pathlib import Path

from langgraph.checkpoint.sqlite import SqliteSaver


def _suppress_model_deserialisation_warnings() -> None:
    """Silence the 'unregistered type deepresearch.models.*' msgpack warning.

    LangGraph warns about every custom type not in its built-in allowlist. All
    deepresearch.models types are safe — we own them.  Rather than passing a
    restrictive allowed_msgpack_modules list (which blocks other types and
    breaks checkpoint reads), we filter the warning at the logger level.
    """

    class _Filter(logging.Filter):
        _pat = re.compile(r"Deserializing unregistered type deepresearch\.")

        def filter(self, record: logging.LogRecord) -> bool:
            return not self._pat.search(record.getMessage())

    logging.getLogger("langgraph.checkpoint.serde.jsonplus").addFilter(_Filter())


_suppress_model_deserialisation_warnings()


def create_checkpointer(state_dir: Path):
    """Create a SQLite-backed LangGraph checkpointer context manager.

    The returned object must be entered with ``with`` (or used as a context
    manager) to obtain the actual ``SqliteSaver``. Callers depend only on this
    factory, so the concrete class can be swapped later.
    """
    db_path = state_dir / "checkpoints.sqlite"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return SqliteSaver.from_conn_string(str(db_path))
