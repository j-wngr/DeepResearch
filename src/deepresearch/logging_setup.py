"""Structured logging setup for CLI entry points."""

from __future__ import annotations

import logging


class _StructuredFormatter(logging.Formatter):
    _standard = set(logging.LogRecord("", 0, "", 0, "", (), None).__dict__)

    def format(self, record: logging.LogRecord) -> str:
        base = super().format(record)
        extras = []
        for key, value in sorted(record.__dict__.items()):
            if key not in self._standard and not key.startswith("_"):
                extras.append(f"{key}={value}")
        if extras:
            return f"{base} {' '.join(extras)}"
        return base


def configure_logging(level: str = "INFO") -> None:
    """Configure root logging as one structured line per record."""
    handler = logging.StreamHandler()
    handler.setFormatter(
        _StructuredFormatter("%(asctime)s level=%(levelname)s logger=%(name)s msg=%(message)s")
    )
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
