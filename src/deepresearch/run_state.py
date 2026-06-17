"""Read persisted run summaries from the SQLite checkpointer."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from pydantic import BaseModel

from deepresearch.persistence import create_checkpointer


class RunSummary(BaseModel):
    slug: str
    question: str
    thread_id: str
    brief_approved: bool
    last_node: str | None
    pending_handoff: bool
    round: int
    auto_round: int
    has_brief: bool
    has_report: bool
    last_checkpoint_at: str | None


def load_runs(state_dir: Path, output_dir: Path | None = None) -> list[RunSummary]:
    """Return one summary per thread id in the checkpoint database."""
    db_path = state_dir / "checkpoints.sqlite"
    if not db_path.exists():
        return []
    with sqlite3.connect(db_path) as conn:
        try:
            rows = conn.execute(
                "select distinct thread_id from checkpoints order by thread_id"
            ).fetchall()
        except sqlite3.OperationalError:
            return []
    summaries = []
    for (thread_id,) in rows:
        summary = load_run(state_dir, output_dir or Path("research"), thread_id)
        if summary is not None:
            summaries.append(summary)
    return summaries


def load_run(state_dir: Path, output_dir: Path, slug: str) -> RunSummary | None:
    """Return a summary for ``slug`` or ``None`` if no thread exists."""
    with create_checkpointer(state_dir) as saver:
        checkpoints = list(saver.list({"configurable": {"thread_id": slug}}))
    if not checkpoints:
        return None
    latest = checkpoints[0]
    values = latest.checkpoint.get("channel_values", {})
    brief = values.get("brief")
    brief_approved = bool(
        getattr(brief, "approved", False) or (isinstance(brief, dict) and brief.get("approved"))
    )
    actual_slug = str(values.get("slug") or slug)
    question = str(values.get("question") or "")
    next_nodes = latest.checkpoint.get("next") or latest.checkpoint.get("channel_values", {}).get(
        "__next__"
    )
    last_node = None
    if isinstance(next_nodes, (list, tuple)) and next_nodes:
        last_node = str(next_nodes[0])
    run_dir = output_dir / actual_slug
    return RunSummary(
        slug=actual_slug,
        question=question,
        thread_id=slug,
        brief_approved=brief_approved,
        last_node=last_node,
        pending_handoff=bool(values.get("pending_handoff", False)),
        round=int(values.get("round", 0) or 0),
        auto_round=int(values.get("auto_round", 0) or 0),
        has_brief=(run_dir / "brief.md").exists(),
        has_report=(run_dir / "report.md").exists(),
        last_checkpoint_at=latest.checkpoint.get("ts"),
    )
