"""Gather join node: prune stale subreports after fan-in."""

from deepresearch.state import ResearchState, prune_subreports


def gather(state: ResearchState) -> dict:
    """Return pruned subreports keyed by subtopic_slug."""
    if state.get("brief") is None:
        return {"subreports": state["subreports"]}
    pruned = prune_subreports(state["subreports"], state["brief"])
    return {"subreports": pruned}
