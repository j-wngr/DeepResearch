"""Supervisor node: fan out subagents via Send."""

from langgraph.types import Send

from deepresearch.state import ResearchState


def supervisor(state: ResearchState) -> list[Send]:
    """Return Send targets for each in-scope sub-topic.

    In user_facing mode all sub-topics are re-run. In autonomous mode only
    sub-topics marked dirty are dispatched. Subtopics whose previous run
    isolated (hard failure) are never re-dispatched.
    """
    brief = state.get("brief")
    if brief is None:
        return []

    mode = state.get("mode", "user_facing")
    subreports = state.get("subreports", {})
    # Subtopics that previously hard-failed are tracked via the shortfall prefix
    # on their SubReport. Re-dispatching them will produce the same failure.
    isolated_slugs = {
        slug
        for slug, report in subreports.items()
        if report.shortfall and report.shortfall.startswith("subagent isolated:")
    }
    sends: list[Send] = []

    for subtopic in brief.subtopics:
        if subtopic.slug in isolated_slugs:
            continue
        if mode == "autonomous" and not subtopic.dirty:
            continue

        subagent_initial_state = {
            "slug": state["slug"],
            "subtopic": subtopic,
            "iteration": 0,
            "scratchpad": "",
            "whitelisted": [],
            "evidence": [],
            "draft": None,
            "candidates": [],
            "pending_acquisitions": [],
            "acquisition_gaps": [],
            "subreport": None,
            "subreports": {},
            "quality_gate_result": None,
            "isolated": False,
        }
        sends.append(Send("research_subagent", subagent_initial_state))

    return sends
