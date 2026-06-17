"""Graph construction."""

from langgraph.graph import END, START, StateGraph

from deepresearch.nodes.brief import approve_node, clarify_node, decompose_node
from deepresearch.nodes.gather import gather
from deepresearch.nodes.subagent import build_subagent_subgraph
from deepresearch.nodes.supervisor import supervisor
from deepresearch.nodes.writer import writer
from deepresearch.state import ResearchState


def _route_after_approve(state: ResearchState):
    """Route back to approve if not yet approved, otherwise fan out subagents."""
    brief = state.get("brief")
    if brief is None or not getattr(brief, "approved", False):
        return "approve"
    return supervisor(state)


def build_graph(checkpointer=None):
    """Build and compile the top-level LangGraph graph."""
    builder = StateGraph(ResearchState)

    builder.add_node("clarify", clarify_node)
    builder.add_node("decompose", decompose_node)
    builder.add_node("approve", approve_node)
    builder.add_node("research_subagent", build_subagent_subgraph())
    builder.add_node("gather", gather)
    builder.add_node("writer", writer)

    builder.add_edge(START, "clarify")
    builder.add_edge("clarify", "decompose")
    builder.add_edge("decompose", "approve")
    builder.add_conditional_edges(
        "approve",
        _route_after_approve,
        {"approve": "approve", "research_subagent": "research_subagent"},
    )
    builder.add_edge("research_subagent", "gather")
    builder.add_edge("gather", "writer")
    builder.add_edge("writer", END)

    return builder.compile(checkpointer=checkpointer)
