"""Brief generation node with HITL clarify/decompose/approve interrupts."""

import json
import logging
from pathlib import Path

import yaml
from langchain_core.runnables import RunnableConfig
from langgraph.types import interrupt

from deepresearch.config import get_config
from deepresearch.llm import extract_json
from deepresearch.models import Brief, SubTopic
from deepresearch.paths import output_path
from deepresearch.paths import slug as make_slug
from deepresearch.state import ResearchState

logger = logging.getLogger("deepresearch.nodes.brief")


def _parse_json_response(response: str) -> dict | list:
    """Parse a JSON response, returning a safe default on failure."""
    cleaned = extract_json(response)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        logger.warning("Failed to parse JSON response: %s", response)
        return {} if cleaned.lstrip().startswith("{") else []


def _build_brief_markdown(brief: Brief) -> str:
    """Render a brief as YAML-frontmatter markdown."""
    frontmatter = {
        "question": brief.question,
        "slug": brief.slug,
        "thread_id": brief.thread_id,
    }
    yaml_text = yaml.safe_dump(frontmatter, sort_keys=False, allow_unicode=True)

    lines = [
        "---",
        yaml_text.rstrip(),
        "---",
        "",
        f"# Research Brief: {brief.question}",
        "",
        f"**Slug:** {brief.slug}",
        f"**Thread ID:** {brief.thread_id}",
        "",
        f"## Sub-topics ({len(brief.subtopics)})",
        "",
    ]
    for st in brief.subtopics:
        lines.append(f"### {st.title} (`{st.slug}`)")
        lines.append("")
        lines.append(f"**Scope:** {st.scope}")
        lines.append("")
        if st.guiding_questions:
            lines.append("**Guiding questions:**")
            for q in st.guiding_questions:
                lines.append(f"- {q}")
            lines.append("")
        if st.seed_queries:
            lines.append("**Seed queries:**")
            for q in st.seed_queries:
                lines.append(f"- {q}")
            lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def _ensure_subtopic_slugs(subtopics: list[dict]) -> list[SubTopic]:
    """Fill missing slugs and construct SubTopic models."""
    result: list[SubTopic] = []
    for raw in subtopics:
        data = dict(raw)
        if not data.get("slug"):
            data["slug"] = make_slug(data.get("title", "subtopic"))
        result.append(SubTopic.model_validate(data))
    return result


def clarify_node(state: ResearchState, config: RunnableConfig) -> dict:
    """Clarify the research question. May interrupt if clarification needed."""
    configurable = config.get("configurable", {})
    chat_fn = configurable["chat_fn"]

    clarify_prompt = (
        f"Given this research question: '{state['question']}', does it need clarification? "
        "Consider scope, audience, depth, time range, explicit exclusions. "
        'Return JSON: {"needs_clarification": bool, "questions": [str, ...]}'
    )
    clarify_response = chat_fn("clarify", [{"role": "user", "content": clarify_prompt}])
    clarify_data = _parse_json_response(clarify_response)

    if isinstance(clarify_data, dict) and clarify_data.get("needs_clarification"):
        questions_list = clarify_data.get("questions", [])
        if not isinstance(questions_list, list):
            questions_list = []
        user_answer = interrupt(
            {"type": "clarify", "questions": questions_list, "question": state["question"]}
        )
        refine_prompt = (
            f"Original question: '{state['question']}'. "
            f"User clarification: '{user_answer}'. "
            "Produce a refined, specific research question. Return only the refined question text."
        )
        refined_question = chat_fn("clarify", [{"role": "user", "content": refine_prompt}])
        return {"question": refined_question}

    return {}


def decompose_node(state: ResearchState, config: RunnableConfig) -> dict:
    """Decompose the research question into sub-topics."""
    cfg = get_config()
    configurable = config.get("configurable", {})
    chat_fn = configurable["chat_fn"]

    decompose_prompt = (
        f"Decompose this research question into sub-topics: '{state['question']}'. "
        "Target 3-7 sub-topics. Each sub-topic must have: slug (URL-safe), title, "
        "scope (what it covers and why), guiding_questions (concrete questions to answer), "
        "seed_queries (optional initial search queries). Return a JSON array of sub-topic objects."
    )
    decompose_response = chat_fn("synth", [{"role": "user", "content": decompose_prompt}])
    raw_subtopics = _parse_json_response(decompose_response)

    if isinstance(raw_subtopics, list):
        subtopics_list = raw_subtopics
    else:
        subtopics_list = []

    ceiling = cfg.subtopics_ceiling
    if len(subtopics_list) > ceiling:
        subtopics_list = subtopics_list[:ceiling]

    subtopics = _ensure_subtopic_slugs(subtopics_list)

    brief = Brief(
        question=state["question"],
        slug=state["slug"],
        thread_id=state["slug"],
        subtopics=subtopics,
    )

    return {"brief": brief}


def approve_node(state: ResearchState, config: RunnableConfig) -> dict:
    """Approve the brief via HITL interrupt. May loop for revisions."""
    configurable = config.get("configurable", {})
    chat_fn = configurable["chat_fn"]
    output_dir = Path(configurable["output_dir"])

    brief = state["brief"]
    if brief is None:
        return {}

    feedback = interrupt({"type": "approve", "brief": brief.model_dump()})

    approve_prompt = (
        f"User feedback on the research brief: '{feedback}'. "
        'Is this an approval? Return JSON: {"approved": bool}'
    )
    approve_response = chat_fn("clarify", [{"role": "user", "content": approve_prompt}])
    approve_data = _parse_json_response(approve_response)

    if not (isinstance(approve_data, dict) and approve_data.get("approved")):
        # Regenerate sub-topics based on feedback; return updated brief so the
        # state is checkpointed before the next interrupt (LangGraph 1.2.x
        # re-executes the node from the beginning on resume).
        regenerate_prompt = (
            f"Current brief: {brief.model_dump_json()}. "
            f"User feedback: '{feedback}'. Revise the sub-topics accordingly. "
            "Return a JSON array of sub-topic objects (same format as before)."
        )
        regenerate_response = chat_fn("synth", [{"role": "user", "content": regenerate_prompt}])
        raw_subtopics = _parse_json_response(regenerate_response)
        if isinstance(raw_subtopics, list):
            subtopics_list = raw_subtopics
        else:
            subtopics_list = []
        ceiling = get_config().subtopics_ceiling
        if len(subtopics_list) > ceiling:
            subtopics_list = subtopics_list[:ceiling]
        subtopics = _ensure_subtopic_slugs(subtopics_list)
        brief = brief.model_copy(update={"subtopics": subtopics})
        return {"brief": brief}

    # Approved: mark and persist
    brief = brief.model_copy(update={"approved": True})
    brief_path = output_path(output_dir, state["slug"], "brief.md")
    brief_path.parent.mkdir(parents=True, exist_ok=True)
    brief_path.write_text(_build_brief_markdown(brief), encoding="utf-8")

    return {"brief": brief}
