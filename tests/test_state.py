"""Tests for graph state and channel reducers."""

from deepresearch.models import Brief, RoundRecord, SubReport, SubTopic
from deepresearch.state import add, merge_subreports, prune_subreports


def test_merge_subreports_upsert_latest_wins():
    old = SubReport(subtopic_slug="s1", body="old", citations=[])
    new = SubReport(subtopic_slug="s1", body="new", citations=[])
    result = merge_subreports({"s1": old}, {"s1": new})
    assert result["s1"].body == "new"
    assert len(result) == 1


def test_merge_subreports_no_duplicates():
    r = SubReport(subtopic_slug="s1", body="b1", citations=[])
    result = merge_subreports({"s1": r}, {"s1": r})
    assert len(result) == 1
    assert result["s1"].body == "b1"


def test_merge_subreports_disjoint():
    r1 = SubReport(subtopic_slug="s1", body="b1", citations=[])
    r2 = SubReport(subtopic_slug="s2", body="b2", citations=[])
    result = merge_subreports({"s1": r1}, {"s2": r2})
    assert result == {"s1": r1, "s2": r2}


def test_prune_subreports_removes_dropped():
    brief = Brief(
        question="Q?",
        slug="q-12345678",
        thread_id="q-12345678",
        subtopics=[SubTopic(slug="s1", title="T1", scope="S1")],
    )
    r1 = SubReport(subtopic_slug="s1", body="b1", citations=[])
    r2 = SubReport(subtopic_slug="s2", body="b2", citations=[])
    result = prune_subreports({"s1": r1, "s2": r2}, brief)
    assert "s1" in result
    assert "s2" not in result
    assert len(result) == 1


def test_prune_subreports_keeps_live():
    brief = Brief(
        question="Q?",
        slug="q-12345678",
        thread_id="q-12345678",
        subtopics=[SubTopic(slug="s1", title="T1", scope="S1")],
    )
    r1 = SubReport(subtopic_slug="s1", body="b1", citations=[])
    result = prune_subreports({"s1": r1}, brief)
    assert result == {"s1": r1}


def test_add_history():
    a = RoundRecord(round=1, mode="autonomous", coverage_score=0.5, fully_covered=False)
    b = RoundRecord(round=2, mode="user_facing", coverage_score=0.9, fully_covered=True)
    assert add([a], [b]) == [a, b]
