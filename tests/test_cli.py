"""Tests for the CLI graph wiring."""

import pytest
from langgraph.types import Interrupt

from deepresearch.cli import _build_configurable, _build_resume_value, _collect_resume_input
from deepresearch.config import get_config


@pytest.mark.unit
def test_build_configurable_injects_network_clients(tmp_workspace):
    """The subagent gates web/PDF acquisition on these clients being present;
    without them a real run is RAG-only and produces empty, uncited reports."""
    cfg = get_config()
    configurable = _build_configurable(cfg)

    # The acquisition seams the subagent reads out of config.
    assert configurable.get("tavily_client") is not None
    assert configurable.get("pdf_client") is not None
    # pdf_client must expose the fetch/convert surface the subagent calls.
    assert hasattr(configurable["pdf_client"], "fetch")
    assert hasattr(configurable["pdf_client"], "convert")

    # Existing dependencies are still wired.
    for key in ("chat_fn", "embeddings", "store", "bibliography_dir", "output_dir"):
        assert key in configurable


@pytest.mark.unit
def test_build_resume_value_uses_interrupt_id_map_for_single_acquire(monkeypatch, tmp_path):
    """LangGraph requires map-form resumes if hidden interrupts are also pending."""
    save_path = tmp_path / "manual.pdf"
    request = {
        "source_id": "source-1",
        "title": "Manual source",
        "url": "https://example.com/manual.pdf",
        "save_path": str(save_path),
    }
    interrupt = Interrupt(value={"type": "acquire", "requests": [request]}, id="interrupt-1")
    monkeypatch.setattr("typer.prompt", lambda *args, **kwargs: "")

    resume = _build_resume_value([interrupt])

    assert resume == {
        "interrupt-1": [
            {
                "source_id": "source-1",
                "kind": "unobtainable",
            }
        ]
    }


@pytest.mark.unit
def test_build_resume_value_uses_interrupt_id_map_for_multiple_acquires(monkeypatch, tmp_path):
    first_path = tmp_path / "first.pdf"
    first_path.write_bytes(b"%PDF-1.4\n")
    second_path = tmp_path / "second.pdf"
    interrupts = [
        Interrupt(
            value={
                "type": "acquire",
                "requests": [
                    {
                        "source_id": "source-1",
                        "title": "First",
                        "url": "https://example.com/first.pdf",
                        "save_path": str(first_path),
                    }
                ],
            },
            id="interrupt-1",
        ),
        Interrupt(
            value={
                "type": "acquire",
                "requests": [
                    {
                        "source_id": "source-2",
                        "title": "Second",
                        "url": "https://example.com/second.pdf",
                        "save_path": str(second_path),
                    }
                ],
            },
            id="interrupt-2",
        ),
    ]
    monkeypatch.setattr("typer.prompt", lambda *args, **kwargs: "")

    resume = _build_resume_value(interrupts)

    assert resume == {
        "interrupt-1": [
            {
                "source_id": "source-1",
                "kind": "saved",
                "save_path": str(first_path),
            }
        ],
        "interrupt-2": [
            {
                "source_id": "source-2",
                "kind": "unobtainable",
            }
        ],
    }


@pytest.mark.unit
def test_collect_resume_input_preserves_interrupt_map(monkeypatch, tmp_path):
    save_path = tmp_path / "manual.pdf"
    state = type(
        "State",
        (),
        {
            "interrupts": [
                Interrupt(
                    value={
                        "type": "acquire",
                        "requests": [
                            {
                                "source_id": "source-1",
                                "title": "Manual source",
                                "url": "https://example.com/manual.pdf",
                                "save_path": str(save_path),
                            }
                        ],
                    },
                    id="interrupt-1",
                )
            ]
        },
    )()
    monkeypatch.setattr("typer.prompt", lambda *args, **kwargs: "")

    resume = _collect_resume_input(state)

    assert resume == {"interrupt-1": [{"source_id": "source-1", "kind": "unobtainable"}]}


@pytest.mark.unit
def test_collect_resume_input_empty_interrupts_returns_empty_map():
    """When get_state() surfaces no interrupts, return {} so LangGraph sets
    resume_is_map=True and bypasses the pending-interrupt count check."""
    state = type("State", (), {"interrupts": []})()

    resume = _collect_resume_input(state)

    assert resume == {}
    # {} satisfies LangGraph's is_xxh3 map check vacuously: resume_is_map=True
    # means the pending-interrupt count check is skipped entirely.
