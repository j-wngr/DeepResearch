"""Tests for deepresearch.llm helpers."""

import json

import pytest

from deepresearch.llm import extract_json


@pytest.mark.parametrize(
    ("response", "expected"),
    [
        ('{"approved": true}', '{"approved": true}'),
        ('  \n{"approved": true}\n  ', '{"approved": true}'),
        ('```json\n{"approved": true}\n```', '{"approved": true}'),
        ('```\n{"approved": true}\n```', '{"approved": true}'),
        ('["a", "b"]', '["a", "b"]'),
        ('```json\n["a", "b"]\n```', '["a", "b"]'),
        ("just plain text", "just plain text"),
        ("", ""),
    ],
)
def test_extract_json_strips_fences(response, expected):
    assert extract_json(response) == expected


def test_extract_json_handles_none():
    assert extract_json(None) == ""


def test_extract_json_preserves_inner_whitespace():
    """Fenced multi-line JSON keeps its internal newlines."""
    inner = '[\n  {"slug": "s"}\n]'
    assert extract_json(f"```json\n{inner}\n```") == inner


def test_extract_json_fenced_output_parses():
    """The canonical real-world case: a fenced JSON array parses as a list."""
    fenced = '```json\n[{"slug": "x", "title": "X", "scope": "S"}]\n```'
    parsed = json.loads(extract_json(fenced))
    assert isinstance(parsed, list)
    assert parsed[0]["slug"] == "x"
