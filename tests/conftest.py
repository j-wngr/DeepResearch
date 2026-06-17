"""Shared pytest fixtures."""

import os
import tempfile
from pathlib import Path

import pytest

from deepresearch.config import reset_config


@pytest.fixture
def tmp_workspace(monkeypatch):
    """Create temporary bibliography/output/state directories and wire env vars."""
    with tempfile.TemporaryDirectory() as tmp:
        bib_dir = Path(tmp) / "Bibliography"
        out_dir = Path(tmp) / "research"
        state_dir = Path(tmp) / ".deepresearch"
        bib_dir.mkdir()
        out_dir.mkdir()
        state_dir.mkdir()
        (bib_dir / "_sources").mkdir()
        (bib_dir / "_sources" / "pdfs").mkdir()
        (bib_dir / "_inbox").mkdir()

        monkeypatch.setenv("BIBLIOGRAPHY_DIR", str(bib_dir))
        monkeypatch.setenv("OUTPUT_DIR", str(out_dir))
        monkeypatch.setenv("STATE_DIR", str(state_dir))
        monkeypatch.setenv("TAVILY_API_KEY", "fake-test-key")
        # Clear any pre-loaded values so next Config() re-reads the env.
        reset_config()

        yield {
            "bibliography_dir": bib_dir,
            "output_dir": out_dir,
            "state_dir": state_dir,
        }

    reset_config()
    os.environ.setdefault("BIBLIOGRAPHY_DIR", "./Bibliography")
    os.environ.setdefault("OUTPUT_DIR", "./research")
    os.environ.setdefault("STATE_DIR", "./.deepresearch")
