"""Tests for the CLI graph wiring."""

import pytest

from deepresearch.cli import _build_configurable
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
