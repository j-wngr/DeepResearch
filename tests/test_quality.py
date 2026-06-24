"""Tests for the web-source quality filter."""

import pytest

from deepresearch.sources.quality import is_acceptable, link_density, word_count

_GOOD_ARTICLE = """\
# Unsupervised Anomaly Detection for Electric Motors

Permanent magnet synchronous motors (PMSMs) are widely deployed in industrial
applications where early fault detection is critical for preventing unplanned
downtime. Traditional supervised approaches require labelled fault datasets that
are rarely available in practice. This survey reviews unsupervised and
semi-supervised methods applicable when only normal operating data is available.

## Feature Extraction

Phase current signals are rich in fault-related spectral content. Stator current
signatures exhibit characteristic frequency components under eccentricity, bearing
degradation, and demagnetisation faults. Time-frequency representations such as
the short-time Fourier transform (STFT) and wavelet decompositions capture
non-stationary behaviour during run-up and load transients.

## Deep Learning Approaches

Autoencoder-based models learn a compact representation of normal operation.
Anomalies produce elevated reconstruction errors that threshold-based detectors
can flag. Variational autoencoders and their convolutional variants handle
multivariate sensor streams with correlated features.

## Threshold Selection

Choosing an appropriate anomaly threshold requires careful calibration on
held-out normal data. Statistical approaches such as extreme value theory allow
threshold selection without requiring fault examples. Adaptive thresholds that
track seasonal or load-dependent drift in motor behaviour improve robustness in
continuously varying operating conditions.
"""


@pytest.mark.unit
def test_word_count_normal_article():
    assert word_count(_GOOD_ARTICLE) >= 150


@pytest.mark.unit
def test_word_count_strips_code_blocks():
    md = "Some prose here.\n\n```python\nfor i in range(100):\n    pass\n```\n\nMore prose."
    # Code block words should not be counted
    assert word_count(md) < 10


@pytest.mark.unit
def test_word_count_strips_inline_code():
    md = "The `create_model_dict` function loads models. It returns a dict."
    wc = word_count(md)
    # "create_model_dict" stripped; remaining words counted
    assert wc >= 8
    assert wc <= 12


@pytest.mark.unit
def test_word_count_strips_markdown_links():
    md = "See [this paper](https://example.com/paper) for details."
    wc = word_count(md)
    # Link anchor text "this paper" kept; URL stripped
    assert "this" in md  # sanity
    assert wc == 5  # "See", "this", "paper", "for", "details"


@pytest.mark.unit
def test_link_density_pure_prose():
    assert link_density(_GOOD_ARTICLE) < 0.05


@pytest.mark.unit
def test_link_density_link_directory():
    md = "\n".join(
        f"- [{i}. Some Link Title Here](https://example.com/{i})" for i in range(30)
    )
    assert link_density(md) > 0.8


@pytest.mark.unit
def test_link_density_empty():
    assert link_density("") == 0.0


@pytest.mark.unit
def test_is_acceptable_good_article():
    assert is_acceptable(_GOOD_ARTICLE, min_words=150, max_link_density=0.5)


@pytest.mark.unit
def test_is_acceptable_too_short():
    stub = "This page is under construction."
    assert not is_acceptable(stub, min_words=150, max_link_density=0.5)


@pytest.mark.unit
def test_is_acceptable_link_directory():
    directory = "\n".join(
        f"- [Article {i} about something interesting](https://example.com/{i})"
        for i in range(40)
    )
    assert not is_acceptable(directory, min_words=150, max_link_density=0.5)


@pytest.mark.unit
def test_is_acceptable_short_but_passes_custom_threshold():
    stub = "This page is under construction."
    assert is_acceptable(stub, min_words=5, max_link_density=0.5)


@pytest.mark.unit
def test_is_acceptable_empty():
    assert not is_acceptable("", min_words=150, max_link_density=0.5)
