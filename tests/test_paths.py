"""Tests for slug derivation and hashing utilities."""

import re

from deepresearch.paths import hash_bytes, hash_url, slug


def test_slug_deterministic():
    q = "Long-term effects of intermittent fasting on cardiovascular health"
    assert slug(q) == slug(q)


def test_slug_different_questions():
    s1 = slug("What is the effect of X on Y?")
    s2 = slug("What is the effect of X on employment?")
    assert s1 != s2


def test_slug_rewording_stable_base():
    # Punctuation and case variants yield the same slugified base.
    s1 = slug("What is AI?")
    s2 = slug("what is ai")
    assert s1.startswith("what-is-ai-")
    assert s2.startswith("what-is-ai-")


def test_slug_special_characters():
    s = slug("What's the best way to learn Python?")
    assert "'" not in s
    assert "?" not in s
    assert s.startswith("what-s-the-best-way-to-learn-python-")


def test_slug_truncation():
    long_q = "A " * 100 + "question"
    s = slug(long_q)
    base = s.rsplit("-", 1)[0]
    assert len(base) <= 60


def test_slug_format():
    s = slug("Test Question 123!")
    assert re.match(r"^[a-z0-9]+(-[a-z0-9]+)*-[a-f0-9]{8}$", s)


def test_hash_url_deterministic():
    url = "https://example.com/article"
    assert hash_url(url) == hash_url(url)
    assert hash_url(url) != hash_url(url + "/other")


def test_hash_bytes_deterministic():
    data = b"some pdf bytes"
    assert hash_bytes(data) == hash_bytes(data)
    assert hash_bytes(data) != hash_bytes(b"other bytes")
