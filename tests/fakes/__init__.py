"""Test doubles for external subsystems."""

from fakes.chat import FakeChat
from fakes.embeddings import FakeEmbeddings
from fakes.pdf import FakePdf
from fakes.tavily import FakeTavily

__all__ = ["FakeChat", "FakeEmbeddings", "FakeTavily", "FakePdf"]
