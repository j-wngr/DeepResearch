"""Scripted per-role chat responses for testing."""

import threading


class FakeChat:
    """Return canned responses per role."""

    def __init__(self, responses: dict[str, list[str]] | None = None) -> None:
        self._responses: dict[str, list[str]] = responses or {}
        self._call_counts: dict[str, int] = {}
        self._lock = threading.Lock()

    def chat(self, role: str, messages: list) -> str:
        """Return the next scripted response for `role`."""
        del messages
        with self._lock:
            self._call_counts[role] = self._call_counts.get(role, 0) + 1
            responses = self._responses.get(role, [])
            idx = self._call_counts[role] - 1
            if idx < len(responses):
                return responses[idx]
            return ""

    def call_count(self, role: str) -> int:
        """Return how many times `chat` has been called for `role`."""
        with self._lock:
            return self._call_counts.get(role, 0)
