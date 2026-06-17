# Implementation Log

Living record of what was actually built and the decisions made during implementation.

---

## Phase 0 — Skeleton, contracts, and test harness

**Completed:** 2026-06-16

### What was implemented

- `uv` project scaffolding with `pyproject.toml`, dependencies, `[project.scripts]` entry point, ruff/pytest config, and hatchling build system.
- `src/deepresearch/` package layout with stubs for future phases (`graph.py`, `gate.py`, `citations.py`, `verify.py`, `nodes/`, `rag/`, `sources/`).
- `config.py`: `Config` singleton with lazy init and `reset_config()` for tests. Layered precedence: `.env` (via `load_dotenv(override=True)`) > env vars > defaults.
- `paths.py`: `slug()` using `slugify + 8-char sha256`, plus `hash_url()`, `hash_bytes()`, and path helpers.
- `models.py`: all Pydantic models from Architecture §4.3, plus `SearchHit` and `Blocked`. `SourceRef` validator rejects ids inconsistent with `url`/`type`.
- `state.py`: `ResearchState`, `SubAgentState`, `merge_subreports` reducer (latest-wins upsert), `prune_subreports` helper for the gather node, and `add` reducer for history.
- `persistence.py`: `create_checkpointer(state_dir)` factory returning `langgraph.checkpoint.sqlite.SqliteSaver`.
- `llm.py`: `chat(role, messages)` mapping roles to configured models via `langchain-ollama`.
- `cli.py`: Typer app with `run`, `resume`, `sync`, `list`, `status` wired to no-op handlers.
- `tests/fakes/`: `FakeChat`, `FakeEmbeddings`, `FakeTavily`, `FakePdf`.
- `tests/conftest.py`: `tmp_workspace` fixture wiring temp directories through env vars.
- Unit tests for slug, models, reducers, and config layering.

### Verification

- `uv run deepresearch --help` lists all 5 commands.
- `uv run pytest` — 38 passed.
- `uv run ruff check src/ tests/` — clean.
- `uv run ruff format --check src/ tests/` — clean.

### Decisions and deviations

- **Build system**: Added `hatchling` build backend so `uv` packages the project and installs the `deepresearch` console entry point. This was not explicit in the original plan but is required for `uv run deepresearch` to work.
- **Config layering**: `Config` accepts an optional `env_file` argument in `__init__` so the `.env` precedence test can point to a temporary file. Without this, `load_dotenv` searches the current working directory and cannot be exercised in a hermetic test.
- **Reducer pruning**: The `merge_subreports` reducer itself does pure latest-wins upsert. A separate `prune_subreports(subreports, brief)` helper is provided for the `gather` node to call before returning. This satisfies the intent of pruning against the brief without requiring the reducer to access cross-channel state, which LangGraph's TypedDict reducers cannot do directly.
- **Fake implementations**: Fakes are intentionally minimal for Phase 0; deterministic vector generation in `FakeEmbeddings` will be fleshed out when RAG retrieval is tested in Phase 1.
- **Source id hash**: full SHA-256 hex digest is used for source ids (matching the slug suffix algorithm but not truncated).

### Open / follow-up

- Numeric defaults are taken from the existing `.env.example` and may be tuned against the real Ollama host in later phases.
- `references.json` schema is deferred to Phase 6.
- Relevance cache persistence format is deferred to Phase 3.
- `research list/status` ergonomics are deferred to later phases.
