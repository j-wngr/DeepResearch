# Implementation Plan

A step-by-step, test-gated bringup of the system described in [Architecture.md](Architecture.md). It expands the build order ([Architecture §13](Architecture.md#13-build-order-milestones)) into phases, each with a goal, the modules it delivers, concrete bringup steps, **exit criteria**, and the tests (unit + integration) that gate it. Every phase leaves the system runnable and verifiable; nothing is "done" until its tests are green.

Terminology is from [CONTEXT.md](../CONTEXT.md); contracts and models are from [Architecture §4–§6](Architecture.md#4-state--data-models).

## Principles

- **Bottom-up, seam-first.** Build leaf subsystems (RAG, sources, gate) before the nodes that orchestrate them, so each is testable in isolation behind its contract.
- **Fakes from day one.** The hermetic-test rule ([CLAUDE.md → Implementation rules](../CLAUDE.md)) requires that Ollama, Tavily, and marker are never hit in tests. The fake doubles are built in Phase 0 and used everywhere.
- **Invariant-driven tests.** Each phase explicitly tests the invariants it introduces (see the per-phase "Tests" and the [invariant matrix](#invariant-test-matrix)).
- **Vertical slice early.** By Phase 6 a question runs end-to-end (single pass); Phases 7–9 add the loop, acquisition UX, and hardening.

## Phase map & status

| Phase | Focus | Arch milestone | Status |
| --- | --- | --- | --- |
| 0 | Skeleton, config, models, state, **test harness + fakes** | 1 | shipped |
| 1 | RAG core (embeddings, store, index, retrieve) | 2 | shipped |
| 2 | Sources (pool, web, pdf, inbox) | 3 | shipped (acquisition UX in Phase 8) |
| 3 | Relevance gate + verdict cache | (part of 4) | shipped |
| 4 | Subagent subgraph (single sub-topic, no fan-out) | 4 | shipped |
| 5 | Orchestration (brief, supervisor fan-out, gather, checkpointer) | 5 | shipped |
| 6 | Writer + verification → first full single-pass run | 6 | shipped |
| 7 | Refinement loop (evaluate, two tiers, two-mode re-run) | 7 | shipped |
| 8 | Acquisition UX (blocked-fetch interrupt, inbox round-trip) | 8 | shipped |
| 9 | Hardening + **full end-to-end integration suite** | 9 | shipped |

Status values:

- **shipped** — modules are in the tree, unit + integration tests are green, and the phase's exit criteria pass. The pre-commit gate (`uv run pytest` + `uv run ruff check .` + `uv run ruff format --check .`) enforces this.
- **planned** — fully specified in this document and in [Architecture.md](Architecture.md), not yet in the tree.

This table is the source of truth for what has landed. It is updated as part of the same change that ships a phase. After a phase lands, also update the "Definition of done" entry that follows once its tests are first green in CI.

Dependencies are mostly linear; Phases 1 and 2 can proceed in parallel after Phase 0.

---

## Phase 0 — Skeleton, contracts, and test harness

**Goal.** An importable, configured package with all data contracts and the fake doubles, so every later phase is test-first.

**Deliverables.**

- `uv` project: `pyproject.toml` with deps ([Architecture §11](Architecture.md#11-external-dependencies)), `ruff` + `pytest` config, `src/deepresearch/` layout ([Architecture §5](Architecture.md#5-module-layout-srcdeepresearch)) with stubs.
- `config.py` — layered settings (env > file > defaults), all knobs/paths from [Architecture §8](Architecture.md#8-configuration--secrets-configpy); `tavily_api_key` from env only.
- `paths.py` — `slug(question) -> str` (slugified title + short stable hash), output/bibliography/state path resolution.
- `models.py` — every Pydantic model in [Architecture §4.3](Architecture.md#43-core-pydantic-models-modelspy).
- `state.py` — `ResearchState`, `SubAgentState`, and the `merge_subreports` reducer (upsert by `subtopic_slug`, prune dropped) + `add` for history.
- `persistence.py` — checkpointer factory (SQLite; swappable).
- `llm.py` — `chat(role, messages)` role→model resolution (real client; faked in tests).
- `cli.py` — Typer app with `run`/`resume`/`sync`/`list`/`status` wired to no-op handlers.
- **`tests/fakes/`** — `FakeChat` (scripted per-role responses), `FakeEmbeddings` (deterministic vector from text hash), `FakeTavily` (canned hits + extract markdown), `FakePdf` (canned bytes + marker output, plus a "blocked" URL set), and a `tmp_workspace` fixture wiring temp `bibliography_dir`/`output_dir`/`state_dir`.

**Bringup.** `uv sync`; `uv run deepresearch --help` lists the verbs; `uv run pytest` collects.

**Exit criteria.** Package imports; CLI help renders; all models round-trip (de)serialize; reducer + slug unit tests green.

**Tests (unit).**

- `slug()` is deterministic and stable under rewording variants; collision-resistant suffix.
- Model (de)serialization round-trips; `SourceRef` rejects an `id` inconsistent with its `url`/`type`.
- `merge_subreports`: upsert-by-slug (latest wins), prune of dropped slugs, no duplicates.
- `config` layering precedence; `tavily_api_key` read only from env.

---

## Phase 1 — RAG core

**Goal.** A working global vector store with race-free, idempotent indexing and provenance-aware retrieval — verifiable against fake embeddings.

**Deliverables.** `rag/embeddings.py`, `rag/store.py` (one Chroma collection; the **single serialized writer**), `rag/index.py` (header-aware chunking ~1000/200, deterministic `chunk_id = f(source_id, idx)`, `ingest`, `reconcile`), `rag/retrieve.py` (`candidates`: top-k → floor → provenance re-rank → dedup to `source_id`).

**Bringup.** Hand-seed a temp pool with 3–4 markdown files (with frontmatter); `ingest` each; `candidates(query, super_slug)` returns expected source ids in expected order.

**Exit criteria.** Ingest→retrieve works with `FakeEmbeddings`; re-ingest is a no-op; concurrent ingest doesn't corrupt.

**Tests (unit + integration).**

- Chunking: header path preserved as metadata; overlap respected.
- **Idempotent upsert**: ingesting the same source twice yields no duplicate chunks (deterministic `chunk_id`).
- Retrieval: below-floor hits dropped; provenance boost ranks same-`super_topic` chunks first while cross-topic stays eligible; chunk hits deduped to distinct `source_id`s.
- **Concurrency**: N threads ingesting simultaneously via the serialized writer → store consistent, count correct (integration).
- `reconcile()` indexes only un-indexed pool markdown.

---

## Phase 2 — Sources

**Goal.** Acquire, identify, dedup, and store sources to the pool with correct ids; convert PDFs; reconcile the inbox.

**Deliverables.** `sources/pool.py` (`save_web`, `save_pdf`, `get`, dedup, id rules), `sources/web.py` (Tavily search/extract), `sources/pdf.py` (`fetch -> path | Blocked`, `convert` via marker), `sources/inbox.py` (`reconcile`).

**Bringup.** With fakes: `save_web` a page → `_sources/<hash(url)>.md`; `save_pdf` → both `pdfs/<id>.pdf` and `<id>.md` under one id; drop a file in `_inbox/` → `reconcile` folds it in.

**Exit criteria.** SourceRefs and pool files are produced deterministically; ids follow the rule; dedup holds.

**Tests (unit).**

- `id = hash(url)` for web and PDF-by-URL; `hash(bytes)` for URL-less drop-ins; `content_hash` always stored.
- **Re-fetch updates in place** under the same id (markdown overwritten, `retrieved_at` bumped); one logical reference.
- Dedup by id before save and before embed; `save_pdf` writes raw + markdown sharing one id.
- `fetch` returns `Blocked` for the fake blocked-URL set; `convert` invoked only on success.
- `inbox.reconcile` converts/dedups/folds, then leaves `_inbox/` empty.

---

## Phase 3 — Relevance gate + verdict cache

**Goal.** The whitelist gate that judges a full document and distills evidence in one pass, with a correctly-scoped cache.

**Deliverables.** `gate.py` — `judge(super_slug, sub, source_ref) -> Verdict`; loads full markdown (truncate above `doc_size_cap`, log), single large-context call returns relevance + `EvidenceExtract`; cache keyed `(super_slug, sub_slug, source_id)` persisted under `state_dir`.

**Bringup.** With `FakeChat` scripted to "relevant + 2 quotes", judge a pooled doc → `Verdict.relevant` with evidence; second call hits cache (no LLM call).

**Exit criteria.** Verdicts cached and isolated by key; truncation handled.

**Tests (unit).**

- Relevant → `EvidenceExtract` with verbatim quotes; irrelevant → no evidence, discarded.
- **Cache key isolation**: same `source_id` under a different `(super, sub)` is judged independently (no leak).
- Cache hit avoids a second LLM call (assert `FakeChat` call count).
- Oversized doc truncated + warning logged; still judged.

---

## Phase 4 — Subagent subgraph (single sub-topic)

**Goal.** One sub-topic researched end-to-end (no fan-out) with the context-engineered loop, producing a `SubReport` and its per-sub `report.md`.

**Deliverables.** `nodes/subagent.py` — the subgraph `plan → acquire (RAG→web/PDF) → gate → reflect → synthesize draft → quality gate`; scratchpad; trajectory compaction (drop raw bodies post-distill); writes `research/<slug>/<sub>/report.md`; emits `SubReport` whose `Citation`s carry `supporting_quote` from distilled evidence.

**Bringup.** Build one `SubTopic`; run the compiled subgraph with all fakes; inspect the emitted `SubReport` + on-disk per-sub report.

**Exit criteria.** A sub-topic produces a grounded, cited report with fakes; loop terminates.

**Tests (integration with fakes).**

- **RAG-before-web**: local candidates consulted before Tavily (assert call ordering).
- **Whitelist gating**: only whitelisted sources appear as citations; non-whitelisted discarded.
- Quality gate: passes when coverage+evidence satisfied; **fails → loops** to acquire, bounded by `subagent_max_iterations`; **cap hit → emits with `shortfall`**.
- Compaction: raw document bodies are not retained in state after distillation (only scratchpad/evidence/refs).
- Per-sub `report.md` written at the expected path.

---

## Phase 5 — Orchestration (brief, fan-out, gather, checkpointer)

**Goal.** Multi-sub-topic runs with the real graph, HITL interrupts, and cross-process resumability.

**Deliverables.** `nodes/brief.py` (clarify/decompose/approve interrupts; target 3–7 / ceiling 12; persists `brief.md`), `nodes/supervisor.py` (Send fan-out; `dirty`-only vs all by `mode`), `nodes/gather.py` (join), `graph.py` (`build_graph()` wiring + conditional edges), real `persistence.py` checkpointer, `cli.py` `run`/`resume`/`sync` implemented; reconcile runs before fan-out.

**Bringup.** `research run` with `FakeChat` → clarify interrupt → resume answer → approve brief (2–3 sub-topics) → fan-out → gather. Then kill the process at an interrupt and `research resume <slug>` to continue.

**Exit criteria.** A run reaches `gather` with multiple sub-topics; interrupts resume in-process and across a restart.

**Tests (integration).**

- Interrupt/resume round-trip via `Command(resume=...)` for clarify and approval.
- `thread_id == slug`; slug seeded by CLI before the graph; `brief.md` frontmatter correct.
- Fan-out cardinality == sub-topic count; `seed_queries` honored.
- Supervisor selects **all** in `user_facing`, **dirty-only** in `autonomous`.
- **Cross-process resume**: re-instantiate the graph from the SQLite checkpointer and complete.
- `reconcile` runs before fan-out (ordering assertion).

---

## Phase 6 — Writer + verification (first full single pass)

**Goal.** The complete single-pass pipeline: question → report. Citations merged/renumbered; fail-loud verification.

**Deliverables.** `citations.py` (`merge`: collapse by `source_id`, global `[n]` renumber), `verify.py` (resolution + groundedness against the **full** source), `nodes/writer.py` (synthesize unified report; run verify; **writer-local** drop/revise on failure; persist `report.md` + `references.json`).

**Bringup.** Feed Phase-5 `gather` output → writer → unified `report.md` with global `[n]` and `references.json`.

**Exit criteria.** End-to-end single pass yields a finalized, fully-grounded `report.md` + `references.json` and per-sub reports.

**Tests (integration).**

- Citation merge: same `source_id` across sub-topics collapses to one entry; global renumber correct.
- **Resolution** catches a dangling `[n]` (fail-loud).
- **Groundedness** against full doc: a fabricated claim fails → dropped/revised in-writer; report not finalized while unclean (assert loop).
- `references.json` lists exactly the cited source ids; Bibliography holds no reports.

---

## Phase 7 — Refinement loop *(shipped)*

**Goal.** The two-tier evaluate loop with two-mode re-run, plateau detection, and bounded termination.

**Deliverables.** `nodes/evaluate.py` (coverage+support scoring against report **and** sources; `gaps`, `followups`, `queued_additions`; sets `mode`/`dirty`/`pending_handoff`; plateau via `history` `coverage_score`), conditional edges from `evaluate`.

**Bringup.** Script `FakeChat` so pass 1 has a gap → an **autonomous** round (dirty-only re-run, no interrupt) → coverage plateaus → set `pending_handoff`, **user-facing** full re-run → hand-off interrupt → user adds a follow-up → re-run → approve → END.

**Exit criteria.** Loop improves then hands off; never loops forever; report overwritten each round.

**Tests (integration).**

- Autonomous loop terminates on each of: full coverage, `auto_round_cap`, plateau.
- **`pending_handoff`** prevents infinite auto-loop and triggers the user interrupt after the pre-hand-off full re-run.
- Two-mode re-run: autonomous = dirty-only fan-out; user-facing = all.
- `max_rounds` cap enforced; single `report.md` overwritten per round; `history` records each round.
- Evaluator flags a thin/single-source answer as not fully supported.

---

## Phase 8 — Acquisition UX (blocked fetches)

**Goal.** The paywalled-source flow: blocked fetch → acquisition interrupt → manual download → resume → ingest, including batching and escape hatches.

**Deliverables.** Wire `pdf.fetch` `Blocked` → acquisition `interrupt()` carrying `{url, title, save_path=_inbox/<hash(url)>.pdf}`; resume responses (saved-at-path | unobtainable→gap | alternative-url); batched concurrent interrupts; reconcile-on-resume ingest.

**Bringup.** Force a fake blocked URL in a subagent → interrupt with `save_path` → test drops the file into `_inbox/` → `research resume` → file converted, pooled under the known id, whitelisted, cited.

**Exit criteria.** All three resume responses work; concurrent blocks batch; an acquisition interrupt can occur inside an autonomous round without making an editorial decision.

**Tests (integration).**

- Blocked → interrupt payload has correct `save_path`/id (nameable before bytes).
- **Saved-at-path**: resume finds the file → ingest → whitelist → citation under the known id.
- **Unobtainable** → recorded permanent gap, run continues, surfaced by evaluator.
- **Alternative URL** → fetched and ingested instead.
- Concurrent blocked fetches across the fan-out surface as one **batch** (resume map); answered in one pass.
- Acquisition interrupt fires during an autonomous round; no editorial decision is taken there.

---

## Phase 9 — Hardening + full end-to-end suite

**Goal.** Production-readiness: failure isolation, resume edge cases, observability, defaults tuned — plus the comprehensive E2E integration suite (next section) green.

**Deliverables.** Retry/backoff for transient errors; sub-topic **isolation** on hard failure / exhausted retries (status `isolated`, surfaced as gap); structured logging; tuned numeric defaults ([Architecture §8](Architecture.md#8-configuration--secrets-configpy)); resolve the [deferred items](Architecture.md#14-deferred--open-implementation-details) (`references.json` schema, cache/checkpointer file decision, `list/status`).

**Exit criteria.** The full E2E suite passes; a failing subagent never aborts a run; `list`/`status` usable.

**Tests.** The [end-to-end scenarios](#end-to-end-integration-testing) below, plus:

- **Failure isolation**: a subagent whose tools always error → retries exhausted → isolated → run completes with the gap surfaced; sibling sub-topics unaffected.
- Resume edge cases: resume after crash mid-fan-out; resume with a stale checkpoint.

---

## End-to-end integration testing

The E2E suite drives the **compiled graph** through realistic runs using only fakes, asserting both outputs (files on disk) and invariants (citations, dedup, loop control). It is the acceptance gate for Phase 9 and the regression net thereafter.

### Harness

- **Fakes** (Phase 0): `FakeChat` scripts per-role, per-step responses for a scenario; `FakeEmbeddings` gives deterministic vectors; `FakeTavily` serves canned hits + extract markdown; `FakePdf` serves bytes/markdown and a configurable blocked-URL set.
- **Workspace**: `tmp_workspace` fixture points `bibliography_dir`/`output_dir`/`state_dir` at temp dirs; assertions read the real files written there.
- **Driver**: a helper that invokes the graph and, at each `interrupt`, supplies the scripted `Command(resume=...)` — exercising the real interrupt/resume machinery and the SQLite checkpointer.
- **Markers**: `unit`, `integration`, `e2e`, and `live` (skipped unless `DR_RUN_LIVE=1`).

### Scenario A — Happy path (single pass)

Question → clarify → approve (2 sub-topics) → each subagent finds sources via fake RAG/Tavily → gate whitelists with evidence → per-sub reports → writer merges → verify clean → evaluate reports full coverage → user approves → END.
**Assert.** `report.md` exists with sequential `[n]` and a matching `references.json`; both per-sub `report.md`s exist; **every citation resolves to a whitelisted source and is grounded**; the pool contains exactly the expected source ids with no duplicates; Chroma chunk count matches; Bibliography contains no reports.

### Scenario B — Refinement loop

First pass leaves a guiding-question partial → one **autonomous** round (dirty-only re-run) improves it → plateau → **user-facing** full re-run → hand-off interrupt → user adds a follow-up sub-topic → user-facing re-run → approve.
**Assert.** `auto_round`/`round` counters; autonomous round re-ran only `dirty` sub-topics while user-facing re-ran all; `pending_handoff` gated the interrupt; `report.md` overwritten (single file); final report consistent across old+new sub-topics; `history` plateau recorded.

### Scenario C — Blocked PDF acquisition

A sub-topic's best source is a fake paywalled PDF → `Blocked` → acquisition interrupt. Run all three branches as separate cases: (1) test writes the file to `_inbox/` then resumes → ingested, whitelisted, cited; (2) resume "unobtainable" → permanent gap surfaced by the evaluator; (3) resume with an alternative open-access URL → fetched and cited.
**Assert.** Correct `save_path`/id; deterministic match on resume; gap recorded only on unobtainable; an acquisition interrupt occurring during an autonomous round makes no editorial change.

### Scenario D — Cross-process resumability

Run to the brief-approval interrupt; **discard the in-memory graph**; rebuild it from the checkpointer (`thread_id = slug`) and resume to completion.
**Assert.** State continuity (brief, counters, partial subreports) across the rebuild; identical final output to an uninterrupted run.

### Scenario E — Cross-run RAG reuse

Run question 1 to completion (populates the global pool + Chroma). Run a **different** question 2 whose sub-topic overlaps; the subagent's RAG query should surface a source saved during run 1.
**Assert.** The overlapping source is reused from the pool (no re-fetch via `FakeTavily` — assert it is not called for that source); the citation in run 2 keys to the same `source_id`.

### Scenario F — Failure isolation

One sub-topic's tools always error.
**Assert.** Bounded retries then isolation (status `isolated`); the run completes; the unified report and evaluator surface the isolated sub-topic as a coverage gap; siblings produce normal reports.

### Optional — Live smoke test (`DR_RUN_LIVE=1`)

A minimal real run against the configured Ollama host + Tavily on a trivial question, asserting only that a non-empty, fully-resolved `report.md` is produced. Not part of CI's default gate.

## Invariant test matrix

Each invariant is owned by the phase that introduces it and re-checked in the E2E suite.

| Invariant | Introduced | Primary test |
| --- | --- | --- |
| Slug stable & == `thread_id` | 0 / 5 | unit + Scenario D |
| Only whitelisted docs are cited | 3 / 4 | Phase 4 + Scenario A |
| Idempotent, race-free indexing | 1 | Phase 1 concurrency test |
| Source id rule (`hash(url)`/`hash(bytes)`) | 2 | Phase 2 unit |
| Relevance-cache key isolation | 3 | Phase 3 unit |
| Fail-loud verification (full-doc grounding) | 6 | Phase 6 + Scenario A |
| Loop terminates; `pending_handoff` correct | 7 | Phase 7 + Scenario B |
| Acquisition ≠ editorial; batched | 8 | Phase 8 + Scenario C |
| Failure isolation, no run abort | 9 | Scenario F |
| Cross-run knowledge reuse | 1–2 | Scenario E |

## Definition of done (per phase)

A phase is complete when: its modules implement the [Architecture §6](Architecture.md#6-subsystem-contracts) contracts; its unit/integration tests are green and hermetic; the bringup step runs by hand; and any divergence from the design docs has been reflected back into them (docs-as-source-of-truth).
