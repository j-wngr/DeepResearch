# Architecture

The canonical, implementation-facing architecture for the Deep Research framework. It turns the [DesignBrief](DesignBrief.md) (the *what* and *why*) into the *how*: runtime topology, the LangGraph graph, state and data models, the Python module layout, subsystem contracts, storage, configuration, and a build order. Domain terms are defined in [CONTEXT.md](../CONTEXT.md).

## 1. Overview

A single CLI process drives a LangGraph state machine that decomposes a **super-topic** into **sub-topics**, researches each in a parallel subagent (RAG-first, web/PDF fallback, full-document relevance gate), synthesizes a unified report, then refines it — first autonomously, then with the user. Two durable stores back it: a **global knowledge layer** (central source pool + one Chroma collection) reused across runs, and a **per-run state layer** (LangGraph checkpointer). All LLM work runs on a **remote Ollama** server with role-tiered chat models.

Guiding invariants (see DesignBrief for rationale):

- The super-topic **slug** is the one identity: filesystem key *and* `thread_id`.
- Only **whitelisted** full documents are evidence or citable; RAG chunks are discovery only.
- The Bibliography stores **sources only**; deliverables live under a separate outputs tree.
- Verification is **fail-loud**; nothing ungrounded ships.

## 2. Runtime topology

- **Process model** — one foreground CLI invocation per command. The graph runs until it reaches an `interrupt()` or terminates. On interrupt the process may exit; state persists in the checkpointer under the slug.
- **Entry points** (CLI verbs):
  - `research run "<question>"` — start a new run; derives the slug **up front from the raw question** (slugified title + short hash) and seeds the initial `ResearchState` (`question` + `slug`) before invoking the graph, so a `thread_id` exists before the first interrupt. The slug is opaque and never changes, even if clarify/approval reword the question. Runs an embed-model pre-flight check (`endpoints.check_embed_model`) before entering the graph and exits with a clear error if the configured model is not available on the Ollama host.
  - `research resume <slug>` — reattach to a thread and answer the pending interrupt (editorial or acquisition). Also runs the embed-model pre-flight check on startup.
  - `research sync` — standalone: reconcile `Bibliography/_inbox/` (convert/dedup/fold into the pool, index) without starting research. The same reconcile also runs automatically at the start of every `run`/`resume`, before fan-out.
  - `research improve <slug>` — re-open the evaluation/refinement loop on a *finished* run (`user_approved=True`). Injects `pending_handoff=True` into the checkpoint via `graph.update_state(as_node="writer")` so `evaluate` fires immediately, presenting the current report + coverage gaps. Research re-runs only if the user requests changes.
  - `research prune [--dry-run]` — quality-gate all sources in `Bibliography/_sources/`; remove garbage (heuristic + LLM check) from the pool and Chroma, then reconcile the RAG index. `--dry-run` reports what would be removed without touching files.
  - `research list` / `research status <slug>` — inspect threads (optional, nice-to-have).
- **Interrupt handling** — the CLI renders the pending interrupt payload (clarify questions, brief for approval, follow-ups, or a batch of blocked downloads) and collects the response, then calls the graph with a `Command(resume=...)`. Concurrent acquisition interrupts arrive as a batch (resume map keyed by interrupt id).
- **Remote dependency** — Ollama (chat + embeddings) over HTTP at a configurable base URL; the only stateful external service besides Tavily.

## 3. LangGraph structure

### 3.1 Top-level graph

| Node | Type | Responsibility | Out-edges |
| --- | --- | --- | --- |
| `clarify` | node (+interrupt) | ask clarifying questions; refine the question on resume | → `decompose` |
| `decompose` | node | decompose the (possibly refined) question into sub-topics; build the `Brief` | → `approve` |
| `approve` | node (+interrupt) | present the brief for user approval; regenerate on feedback; persist `brief.md` on approval | → `supervisor` (via conditional edge) |
| `supervisor` | **conditional routing function** (not a separate node in LangGraph) | fan out one `research_subagent` per in-scope sub-topic by returning `Send`s | → `research_subagent` (×N) |
| `research_subagent` | subgraph | per-sub-topic ReAct loop (§3.2); writes its per-sub `report.md` | → `gather` |
| `gather` | join node | fan-in barrier the subagents return to; merge is done by the `subreports` channel reducer | → `writer` |
| `writer` | node | synthesize unified report; merge/renumber citations; run verification; persist `report.md` + `references.json` | → `evaluate` |
| `evaluate` | node (+interrupts) | score coverage+support; route the refinement loop | → `supervisor` (re-run) \| `END` |

Conditional edges out of `evaluate` implement the two-tier refinement loop via explicit state (#4):

- `supervisor` reads `mode`: in `autonomous` it fans out only **`dirty`** sub-topics; in `user_facing` it fans out **all** of them.
- `evaluate` decides each visit: if gaps remain and `auto_round < cap` and coverage is still improving → set `mode=autonomous`, mark changed sub-topics `dirty`, loop to `supervisor` (no interrupt). Otherwise prepare the hand-off: set `mode=user_facing` + `pending_handoff=True` and loop once more for the full consistency re-run.
- When `evaluate` runs again with `pending_handoff` set, it does **not** loop — it raises the user interrupt (or `END` after approval). This flag is what prevents an infinite auto-loop and guarantees the user sees a freshly re-run, consistent report.

### 3.2 Subagent subgraph

`plan → acquire (RAG → web/PDF → source quality gate) → relevance gate → reflect → synthesize draft → coverage quality gate`

**Source quality gate** (inline in `acquire`): newly fetched web/PDF sources are judged by `source_quality.assess()` before entering the relevance gate — a two-phase check: heuristic pre-screen (`sources/quality.is_acceptable`) then LLM `"quality"` role. Garbage (navigation pages, link farms, cookie walls) is deleted from the pool immediately and not passed downstream. RAG-retrieved existing pool sources skip this check during research runs; standalone cleanup is handled by `prune`. Verdicts are cached at `state_dir/quality_cache.json` keyed by `source_id` only (topic-agnostic). Controlled by `QUALITY_GATE_ENABLED` (default `true`).

**Coverage quality gate** (subgraph exit criterion): scores **coverage** (every `guiding_question` addressed?) and does a **cheap groundedness self-check against the distilled evidence** (does each claim have a backing quote?). Pass → emit report; fail → loop to `acquire` (bounded by `max_iterations`); cap hit → emit with shortfall flagged. Acquisition gaps are carried as context through the loop and are surfaced in the final shortfall only if the subagent cannot recover with other sources. This is a fast per-sub-topic check — authoritative full-document groundedness happens once, globally, at `verify` (#6, §6.4). On emit, the subagent **writes its own** `research/<slug>/<sub-topic-slug>/report.md` (#8). Acquisition may raise a blocking interrupt for a manual download (§6.2).

The loop is context-engineered (§10): it carries a **scratchpad** (findings, tried queries, open questions) and **distilled evidence** rather than raw message history. Concretely, once the gate distills a document the raw body is dropped from the subagent's message list (trajectory compaction) — only the `scratchpad`, `EvidenceExtract`s, and `SourceRef`s persist in state.

### 3.3 Checkpointer

`langgraph.checkpoint.sqlite.SqliteSaver` (sync) over `.deepresearch/checkpoints.sqlite`, `thread_id = slug`. Upgrade path to the async/Postgres saver is isolated behind `persistence.py`.

## 4. State & data models

Graph state is a `TypedDict` with reducer-annotated fields; payloads are Pydantic models (`models.py`) for validation and (de)serialization.

### 4.1 Top-level state (`state.py`)

```python
class ResearchState(TypedDict):
    question: str                      # verbatim super-topic
    slug: str                          # derived up front from question; == thread_id
    brief: Brief | None
    round: int                         # refinement round counter
    auto_round: int                    # autonomous-tier counter
    mode: Literal["autonomous", "user_facing"]   # current round tier (#4)
    pending_handoff: bool              # set before the pre-hand-off full re-run (#4)
    subreports: Annotated[dict[str, SubReport], merge_subreports]  # keyed by subtopic_slug
    report: str | None                 # current unified report.md body
    report_references: list[SourceRef] # references used in the current report body
    coverage: CoverageReport | None    # latest evaluator output
    history: Annotated[list[RoundRecord], add]  # per-round coverage for plateau detection
    user_approved: bool                 # evaluator sets true after explicit hand-off approval; routes to END
    # Verification state (fail-loud, resolved inside the writer step)
    verify_ok: bool                    # did the latest verification pass?
    verify_attempts: int               # bounded revision attempts consumed
    verify_unsupported: list           # claims that failed groundedness (post-loop)
    verify_dangling: list              # citation numbers that failed resolution
```

`merge_subreports` **upserts by `subtopic_slug` (latest wins)** so re-runs replace, never duplicate; sub-topics dropped from the brief are pruned explicitly. State therefore holds exactly one current report per live sub-topic.

### 4.2 Subagent state

```python
class SubAgentState(TypedDict):
    slug: str                          # super-topic slug
    subtopic: SubTopic
    iteration: int
    scratchpad: str                    # write: running notes, tried queries, open questions
    whitelisted: list[SourceRef]       # references only (full bodies stay on disk)
    evidence: list[EvidenceExtract]    # compress: distilled, sub-topic-relevant quotes
    draft: str | None
    candidates: list[SourceRef]        # transient: raw body refs dropped after gate (trajectory compaction)
    pending_acquisitions: list[AcquisitionRequest]  # batched blocked-fetch requests for the next interrupt
    acquisition_gaps: list[str]        # permanent gaps ("source unobtainable: <url>"); skip on re-acquire
    subreport: SubReport | None        # emitted on quality-gate pass or cap-hit (with shortfall)
    quality_gate_result: QualityGateResult | None  # latest quality-gate verdict; drives loop/END
    subreports: dict[str, SubReport]   # pass-through channel for the parent state's subreports
    isolated: bool                     # set by _with_isolation on hard failure; routes remaining nodes to END
```

Note the deliberate **state-schema isolation** (§10): full document bodies never live in this state — only `SourceRef`s and distilled `EvidenceExtract`s do. The full markdown is loaded from the pool on demand (gate, final verification) and discarded.

### 4.3 Core Pydantic models (`models.py`)

- `Brief { question, slug, thread_id, subtopics: list[SubTopic] }`
- `SubTopic { slug, title, scope, guiding_questions: list[str], seed_queries: list[str] = [], status: Literal["pending","active","done","isolated"], dirty: bool = False }` — `dirty` marks new/changed sub-topics for incremental autonomous re-runs (#4).
- `SourceRef { id, type: Literal["web","pdf"], url | None, title, source_path, retrieved_at, content_hash, doi: str | None }` — `id` = `hash(url)` whenever a URL exists (web pages **and** PDFs-by-URL), else `hash(bytes)` for URL-less drop-ins. `content_hash` is always stored separately (dedup / change-detection). `doi` is auto-extracted from the markdown body on save (best-effort; `None` when absent). `id` is the citation key.
- `Citation { source_id, claim, supporting_quote | None }`
- `EvidenceExtract { source_id, points: list[EvidencePoint] }`; `EvidencePoint { claim, quote }` — distilled at gate time from the full document; the compression unit the loop and writer carry instead of full bodies.
- `SubReport { subtopic_slug, body, citations: list[Citation], shortfall: str | None }`
- `Verdict { super_slug, sub_slug, source_id, relevant: bool, reason, evidence: EvidenceExtract | None }` — relevance-cache entry; carries the distilled evidence when `relevant`.
- `CoverageReport { per_question: list[QuestionScore], gaps: list[str], followups: list[SubTopic], queued_additions: list[SubTopic] }`
- `QuestionScore { subtopic_slug, question, status: Literal["answered","partial","unanswered"], supported: bool }`
- `QualityGateResult { passed: bool, coverage_score: float, reason: str }` — subagent quality-gate pass/fail verdict; drives the loop/END routing.
- `RoundRecord { round, mode, coverage_score: float, fully_covered: bool }` — one per round in `history`; the `coverage_score` series drives plateau detection.

## 5. Module layout (`src/deepresearch/`)

```text
src/deepresearch/
  __init__.py
  cli.py            # Typer/argparse entry: run, resume, sync, improve, prune, list, status
  config.py         # Settings (env + file), path roots, model tiers, caps
  paths.py          # slug derivation, output_dir/bibliography_dir resolution
  models.py         # Pydantic domain models (§4.3)
  state.py          # TypedDict graph states + reducers
  graph.py          # build_graph(): nodes, edges, conditional routing, compile
  persistence.py    # checkpointer factory (SQLite now; pluggable)
  llm.py            # Ollama chat client; role -> model resolution
  nodes/
    brief.py        # clarify_node, decompose_node, approve_node (split brief generation)
    supervisor.py   # fan-out via Send; dirty-only (autonomous) or all (user_facing) by mode
    subagent.py     # research_subagent subgraph (plan..quality gate); writes its per-sub report.md
    gather.py       # fan-in join node (merge via the subreports channel reducer)
    writer.py       # synthesis + citation merge + verify; writes report.md + references.json
    evaluate.py     # evaluator + refinement routing
  rag/
    embeddings.py   # Ollama embeddings
    store.py        # Chroma collection wrapper (upsert, query, metadata)
    index.py        # chunking, serialized upsert (single writer), reconcile
    retrieve.py     # query -> top-k -> floor -> provenance re-rank
  sources/
    pool.py         # central content-addressed pool; ids, dedup, save
    web.py          # Tavily search + extract -> markdown
    pdf.py          # httpx fetch, blocked-fetch interrupt, pluggable convert (pymupdf4llm/marker/remote)
    quality.py      # word-count + link-density pre-filter for web extracts
    inbox.py        # _inbox reconcile
  gate.py           # full-document relevance gate + verdict cache
  source_quality.py # topic-agnostic LLM source quality gate + quality cache
  citations.py      # key resolution, merge, global renumber
  verify.py         # resolution (programmatic) + groundedness (LLM)
```

Dependency direction: `nodes/*` orchestrate and depend on subsystems (`rag/`, `sources/`, `gate`, `citations`, `verify`, `llm`); subsystems depend only on `models`, `config`, `paths`. No subsystem imports a node. `graph.py` wires nodes; `cli.py` drives `graph.py` + `persistence.py`.

One deliberate exception: `rag/index.py`'s `reconcile()` calls `sources/inbox.py`'s `reconcile()` via a deferred import, because the Architecture §6.1 contract requires inbox processing before indexing. This is the only subsystem→subsystem seam; it is kept as a deferred import to avoid a module-level circular dependency.

## 6. Subsystem contracts

### 6.1 RAG

- `embeddings.embed(texts) -> vectors` via Ollama (`mxbai-embed-large` default).
- `store`: one global Chroma collection under `.deepresearch/chroma/`; `upsert(chunks)`, `query(text, k)`; chunk metadata: `source_id` (the canonical citation key, #5), `content_hash`, `source_path`, `source_url`, `title`, `type`, `super_topic`, `sub_topic`.
- `index.ingest(source_ref)`: header-aware chunk (~1000/200) → embed → **replace all chunks for that source through the single serialized writer** (a process-wide lock/queue over one shared Chroma client; reads stay concurrent). Chunk ids are deterministic — `chunk_id = f(source_id, chunk_index)` — so replacement is idempotent and removes stale chunks if a source shrinks. Called by the subagent immediately after a save, so siblings can retrieve it within the run.
- `index.reconcile()`: inbox first (convert/dedup/fold drop-ins), then index pool markdown that is either new or whose `content_hash` has changed since it was last indexed (so a re-fetched web source with updated content is re-embedded, not silently stale). Reconcile uses the source frontmatter for chunk metadata (`type`, `title`, URL, path) and replaces all previous chunks for changed sources. Runs once at `run`/`resume` startup **before** the supervisor fan-out (so it never overlaps subagent writes), and standalone via `research sync`.
- `retrieve.candidates(query, super_slug)`: top-k → drop below absolute floor → re-rank with provenance boost for matching `super_topic` → dedup chunk hits to distinct `source_id`s.

### 6.2 Sources

- **Type-specific save** (#7), each computing the id from the inputs it has and returning a `SourceRef`:
  - `pool.save_web(markdown, url, meta) -> SourceRef`: `id = hash(url)`; write `_sources/<id>.md` (+frontmatter).
  - `pool.save_pdf(pdf_bytes, markdown, url, meta) -> SourceRef`: `id = hash(url)` if `url` else `hash(pdf_bytes)`; write `_sources/pdfs/<id>.pdf` **and** `_sources/<id>.md` under the same id.
  - `pool.get(id) -> full markdown`. Both savers dedup before writing.
- `web.search(query) -> list[SearchHit]` where `SearchHit { url, title, snippet }`; `web.extract(url) -> markdown` (Tavily, with a configurable inter-request delay). Web id = `hash(url)`; re-fetch updates in place. Extracted markdown is passed through a **heuristic pre-filter** (`sources/quality.py`) before save: pages below `min_source_words` (default 150) or above `max_link_density` (default 0.5) are silently discarded. After save, newly fetched sources also pass through the **LLM source quality gate** (`source_quality.py`) before entering the relevance gate — see §3.2.
- `pdf.fetch(url) -> path | Blocked`: httpx download; on 403/login-wall returns `Blocked`, which the subagent turns into an **acquisition `interrupt()`** carrying `{url, title, save_path=_inbox/<id>.pdf}` with `id = hash(url)` — nameable before the bytes exist (#2). On resume the CLI auto-detects whether the file was placed at the expected path: present → `saved`; absent → `unobtainable` (permanent gap). The `alternative` response kind (user pastes an open-access URL) is supported at the model layer but not surfaced by the current CLI. `pdf.convert(path) -> markdown` via **pymupdf4llm** (default, `PDF_CONVERTER=pymupdf`), optional **marker** (`PDF_CONVERTER=marker`, requires `uv sync --extra ocr`), or a **remote converter service** (`PDF_CONVERTER=remote`, `PDF_CONVERTER_URL=...`).
- `inbox.reconcile()`: convert/dedup/fold dropped PDFs into the pool.

### 6.3 Relevance gate (`gate.py`)

`gate.judge(super_slug, sub, source_ref) -> Verdict`: cache lookup on `(super_slug, sub_slug, source_id)`; on miss, load full markdown (truncate above cap, log), run the large-context model with the sub-topic mandate to (a) judge relevance and (b) **distill the relevant points as verbatim quotes** (`EvidenceExtract`) in the same pass. Only `relevant` sources enter the whitelist; their evidence is cached with the verdict so neither the loop nor the writer re-loads the full body. Cache persisted under `.deepresearch/` (e.g. a small SQLite/JSON table).

### 6.4 Citations & verification

- `citations.merge(subreports) -> (body, references)`: collapse by `source_id`, renumber `[n]` globally. Subagents are prompted to cite stable source ids (`[source_id]`); numeric fallback citations are interpreted against the subagent's whitelisted-source order, not evidence-point order.
- `verify.check(report, references)`: the **authoritative** pass (the subagent quality gate did only a cheap evidence-level self-check, #6). (1) resolution — every `[n]` resolves to a whitelisted ref; (2) groundedness — LLM confirms each cited claim **against the full source document** (loaded from the pool for this step), not merely against the distilled extract — so the compression in §10 never weakens the citation guarantee, and writer paraphrase drift is caught post-merge. Fail-loud, resolved **within the writer step**: a failing claim is either dropped or fixed by a bounded in-writer revision of that passage using the full source already loaded for grounding (loop locally until clean). No subagent re-run and no extra graph edge — the topology stays `writer → evaluate`. The report is not finalized until clean.

### 6.5 LLM tiers (`llm.py`)

`chat(role, messages)` where `role ∈ {clarify, gate, synth, writer, eval, quality}` maps to a configured model. `gate`/`synth`/`quality` use the large-context model; `writer`/`eval` the strong model; `clarify` the fast model.

## 7. Storage & filesystem layout

```text
.deepresearch/                 # run state (git-ignored)
  checkpoints.sqlite           # LangGraph checkpointer
  chroma/                      # global vector store
  relevance_cache.json         # whitelist verdict cache (keyed by super+sub+source)
  quality_cache.json           # source quality verdict cache (keyed by source_id)

Bibliography/                  # knowledge: sources only (bibliography_dir)
  _inbox/                      # user-dropped / manual downloads, pre-reconcile
  _sources/
    <id>.md                    # pooled markdown (+frontmatter)
    pdfs/<id>.pdf              # raw PDFs

Research/                      # deliverables (output_dir)
  <super-topic-slug>/
    brief.md                   # frontmatter: question, slug, thread_id
    report.md                  # unified report (overwritten each round)
    references.json            # source ids referenced
    <sub-topic-slug>/report.md # per-sub-topic report
```

`.deepresearch/` is git-ignored; `Bibliography/` and `research/` are durable (committable). Both roots are configurable.

## 8. Configuration & secrets (`config.py`)

Layered: env vars > config file > defaults. Settings:

- `ollama_base_url`, optional separate `embed_base_url` (defaults to `ollama_base_url`), and optional `ollama_api_key` for authenticated/cloud hosts.
- Model tiers (`model_fast`, `model_long`, `model_writer`) and `embed_model`.
- Paths: `bibliography_dir` (default `./Bibliography`), `output_dir` (default `./Research`), `state_dir` (default `./.deepresearch`).
- PDF converter: `pdf_converter` (`pymupdf` | `marker` | `remote`, default `pymupdf`); `pdf_converter_url` and `pdf_converter_api_key` for the remote mode.
- Source quality: `min_source_words` (default 150), `max_link_density` (default 0.5) — heuristic pre-filter thresholds. `quality_gate_enabled` (`QUALITY_GATE_ENABLED`, default `true`) — enable/disable the LLM source quality gate for newly fetched sources.
- Tavily: `tavily_api_key` — **from environment only**, never persisted; `tavily_request_delay` (default 1.0 s) — inter-request pause to avoid rate-limiting.
- Caps/knobs: `max_concurrency`, `subagent_max_iterations`, `auto_round_cap`, `max_rounds`, `subtopics_target` (3–7), `subtopics_ceiling` (12), `retrieval_k`, `similarity_floor`, `provenance_boost`, `doc_size_cap`, `writer_max_revisions`.

Concrete numeric defaults are chosen at implementation time against the real Ollama host (DesignBrief defers them deliberately).

## 9. Concurrency & failure model

- **Concurrency** — `Send` fan-out bounded by `max_concurrency` (remote-Ollama-throughput limited). **RAG writes go through a single serialized writer** (lock/queue over one Chroma client); reads are concurrent; deterministic chunk ids plus replace-by-source semantics make repeated indexing safe and clear stale chunks. `reconcile()` runs once before fan-out so it never races subagent writes. There is **no runtime full rebuild** — indexing is incremental by changed source; the only full reindex is the deliberate, offline embedding-model change (exclusive).
- **Failure** — transient errors: bounded retry + backoff; hard failures / exhausted retries: isolate the sub-topic (status `isolated`, surfaced as a gap). A **blocked fetch is not a failure** — it routes to the acquisition interrupt.
- **Resumability** — every interrupt is checkpointed; `research resume <slug>` continues. Reconcile-on-resume re-indexes anything added out-of-band.

## 10. Context engineering

Long multi-round runs and full-document gating make the token budget a first-order concern, so the system applies LangChain's four-strategy taxonomy — **write, select, compress, isolate** ([context engineering for agents](https://www.langchain.com/blog/context-engineering-for-agents)).

- **Write** (persist outside the window). *Cross-session:* the source pool + global Chroma collection are durable semantic memory; the relevance-verdict cache is procedural memory; briefs/reports are written to the outputs tree. *Intra-task:* each subagent keeps a **scratchpad** (`SubAgentState.scratchpad`) of findings, tried queries, and open questions, checkpointed so the ReAct loop never depends on raw message history.
- **Select** (retrieve only what's needed). The RAG candidate finder *is* selective retrieval — pull relevant sources, not the whole library — and the relevance gate then loads full documents only for the few candidates (top-k → floor → provenance re-rank, §6.1). Verdict-cache lookups select prior judgments instead of re-reading. (Tool-selection-via-RAG is not needed — the toolset is small and fixed.)
- **Compress** (keep only essential tokens). At whitelist time the gate emits **distilled evidence** (`EvidenceExtract`: sub-topic-relevant claims with verbatim quotes), cached with the verdict. The subagent synthesis and the writer work from distilled evidence, not full bodies; raw tool outputs (search results, fetched document text) are **dropped from the trajectory once distilled** (trajectory compaction). The full-document grounding invariant is preserved exactly where it counts: final **groundedness verification re-checks each claim against the full source** (§6.4), so compression accelerates drafting without weakening citations.
- **Isolate** (split context across subsystems). The multi-agent fan-out is isolation by construction — each subagent has its own window, mandate, and tools and never sees sibling state. **State-schema isolation** is deliberate: the subagent LLM is shown only its mandate + scratchpad + distilled evidence + current step, never the global pool, the full message history, or other sub-topics. Token-heavy objects (full markdown, PDFs) live on disk in the pool and are referenced by `source_id`, loaded only at the gate and final verification, never parked in graph state. PDF conversion (pymupdf4llm by default; marker or remote as alternatives) and embeddings run out-of-process on their own hosts.

## 11. External dependencies

`langgraph`, `langchain-core`, `langchain-ollama` (chat + embeddings), `chromadb`, `tavily-python`, `pymupdf4llm` (default PDF converter), `httpx`, `pydantic`, a CLI lib (`typer`), all via `uv`. `marker-pdf` is an optional extra (`uv sync --extra ocr`) for the higher-quality OCR converter mode. Integration seams isolated per subsystem so each is independently testable with fakes.

## 12. Dev workflow

- `uv sync` — install; `uv run deepresearch ...` — run the CLI; `uv add <pkg>` — deps.
- `uv run pytest` — tests. Unit-test subsystems with fakes (fake Ollama/Tavily/marker); integration-test the graph with an in-memory checkpointer and a stubbed LLM.
- Lint/format per chosen toolchain (e.g. `ruff`).

## 13. Build order (milestones)

The phased, test-gated expansion of this build order — with per-phase bringup steps, exit criteria, and the end-to-end integration suite — lives in [ImplementationPlan.md](ImplementationPlan.md).

1. **Skeleton** — `uv` project, `config`, `paths`, `models`, slug derivation; CLI stub.
2. **RAG core** — `embeddings`, `store`, `index` (ingest + reconcile), `retrieve`; verify against a hand-seeded pool.
3. **Sources** — `pool`, `web` (Tavily), `pdf` (httpx + marker), `inbox`; no blocked-fetch UX yet.
4. **Subagent** — subgraph with gate (emitting distilled evidence) + quality gate; scratchpad + trajectory compaction (§10); single sub-topic end-to-end (no fan-out).
5. **Orchestration** — `clarify`/`decompose`/`approve` nodes (split brief generation with per-stage interrupts), `supervisor` fan-out, `gather`, checkpointer; multi-sub-topic run.
6. **Writer + verification** — synthesis, citation merge/renumber, fail-loud verify.
7. **Refinement loop** — `evaluate`, two-tier autonomous/user routing, two-mode re-run, plateau/caps.
8. **Acquisition UX** — blocked-fetch interrupt (batched), resume responses, inbox round-trip.
9. **Hardening** — failure isolation, retries, resume edge cases, defaults tuning.

## 14. Deferred / open implementation details

- Concrete numeric defaults for §8 knobs (tune on the host).
- Optional `research list/status` ergonomics.
- Re-run-all vs. cached-source reuse cost profiling (informs whether the two-mode split needs further tuning).

*Resolved details now in code:*
- `references.json` schema — a JSON list of `SourceRef` objects for the sources cited in the final report (written by `writer.py`).
- Relevance cache/checkpointer storage — kept in separate files: `state_dir / checkpoints.sqlite` (LangGraph) and `state_dir / relevance_cache.json` (gate verdict cache).
- Transient-error retry/backoff — a `retry.py` helper (`retry_call`) wraps the LLM and Tavily seams with bounded attempts (defaults: 3 / 0.5s initial / 2.0x backoff, tunable via `LLM_MAX_RETRIES`, `LLM_RETRY_INITIAL_INTERVAL`, `TAVILY_MAX_RETRIES`); the allow-list is narrow (transport-class errors only) so application/model errors are not retried.
- Failure isolation — each subagent node is wrapped with a `_with_isolation` decorator (`nodes/subagent.py`). The decorator re-raises `GraphInterrupt` unchanged (so LangGraph's interrupt/resume mechanism works correctly across parallel subagents) and converts any other `Exception` into an isolated `SubReport` with a citation-free shortfall and a per-sub `report.md` written to disk. Conditional edges after each node check `isolated=True` and route to `END`, so sibling sub-topics continue normally. This flat, single-compiled-subgraph structure (no nested `compiled.invoke()`) is required for the outer graph's checkpointer to manage all task state and for `CONFIG_KEY_RESUME_MAP` to propagate correctly to inner `interrupt()` calls.
- `research list` / `research status` — `run_state.py` reads thread metadata from the SQLite checkpointer and per-run file presence from the outputs tree; the CLI commands render one `RunSummary` per thread (or a single summary for a slug) with `brief`/`report` presence, round counters, and `pending_handoff` state.
