# Deep Research Framework

The proposed framework is inspired by the LangChain ["Open Deep Research"](https://www.langchain.com/blog/open-deep-research) framework. The source code is publicly available on GitHub: <https://github.com/langchain-ai/open_deep_research>.

The proposed deep research framework differs in some points from the LangChain solution:

- A research question (super-topic) is broken down into different sub-topics. These are defined together with the user during the generation of the research brief.
- Every super-topic has a **stable slug** derived **at run start** from the raw question (slugified title + a short uniqueness hash). The slug is the durable key for all on-disk paths and for run state (it is the `thread_id` from the first step), and never changes even if the question is later reworded. The verbatim question, slug, and `thread_id` are recorded in the super-topic's `brief.md`.
- The **Bibliography is the source store only**: gathered sources are saved once in a **central source pool**, shared across all runs — mirroring the global vector store. Each source has a **stable id**: `hash(url)` whenever a URL exists (web pages **and** PDFs reached by URL — re-fetching updates the doc in place), and `hash(bytes)` only for URL-less drop-in files. A `content_hash` is always stored separately for dedup/change-detection.
- **Deliverables live in a separate, configurable outputs tree** (default `research/`), not in the Bibliography. Each super-topic gets `research/<super-topic-slug>/` holding its `brief.md`, `report.md`, per-sub-topic `report.md`s, and the list of source ids it references. Both roots are configurable (`output_dir`, `bibliography_dir`) so reports can be committed while the bulky source library lives elsewhere.
  - Source pool markdown: `Bibliography/_sources/<id>.md`
  - Raw PDFs (web-downloaded or user-supplied): `Bibliography/_sources/pdfs/<id>.pdf`
  - Per-super-topic deliverables: `research/<super-topic-slug>/...`
- PDFs are parsed to markdown and stored in the source pool alongside the original. The default converter is **pymupdf4llm** (fast, no extra dependencies); **marker** (OCR + LLM refinement) is available as an optional extra (`uv sync --extra ocr`); a **remote converter service** is also supported for offloading conversion.
- The research supervisor spawns a subagent per sub-topic.
  - **Tavily** is used for searching the web and extracting markdown from HTML.
  - A **local RAG** is built from the markdown files in the Bibliography. The RAG can be queried by the research subagents to find information already present in the Bibliography.
- Each research subagent synthesizes a report with its findings. **All information must be backed by citations.**
- The research supervisor passes the reports from the subagents to a **writer agent**, which synthesizes the findings into one consistent research report.

## Orchestration & flow

The system is built on **LangGraph** (on top of LangChain). LangGraph provides the stateful graph, checkpointing, human-in-the-loop interrupts, and fan-out primitives that this multi-agent pipeline needs; LangChain models and tools are used underneath.

### Why LangGraph over bare LangChain

- **Checkpointing** → long, expensive research runs become resumable (re-run a failed sub-topic without redoing the rest).
- **`interrupt()`** → the interactive brief-approval step is a native primitive rather than a custom loop.
- **`Send` API** → clean fan-out of one subagent per sub-topic.

### Top-level graph

1. `generate_brief` (implemented as `clarify` → `decompose` → `approve` nodes) — breaks the super-topic into sub-topics, then `interrupt`s for the user to approve/edit the brief.
2. `supervisor` — fans out via the `Send` API, dispatching one `research_subagent` per sub-topic.
3. `research_subagent` (subgraph) — see below.
4. `gather` — collects the per-sub-topic reports (state reducer).
5. `writer` — synthesizes the unified final research report from all sub-topic reports.
6. `evaluate` — checks the report (and its sources) against the brief and loops back to `supervisor` with an updated brief — first autonomously to raise quality, then with the user — or ends after user approval. See [Evaluation & refinement loop](#evaluation--refinement-loop).

### Research subagent subgraph (inner loop)

A ReAct-style loop per sub-topic:

`plan → find candidates (RAG, else Tavily/PDF → save → ingest) → fetch full docs → relevance gate (whitelist) → reflect → synthesize draft report → quality gate`

The acquisition and whitelisting mechanics are detailed in [RAG & sources](#rag--sources).

**Quality gate (the subagent's exit criterion).** Instead of a vague "sufficient coverage" check, each subagent scores its own draft against its `guiding_questions` for **coverage** (every question addressed?) and a **cheap groundedness self-check against its distilled evidence** (does each claim have a backing quote?). Authoritative full-document groundedness is not repeated here — it runs once, globally, at final verification. On **pass**, it emits the report. On **fail**, it loops back to gather more (refine queries, fetch more sources), bounded by a **maximum iteration count**. If the cap is hit while still failing, it emits the best report with the **shortfall explicitly flagged** so the final evaluator can pick it up. This is the only per-sub-topic check; the final evaluator handles cross-topic coverage, not re-judging single sub-topics.

### Concurrency

Subagents run **in parallel with a configurable max-concurrency cap**. Because all agents share one remote Ollama server, the cap is bounded by that server's throughput; it defaults conservatively and is tuned to the host.

### Persistence (two distinct layers)

- **Run state** — LangGraph checkpointer (e.g. SQLite under `.deepresearch/`), keyed by a `thread_id` that **equals the super-topic slug** (one identity per research question). Enables resuming an interrupted run; re-running the same super-topic resumes its existing thread.
- **Knowledge** — the Bibliography on disk (a central source pool, sources only) and a single global vector store. Both are durable and reused *across* runs. Human deliverables (briefs, reports) live separately under the configurable outputs tree (`research/<slug>/`), not in the Bibliography.

### Human-in-the-loop & resumption

The interrupt points (clarify, brief approval, evaluate follow-ups, and **manual-download acquisition blocks**) can be hours apart. They are **resumable across processes**: on `interrupt()` the process may exit, with all state held in the checkpointer under the slug. A later CLI invocation — `research resume <slug>` — reattaches to the same thread, answers the pending prompt, and continues. Human latency is thus decoupled from any live process, and runs survive restarts or crashes.

**Two kinds of interrupt.** *Editorial* interrupts ask for human judgment (clarify, brief approval, evaluate follow-ups) and define the human-in-the-loop tier. *Acquisition* interrupts are mechanical necessities — a blocked PDF fetch that only a human can complete (log in, click download). Acquisition interrupts may fire **even during autonomous refinement rounds**: "autonomous" means no human *judgment* is required, not that the process can never pause to physically obtain a file. So a paywalled source can block any round; an autonomous round still makes no editorial decisions.

**Concurrent interrupts batch.** Because subagents run as a parallel fan-out, several can raise an acquisition interrupt within the same superstep. LangGraph collects these and surfaces them **together**, resumed as one batch (a resume map keyed by interrupt id). So although the model is "block per blocked fetch," in practice the user is presented with the set of blocked downloads outstanding at that superstep and answers them in one pass — not necessarily one prompt per file across the whole run.

### Failure handling

Subagent failures do not abort the run. Transient errors (network, Tavily rate-limit, remote Ollama blip) get **bounded retries with backoff**; hard failures (e.g. marker crashing on a PDF) or exhausted retries cause that sub-topic to be **isolated** — marked failed/empty with a reason — while siblings continue. The writer and evaluator surface isolated sub-topics as coverage gaps and retry candidates for the next round, and the checkpointer allows resuming just that sub-topic.

A **blocked PDF fetch is not a failure** in this sense: rather than isolate the sub-topic, the subagent raises an **acquisition `interrupt()`** for a manual download (see [RAG & sources](#rag--sources) and the autonomy note above). Only if the user marks the source unobtainable does it become a recorded coverage gap.

## Brief generation

The `generate_brief` step (split into `clarify` → `decompose` → `approve` nodes) is the single interactive entry point. It turns a raw research question into an approved, structured brief that drives the rest of the graph.

### Flow

```text
super-topic in → clarify (one adaptive round) → decompose into sub-topics
  → present brief → interrupt(user) → approve | revise ↺ → persist brief
```

1. **Input** — the user supplies the super-topic (research question) via the CLI.
2. **Clarify** — before decomposing, the agent asks one short, adaptive round of clarifying questions (scope, audience, depth, time range, explicit exclusions). The round is skipped when the question is already specific enough.
3. **Decompose** — the LLM proposes N sub-topics, guided toward a quality target of **~3–7** and bounded by a configurable **hard ceiling (~12)** as a safety net against runaway decomposition (the user can still adjust the count at approval; the concurrency cap throttles actual host load). Each sub-topic is the subagent's *mandate* and is a typed object:
   - `slug` — the `<sub-topic>` folder name; slugified and stable.
   - `title`
   - `scope` — what the sub-topic covers and why (rationale).
   - `guiding_questions` — concrete questions the subagent must answer.
   - `seed_queries` *(optional)* — initial Tavily searches.
4. **Approve** — via `interrupt()`: the proposed brief is shown, the user replies with free-form natural-language feedback, the agent regenerates, and the loop repeats until the user approves. Feedback can add/remove/merge/reword sub-topics, change their count, or adjust scope.
5. **Persist** — the super-topic **slug** is derived **at run start** from the raw question (slugified title + short uniqueness hash) and is the `thread_id` from the first step, so the clarify and approval interrupts checkpoint under it (the slug is opaque and never changes even if the question is reworded). On approval the brief is written both into graph state (it drives the `supervisor` fan-out via `Send`) and as a human-readable `research/<super-topic-slug>/brief.md` (in the configurable outputs tree), whose frontmatter records the verbatim question, the slug, and the `thread_id`.

The brief is therefore a typed object — super-topic (verbatim question + stable slug) plus a list of structured sub-topics — that the supervisor maps over directly.

## RAG & sources

The local RAG is the mechanism for reusing sources already gathered, so a subagent never re-fetches what is already known. The RAG is a **candidate finder**, not the source of truth: chunk hits only point to documents worth inspecting; the actual evidence and citations are grounded against **full documents** that have passed an explicit relevance gate.

### Acquisition flow (sources → Bibliography → RAG)

```text
subagent needs info
  → RAG query → candidate source documents (chunk hits → dedup to source docs)
  → for each candidate: fetch FULL md → LLM relevance check
        → relevant? yes → whitelist (usable as evidence + citable)
                    no  → discard
  → still a gap? → Tavily search
        → result is a .pdf? → fetch via httpx
              ok      → pdf converter (pymupdf4llm/marker/remote) → md
              blocked → interrupt(user): "save <url> as _inbox/<id>.pdf"
                        → resume → auto-detect file presence → convert → md
          else            → extract (HTML→md) → heuristic pre-filter (min words, max link density)
        → dedup → save to central source pool → ingest (chunk → embed → upsert)
        → LLM source quality gate (is this real content or garbage?)
              fail → remove from pool immediately; skip to next result
              pass → new doc re-enters the relevance check before it can be used
```

User-supplied PDFs may also be dropped into the Bibliography inbox (`Bibliography/_inbox/`); on the next reconcile they are converted, deduped, and folded into the central pool just like web-found PDFs.

- **RAG-before-web** — each subagent queries the local index *before* searching the web, so siblings' findings and prior runs are reused (saves Tavily calls and tokens).
- **Candidate → full-document gate (whitelist)** — RAG chunk hits are deduped to their source documents; the gate then runs each candidate's **full markdown** through a large-context chat model and judges its relevance to the sub-topic. In the same pass it **distills the relevant points as verbatim quotes** (the *distilled evidence*), so later steps need not re-load the full body — a context-engineering compression (see [Architecture §10](Architecture.md#10-context-engineering)); final groundedness still re-checks claims against the full source. Documents larger than a generous size cap are truncated with a logged warning (treated as a rare edge case). Only documents that pass are **whitelisted**, and only whitelisted documents may be used as evidence or cited. Chunks are the discovery signal, never the grounding — this avoids chunk-out-of-context errors.
- **Relevance caching** — the whitelist/discard verdict is cached per `(super-topic, sub-topic, source)`, fully qualified by super-topic slug so a verdict never leaks across runs that share a sub-topic slug. A full document is therefore not re-read on every iteration; newly fetched web/PDF sources enter the same gate before use.
- **Web sources** — for each chosen search result, **Tavily `extract`** fetches the page and converts HTML→markdown; it is saved once to the central pool as `Bibliography/_sources/<id>.md` (`id` derived from the URL) with frontmatter (`url`, `title`, `retrieved_at`). This is the standard path for every web page: search to discover URLs, `extract` to turn each into a stored markdown source.
- **PDFs** — `.pdf` results found during Tavily search are fetched directly with **httpx** to `Bibliography/_sources/pdfs/<id>.pdf` (`id` = `hash(url)`, so it is nameable before download; URL-less drop-ins use `hash(bytes)`), then converted to markdown by the configured PDF converter (pymupdf4llm by default) and stored in the pool under the same id.
- **Source quality gate** — newly fetched web/PDF sources pass through two quality checks before entering the relevance gate. First, a **fast heuristic pre-filter**: pages below a minimum word count or above a maximum link density are discarded immediately. Second, an **LLM quality judge** (`source_quality.assess()`): a topic-agnostic model decides whether the content is substantive research material or garbage (navigation page, link farm, cookie wall). Sources that fail either check are deleted from the pool immediately. RAG-retrieved existing sources skip this check during research runs; the `prune` command handles retroactive cleanup of the pool.
- **Blocked fetches (paywalls / bot walls)** — many publishers (IEEE, ResearchGate, Elsevier, login walls) return 403/redirects so httpx cannot retrieve the PDF. When a fetch is blocked, the subagent **blocks via `interrupt()`** and asks the user to download it manually, showing the **URL, title, and the exact path to save to** (`Bibliography/_inbox/<id>.pdf`, `id` derived from the source URL). The CLI uses **auto-detection** on resume: after the user saves the files and presses Enter, the CLI checks each expected path and classifies automatically:
  1. **Saved** (file present) — runs the PDF converter → pool → relevance gate under the already-known source id (deterministic match, no guessing).
  2. **Unobtainable** (file absent) — the source is recorded as a **permanent coverage gap** and the run continues without it (the gap surfaces in the evaluator). Once declared unobtainable, the source is not re-requested on subsequent acquire iterations.
  3. **Alternative URL** (model layer only, not surfaced by the current CLI) — the user provides an open-access URL; the agent fetches that instead.

  This is an **acquisition** block, not an editorial decision — see the autonomy note above.
- **Dedup & source identity** — a source's **identity is its URL when it has one** (web pages and PDFs-by-URL): the id is `hash(url)`, and a same-URL re-fetch with changed content **updates the existing doc in place** (overwrite markdown, bump `retrieved_at`, re-embed), keeping one logical reference and one citation key. The `content_hash` is stored to *detect* whether a re-fetch actually changed; only **URL-less drop-in files** are identified by `hash(bytes)`. Dedup runs on the id before saving and before embedding — nothing is stored or embedded twice.

### Indexing

- **Incremental upsert on save** — every newly saved markdown file is chunked, embedded, and upserted immediately, so the next iteration can retrieve it. To stay race-free under the parallel fan-out, **all writes go through a single serialized writer** (a process-wide lock/queue over one Chroma client; reads stay concurrent), and chunk ids are **deterministic** (`source_id` + chunk index) so the upsert is idempotent — two subagents finding the same source can't duplicate it. There is no runtime full rebuild; the only full reindex is the deliberate, offline embedding-model change.
- **Reconcile on startup / resume** — runs once **before** the supervisor fan-out (so it never overlaps subagent writes), and standalone via the `research sync` CLI verb. It first processes anything in the Bibliography inbox (`_inbox/`: convert PDFs via the configured converter, dedup, fold into the pool), then indexes any Bibliography markdown that is either new or whose `content_hash` has changed since it was last indexed (so a re-fetched web source with updated content is re-embedded, not silently stale). Recovers from interrupted runs and picks up manually-downloaded sources, consistent with the LangGraph checkpointer.
- **Chunking** — markdown-header-aware splitting, then size-based sub-splitting (~1000 chars, ~200 overlap); the header path is kept as metadata.
- **Chunk metadata** — `source_id` (pool identity / citation key), `content_hash`, `source_path`, `source_url`, `title`, `type` (web/pdf), plus `super_topic`/`sub_topic` of first discovery (provenance only). Citation mapping keys on `source_id`, not on the discovering topic, since a pooled source can be referenced by many runs.

### Vector store & embeddings

- **Vector store** — **Chroma** (embedded, persistent, metadata filtering), persisted under `.deepresearch/`.
- **Embeddings** — served by the **remote Ollama server** (same host as the chat models, configurable base URL). Default model `mxbai-embed-large`, chosen for quality; configurable.
- **Retrieval scope** — **global**: a single shared Chroma collection spans all research questions, maximising reuse across runs. Bleed-in is contained in three steps: retrieve **top-k**, drop anything below an **absolute similarity floor**, then **re-rank with a mild provenance boost** for chunks whose `super_topic` matches the current run — so same-topic sources rank first while cross-topic reuse stays eligible (a re-rank, not a filter). The full-document relevance gate remains the real correctness filter; the floor just trims obvious junk before expensive gating. `super_topic` / `sub_topic` remain on each chunk as discovery provenance even though retrieval does not hard-filter on them.
- **Embedding-model consistency** — because the collection is global, the embedding model must stay consistent across runs; changing it requires a full reindex.

## Citations & output

Only **whitelisted** source documents (see RAG & sources) may be cited. Everything in the reports traces back to one of them.

### Citations

- **Citation identity** — each whitelisted document has a **stable citation key** = its source id: `hash(url)` when a URL exists (web pages and PDFs-by-URL), `hash(bytes)` for URL-less drop-ins. The key is independent of where it appears, so the same source cited by different subagents is one logical reference, and a re-fetch of a changed web page keeps the same key (its content is updated in place).
- **Style** — inline numeric markers `[n]` with a numbered References list. Compact, machine-checkable, and straightforward to renumber on merge.
- **Reference entry** — resolves to `title`, `source_url`, local `source_path`, `type` (web/pdf), and `retrieved_at`.
- **Merging into the final report** — the writer agent receives the per-sub-topic reports (each using stable keys), merges and **de-duplicates** references into one unified bibliography, then **renumbers** `[n]` globally. Identity-based keys make the same source collapse to a single entry across subagents.

### Verification (the "no uncited claims" rule)

A validation pass enforces the brief's hard rule, in two stages:

1. **Resolution (programmatic)** — every claim carries a citation, and every citation resolves to a whitelisted source file; no dangling references.
2. **Groundedness (LLM)** — each cited claim is checked against its source document to confirm the source actually supports the claim.

**Fail-loud.** Verification is a hard gate, not an annotation, and runs **inside the writer step**. A claim that fails resolution or groundedness is either dropped or revised in place by the writer (a bounded revision pass using the full source) — no subagent re-run; the report is not finalized while any claim is unresolved or ungrounded.

### Output (writer agent)

- **Synthesis, not concatenation** — the writer reconciles overlaps and contradictions across sub-topics into one coherent narrative.
- **Structure** — title → executive summary → per-sub-topic sections → conclusion → unified References.
- **Location** — in the configurable outputs tree: final report at `research/<super-topic-slug>/report.md`; per-sub-topic reports at `research/<super-topic-slug>/<sub-topic-slug>/report.md`. The Bibliography holds only sources, never reports.
- **Who writes what** — each subagent writes its own per-sub-topic `report.md` on emit; the writer persists the unified `report.md` plus `references.json` (the deduped source-id list); the brief-generation step (`clarify`/`decompose`/`approve` nodes) persists `brief.md`.

## Evaluation & refinement loop

After the writer produces the report, an `evaluate` node closes the loop back to the brief, turning the pipeline into an iterative system. The loop has **two tiers**: an *autonomous* refinement tier that improves the report before involving anyone, then a *human-in-the-loop* tier. LangGraph's conditional edges make both cycles native.

```text
writer → evaluate (report + sources ⇄ brief: coverage & support per guiding_question)
   │
   ├─ AUTONOMOUS round   [gaps AND auto_round < cap AND coverage improving]
   │     → auto-update brief (deepen existing mandates; may add sub-topics)
   │     → supervisor re-run (only NEW/CHANGED sub-topics) ↺      [no user]
   │
   └─ HAND OFF TO USER   [fully covered | cap hit | coverage plateaued]
         → full consistency re-run (ALL sub-topics) → writer
         → interrupt(user): best report + coverage summary + queued additions
            ├─ accepts / edits / adds → update brief → user-facing re-run ↺
            └─ approves / declares done → END
```

- **Evaluator agent** — takes the brief (each sub-topic's `guiding_questions`), the final report, **and the whitelisted sources**. For each question it judges both **coverage** (is it addressed?) and **support** (is the answer actually backed by sources?), scoring **answered / partial / unanswered** and flagging thin or single-source answers — so a confident-but-unsupported report does not pass. It produces a gap analysis plus proposed **follow-ups** (new sub-topics or refinements to existing ones).
- **Autonomous refinement (before the user)** — while gaps remain, the auto-round count is under a configurable **cap** (e.g. 2), *and* coverage still improved over the previous round (**plateau detection**), the loop iterates without interrupting: it auto-applies follow-ups and re-runs. This front-loads quality so the user's first look is at an already-improved report. The loop stops and hands off when the report is fully covered, the cap is hit, or coverage stops improving.
- **Hand-off to user** — via `interrupt()` (the same primitive as brief generation): the user sees the best report, the coverage assessment, and any **queued additions**, and can accept, edit, add their own, or declare the research done.
- **Brief update** — accepted follow-ups are appended/merged into the brief as new or refined sub-topics — an *incremental* update to the brief, not a regeneration. **Autonomous rounds may deepen existing mandates and add new sub-topics; both apply, but to bound cost the new ones drive the cheaper re-run mode below.**
- **Two-mode re-run** — the supervisor re-dispatch differs by tier:
  - *Autonomous rounds* fan out over **only new/changed sub-topics** (incremental), keeping cost low while iterating unattended.
  - *User-facing rounds* (the hand-off and any subsequent user-driven round) fan out over **all** sub-topics — a full **consistency re-run** so that every per-sub-topic report and the final report are mutually consistent at exactly the moment a human reads them (a new sub-topic can surface evidence that updates or contradicts an earlier one). Cost is contained either way because re-runs reuse the cached whitelist verdicts, the central source pool, and the global RAG from prior rounds — an unchanged sub-topic mostly re-reads known sources rather than re-fetching them.
- **Report handling** — the report is **overwritten** each round (single `report.md`); run state across rounds is recorded by the LangGraph checkpointer.
- **Termination** — two caps apply: the **auto-round cap** bounds the autonomous tier (after which it hands off to the user), and an overall **max-rounds** cap bounds the whole loop. The loop ends when the evaluator reports full coverage *and* the user approves, the user explicitly stops, or max-rounds is hit.

## Tech stack

- **Python** — implementation language.
- **uv** — Python packaging, dependency, and environment management.
- **LangGraph** (on LangChain) — agent orchestration, state, checkpointing, and fan-out.
- **Ollama** — LLM backbone (chat + embeddings), served from a remote server (configurable base URL; optional API key for authenticated/cloud hosts). Chat models are **role-tiered** (all configurable): a small/fast model for clarify & routing, a **large-context** model for the relevance gate and synthesis, and a strong model for the final writer and evaluator. Embeddings use `mxbai-embed-large` by default.
- **Chroma** — embedded, persistent vector store for the RAG.
- **Tavily** — web search and HTML→markdown extraction.
- **httpx** — direct PDF downloads (paywalled/bot-walled fetches fall back to a manual-download interrupt).
- **pymupdf4llm** — default PDF→markdown converter (fast, no extra dependencies). **marker** (OCR + LLM refinement) is available as an optional extra for higher-quality conversion; a remote converter service is also supported.
- A **local RAG** over the Bibliography markdown for reusing already-gathered sources.
