# Context — Glossary

The ubiquitous language for the Deep Research framework. Definitions only; no implementation details.

## Super-topic

The overall research question a run answers (e.g. *"Long-term effects of intermittent fasting on cardiovascular health"*). The user supplies it verbatim. It is decomposed into **sub-topics**.

Identity: each super-topic has a **stable slug** derived **at run start** from the raw question (slugified title + short uniqueness hash). It is the `thread_id` from the first step, the durable key for everything on disk and for run state; it never changes even if the question is later reworded. The verbatim question, slug, and `thread_id` are recorded in the super-topic's `brief.md`.

## Sub-topic

One bounded facet of the super-topic, and the **mandate** for exactly one research subagent. A typed object: `slug`, `title`, `scope`, `guiding_questions`, optional `seed_queries`. Its `slug` is a stable folder name and is never renamed.

## Run / thread

A single execution of the graph for one super-topic. Identified by `thread_id`, which equals the super-topic slug (one identity). Re-running the **same** super-topic resumes the existing thread (loads the checkpoint, reuses the Bibliography) rather than starting over.

## Brief

The approved, structured output of brief generation: the super-topic plus its list of sub-topics. The contract the rest of the graph executes against.

## Bibliography

The durable, on-disk **source store only** (no deliverables), persisted across runs. Every gathered source document is saved exactly once in a central pool, independent of which super-topic discovered it. Each source has a **stable id** that is also its citation key: `hash(url)` whenever a URL exists (web pages and PDFs reached by URL — re-fetching a changed page updates it in place under the same id), and `hash(bytes)` only for URL-less drop-in files. A `content_hash` is always kept separately for dedup/change-detection. This matches the global vector store: one logical copy of a source, reusable by any run. Root is configurable (`bibliography_dir`).

## Outputs

The human deliverables, kept in a separate configurable tree (`output_dir`, default `research/`), distinct from the Bibliography. Per super-topic slug: `brief.md`, the final `report.md`, per-sub-topic reports, and the list of source ids referenced. Separating outputs from the source library lets reports be committed while the bulky source library lives elsewhere.

## Quality gate

A subagent's structured exit criterion: it scores its own draft report against its `guiding_questions` for **coverage** and a **cheap groundedness self-check against its distilled evidence** (does each claim have a backing quote?). Authoritative full-document grounding is not repeated here — it runs once, globally, at final verification. Passing lets it emit the report; failing loops it back to gather more (bounded by a max-iteration cap, after which it emits with the shortfall flagged). It is the only per-sub-topic check; the **evaluator** handles cross-topic coverage.

## Refinement loop

The post-writer cycle, in two tiers. **Autonomous tier**: improves the report without human *judgment* (bounded by an auto-round cap, plateau detection, and full-coverage) and re-runs only new/changed sub-topics. **Human tier**: hands off to the user via interrupt and does a full consistency re-run of all sub-topics so everything is mutually consistent at the moment a human reads it. Note: autonomous rounds may still pause for an **acquisition interrupt** (a manual download) — that is mechanical, not editorial.

## Editorial vs acquisition interrupt

Two kinds of human pause. **Editorial** interrupts ask for human judgment — clarify, brief approval, evaluate follow-ups — and define the human-in-the-loop tier. **Acquisition** interrupts are mechanical necessities the agent cannot perform itself — chiefly a blocked PDF fetch (paywall/login wall) that the user must download manually. Acquisition interrupts may fire in any round, autonomous included; they carry no editorial decision. The manual-download prompt offers: save-at-named-path, mark-unobtainable (permanent gap), or supply an alternative open-access URL.

## Distilled evidence

The sub-topic-relevant points of a whitelisted document, extracted as verbatim quotes by the relevance gate at whitelist time and cached with the verdict. It is the compression unit the subagent loop and the writer carry **instead of** full document bodies (a context-engineering measure). It speeds drafting but does not replace grounding: final verification re-checks each cited claim against the **full** source.

## Scratchpad

A subagent's running working notes (findings so far, tried queries, open questions), held in its state and checkpointed. Lets the research loop persist progress without depending on raw LLM message history.

## Whitelist

The set of source documents that have passed the full-document relevance gate for a given sub-topic. Only whitelisted documents may be used as evidence or cited. (Distinct from RAG chunk hits, which are only discovery signals.)

The relevance verdict is cached per `(super-topic, sub-topic, source)` — fully qualified by super-topic slug so a verdict formed in one research context never leaks to another that happens to share a sub-topic slug.
