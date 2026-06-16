# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Intent

A **Deep Research framework**: it decomposes a research question (super-topic) into sub-topics, gathers and cites web/local sources per sub-topic, and synthesizes the findings into one unified research report. See [docs/DesignBrief.md](docs/DesignBrief.md) for the design (what/why), [docs/Architecture.md](docs/Architecture.md) for the implementation architecture (how), and [CONTEXT.md](CONTEXT.md) for the domain glossary.

## Tech stack

- **Python** — implementation language.
- **uv** — Python packaging, dependency, and environment management.
- **LangGraph** (on LangChain) — agent orchestration, state, checkpointing, and fan-out.
- **Ollama** — LLM backbone (chat + embeddings), served from a remote server (configurable base URL).
- **Chroma** — embedded, persistent vector store for the RAG.
- **Tavily** — web search and HTML→markdown extraction.
- **httpx** — direct PDF downloads; paywalled/bot-walled fetches fall back to a manual-download interrupt.
- **marker** — PDF→markdown conversion (OCR + LLM refinement).
- A **local RAG** over the Bibliography markdown for reusing already-gathered sources.

## Conventions

- **uv for everything** — run code and tools via `uv run ...` and manage deps with `uv add ...`; don't call `pip` or a global Python.
- **Bibliography is the source store only** — gathered sources live once in a central pool (`Bibliography/_sources/<id>.md`, PDFs in `_sources/pdfs/`), shared across runs; `id` = `hash(url)` when a URL exists (web pages and PDFs-by-URL), else content hash (URL-less drop-ins). User drop-ins land in `Bibliography/_inbox/`. No reports live here.
- **Deliverables live in the outputs tree** — briefs and reports go under a separate, configurable `output_dir` (default `research/<super-topic-slug>/`: `brief.md`, `report.md`, per-sub-topic reports, source-id references). Both roots are configurable (`output_dir`, `bibliography_dir`). Super-topic and sub-topic `slug`s are stable; don't rename them. The super-topic slug is also the run `thread_id`.
- **RAG is a candidate finder, not the grounding** — ground claims on the **full whitelisted document**, never on raw retrieved chunks.
- **Citations are mandatory and gated** — only **whitelisted** documents may be cited; inline numeric `[n]`; every claim carries a citation that resolves to a whitelisted source file.
- **Global vector store** — one shared Chroma collection across all runs, so the embedding model must stay consistent; changing it requires a full reindex.
- **Human-in-the-loop via LangGraph `interrupt()`** — used for brief approval and the evaluation/refinement loop; keep these as graph interrupts, not ad-hoc input prompts.
- **Local state lives in `.deepresearch/`** — LangGraph checkpointer + Chroma store; keep it git-ignored.
- **Config & secrets** — the Ollama base URL is configurable (remote server); the Tavily API key comes from the environment, never hard-coded.
