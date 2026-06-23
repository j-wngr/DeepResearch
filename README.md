# DeepResearch

A multi-agent deep research framework that decomposes a research question into sub-topics, gathers and cites web and local sources per sub-topic, and synthesizes the findings into one unified, fully-cited research report.

Inspired by [LangChain's Open Deep Research](https://www.langchain.com/blog/open-deep-research), but with a focus on stable run identity, a shared source bibliography, and a human-in-the-loop approval/refinement cycle.

## How it works

1. You provide a research question.
2. The agent asks a short round of clarifying questions, then proposes a breakdown into sub-topics.
3. You approve or revise the research brief.
4. Sub-topic research agents run in parallel, searching the web (Tavily) and a local RAG over previously gathered sources.
5. A writer agent synthesizes all findings into one cited report.
6. An evaluator scores coverage and loops autonomously (up to a configurable cap) before handing off to you for a final approval.

All sources are stored in a central Bibliography; reports land in `research/<slug>/`.

## Prerequisites

- **Python ≥ 3.11**
- **[uv](https://docs.astral.sh/uv/)** — used for everything (install, run, deps)
- **[Ollama](https://ollama.com/)** — LLM backend (chat + embeddings); can be a remote server
- **[Tavily](https://tavily.com/)** API key — for web search and HTML extraction

## Installation

```bash
git clone https://github.com/your-org/DeepResearch.git
cd DeepResearch
uv sync
```

## Configuration

Copy `.env.example` to `.env` and fill in your values:

```bash
cp .env.example .env
```

Key settings:

| Variable | Description |
|---|---|
| `OLLAMA_BASE_URL` | Base URL of your Ollama server (chat models) |
| `EMBED_BASE_URL` | Base URL of your Ollama server (embeddings) |
| `OLLAMA_API_KEY` | Optional — for authenticated/cloud Ollama hosts |
| `MODEL_FAST` | Small/fast model (clarify, routing) |
| `MODEL_LONG` | Large-context model (relevance gate, synthesis) |
| `MODEL_WRITER` | Strong model (final writer, evaluator) |
| `EMBED_MODEL` | Embedding model (default: `mxbai-embed-large`) |
| `TAVILY_API_KEY` | Tavily API key |
| `BIBLIOGRAPHY_DIR` | Where gathered sources are stored (default: `./Bibliography`) |
| `OUTPUT_DIR` | Where briefs and reports are written (default: `./research`) |
| `MAX_CONCURRENCY` | Parallel sub-topic agents (tune to your Ollama server) |

## Usage

All commands are run via `uv run deepresearch` (or `uv run python -m deepresearch.cli`).

### Start a new research run

```bash
uv run deepresearch run "What are the key challenges in large-scale distributed database systems?"
```

The tool will:
- Ask clarifying questions (one short round)
- Propose sub-topics — you approve or give feedback to revise
- Run sub-topic research agents in parallel
- Write and evaluate the report, looping autonomously to fill gaps
- Present the final report for your approval

The report is saved to `research/<slug>/report.md`.

### Resume an interrupted run

Runs are checkpointed after every step. If a run is interrupted (process killed, paywalled PDF that needs a manual download, etc.), resume it by slug:

```bash
uv run deepresearch resume <slug>
```

The slug is printed when you start a run and is also the folder name under `research/`.

### List all runs

```bash
uv run deepresearch list
```

### Check the status of a run

```bash
uv run deepresearch status <slug>
```

Shows round number, whether a brief/report exists, and a preview of the report if one has been written.

### Sync the Bibliography

Indexes any un-indexed sources and processes PDFs dropped into `Bibliography/_inbox/`:

```bash
uv run deepresearch sync
```

This also runs automatically at the start of every `run` and `resume`.

## Dropping in your own PDFs

Place PDFs in `Bibliography/_inbox/`. They are picked up, converted to markdown via [marker](https://github.com/VikParuchuri/marker), deduplicated, and indexed on the next `sync` (or at the start of the next run).

## Paywalled sources

If a subagent encounters a PDF it cannot fetch (paywall, login wall), it will pause and ask you to download it manually, giving you the exact path to save it to (`Bibliography/_inbox/<id>.pdf`). Save the file, then run `deepresearch resume <slug>` to continue.

## Output structure

```
research/
  <super-topic-slug>/
    brief.md          # approved research brief
    report.md         # final unified report
    references.json   # deduped source list
    <sub-topic-slug>/
      report.md       # per-sub-topic report

Bibliography/
  _sources/
    <id>.md           # gathered web/PDF sources (shared across all runs)
    pdfs/
      <id>.pdf
  _inbox/             # drop your own PDFs here
```

## Development

```bash
# Install dev dependencies
uv sync --group dev

# Run tests (hermetic — no real network or Ollama calls)
uv run pytest

# Lint / format
uv run ruff check src tests
uv run ruff format src tests
```

See [docs/Architecture.md](docs/Architecture.md) for the implementation design and [docs/DesignBrief.md](docs/DesignBrief.md) for the motivation and high-level approach.
