"""Typer CLI entry point."""

from pathlib import Path

import typer

app = typer.Typer()


@app.callback()
def main(log_level: str = typer.Option("INFO", "--log-level", help="Logging level.")) -> None:
    """Configure process-wide CLI behavior."""
    from deepresearch.config import Config
    from deepresearch.endpoints import resolve_ollama_endpoints
    from deepresearch.logging_setup import configure_logging

    # Search for .env relative to this file (project root), not cwd, so the
    # CLI works regardless of where it is invoked from on a remote machine.
    _env_file = Path(__file__).parent.parent.parent / ".env"
    cfg = Config.load_env(_env_file if _env_file.exists() else ".env")
    configure_logging(log_level)
    # Fall back to a local Ollama when the configured hosts are unreachable
    # (e.g. a LAN host that is not on the current network).
    resolve_ollama_endpoints(cfg)


def _build_configurable(cfg):
    """Build the configurable dict for graph invocation."""
    from langchain_ollama import OllamaEmbeddings as LCOllamaEmbeddings
    from tavily import TavilyClient

    from deepresearch.llm import chat
    from deepresearch.rag.store import ChromaStore
    from deepresearch.sources.pdf import HttpxPdfClient

    embeddings_kwargs = {"model": cfg.embed_model, "base_url": cfg.embed_base_url}
    if cfg.ollama_api_key:
        embeddings_kwargs["client_kwargs"] = {
            "headers": {"Authorization": f"Bearer {cfg.ollama_api_key}"}
        }
    embeddings = LCOllamaEmbeddings(**embeddings_kwargs)
    store = ChromaStore(cfg.state_dir, embeddings.embed_query)

    return {
        "chat_fn": chat,
        "embeddings": embeddings,
        "store": store,
        # Without these the subagent gates off all web/PDF acquisition and runs
        # RAG-only, which yields empty, uncited reports on a fresh Bibliography.
        "tavily_client": TavilyClient(api_key=cfg.tavily_api_key),
        "pdf_client": HttpxPdfClient(),
        "bibliography_dir": str(cfg.bibliography_dir),
        "state_dir": str(cfg.state_dir),
        "output_dir": str(cfg.output_dir),
    }


def _do_sync(cfg):
    """Reconcile inbox and index un-indexed sources."""
    from langchain_ollama import OllamaEmbeddings as LCOllamaEmbeddings

    from deepresearch.rag.index import reconcile
    from deepresearch.rag.store import ChromaStore

    embeddings_kwargs = {"model": cfg.embed_model, "base_url": cfg.embed_base_url}
    if cfg.ollama_api_key:
        embeddings_kwargs["client_kwargs"] = {
            "headers": {"Authorization": f"Bearer {cfg.ollama_api_key}"}
        }
    embeddings = LCOllamaEmbeddings(**embeddings_kwargs)
    store = ChromaStore(cfg.state_dir, embeddings.embed_query)
    reconcile(cfg.bibliography_dir, store, embeddings)


def _display_interrupt(data):
    """Render an interrupt payload for the user."""
    interrupt_type = data.get("type", "unknown")
    typer.echo(f"\n--- Interrupt: {interrupt_type} ---")
    if interrupt_type == "clarify":
        typer.echo(f"Question: {data.get('question', '')}")
        typer.echo("Clarifying questions:")
        for q in data.get("questions", []):
            typer.echo(f"  - {q}")
    elif interrupt_type == "approve":
        brief = data.get("brief", {})
        typer.echo(f"Question: {brief.get('question', '')}")
        typer.echo(f"Sub-topics ({len(brief.get('subtopics', []))}):")
        for st in brief.get("subtopics", []):
            typer.echo(f"  - {st.get('title', '')}: {st.get('scope', '')}")
        typer.echo("\nApprove, or provide feedback to revise.")
    elif interrupt_type == "acquire":
        requests = data.get("requests", [])
        typer.echo(f"{len(requests)} blocked source(s) need a manual download:")
        for idx, request in enumerate(requests, start=1):
            typer.echo(f"  [{idx}] {request.get('title', '')}  {request.get('url', '')}")
            typer.echo(f"      save as: {request.get('save_path', '')}")
        typer.echo("\nDownload each file and save it to the path shown above, then press Enter.")
    else:
        typer.echo(str(data))


def _collect_acquire_responses(requests: list[dict]) -> list[dict]:
    """Wait for user confirmation, then auto-detect which inbox files are present."""
    typer.prompt("\nPress Enter when ready", default="", show_default=False)
    responses = []
    found = []
    missing = []
    for req in requests:
        if Path(req["save_path"]).exists():
            responses.append({"source_id": req["source_id"], "kind": "saved", "save_path": req["save_path"]})
            found.append(req.get("title") or req["source_id"])
        else:
            responses.append({"source_id": req["source_id"], "kind": "unobtainable"})
            missing.append(req.get("title") or req["source_id"])
    if found:
        typer.echo(f"  Found ({len(found)}): {', '.join(found)}")
    if missing:
        typer.echo(f"  Not found — skipping ({len(missing)}): {', '.join(missing)}")
    return responses


def _collect_resume_input(current_state):
    """Display the pending interrupt and collect user input."""
    interrupts = current_state.interrupts if hasattr(current_state, "interrupts") else []
    for interrupt_data in interrupts:
        _display_interrupt(interrupt_data.value)
    if interrupts and interrupts[0].value.get("type") == "acquire":
        return _collect_acquire_responses(interrupts[0].value.get("requests", []))
    return typer.prompt("\nYour response")


def _run_interactive(graph, input_state, config):
    """Run the graph, handling interrupts interactively."""
    from langgraph.types import Command

    current_input = input_state
    while True:
        result = graph.invoke(current_input, config)
        interrupts = result.get("__interrupt__", [])
        if not interrupts:
            typer.echo("\nResearch run complete.")
            typer.echo(f"Output: {config['configurable']['output_dir']}/{config['thread_id']}/")
            return result
        for interrupt_data in interrupts:
            _display_interrupt(interrupt_data.value)
        if interrupts and interrupts[0].value.get("type") == "acquire":
            user_input = _collect_acquire_responses(interrupts[0].value.get("requests", []))
        else:
            user_input = typer.prompt("\nYour response")
        current_input = Command(resume=user_input)


@app.command()
def run(question: str) -> None:
    """Start a new research run."""
    from deepresearch.config import get_config
    from deepresearch.endpoints import check_embed_model
    from deepresearch.graph import build_graph
    from deepresearch.paths import slug
    from deepresearch.persistence import create_checkpointer
    from deepresearch.state import ResearchState

    cfg = get_config()
    check_embed_model(cfg)
    run_slug = slug(question)

    # Reconcile before fan-out
    _do_sync(cfg)

    # Build graph with checkpointer
    with create_checkpointer(cfg.state_dir) as checkpointer:
        graph = build_graph(checkpointer=checkpointer)

        # Initial state
        initial_state: ResearchState = {
            "question": question,
            "slug": run_slug,
            "brief": None,
            "round": 0,
            "auto_round": 0,
            "mode": "user_facing",
            "pending_handoff": False,
            "subreports": {},
            "report": None,
            "report_references": [],
            "coverage": None,
            "history": [],
            "verify_ok": False,
            "verify_attempts": 0,
            "verify_unsupported": [],
            "verify_dangling": [],
            "user_approved": False,
        }

        config = {
            "configurable": _build_configurable(cfg),
            "thread_id": run_slug,
        }

        _run_interactive(graph, initial_state, config)


@app.command()
def resume(slug: str) -> None:
    """Resume an interrupted research run."""
    from langgraph.types import Command

    from deepresearch.config import get_config
    from deepresearch.endpoints import check_embed_model
    from deepresearch.graph import build_graph
    from deepresearch.persistence import create_checkpointer

    cfg = get_config()
    check_embed_model(cfg)

    # Reconcile on resume
    _do_sync(cfg)

    with create_checkpointer(cfg.state_dir) as checkpointer:
        graph = build_graph(checkpointer=checkpointer)

        config = {
            "configurable": _build_configurable(cfg),
            "thread_id": slug,
        }

        # Check if a run exists
        current_state = graph.get_state(config)
        if current_state is None or current_state.values == {}:
            typer.echo(f"No run found for slug: {slug}")
            raise typer.Exit(1)

        # Collect resume input and continue
        resume_value = _collect_resume_input(current_state)
        _run_interactive(graph, Command(resume=resume_value), config)


@app.command()
def sync() -> None:
    """Reconcile the Bibliography inbox and index un-indexed sources."""
    from deepresearch.config import get_config

    _do_sync(get_config())
    typer.echo("Sync complete.")


@app.command(name="list")
def list_runs() -> None:
    """List research runs."""
    from deepresearch.config import get_config
    from deepresearch.run_state import load_runs

    cfg = get_config()
    runs = load_runs(cfg.state_dir, cfg.output_dir)
    if not runs:
        typer.echo("No runs found.")
        return
    for run in runs:
        typer.echo(
            f"{run.slug}  {run.question}  round={run.round}  auto_round={run.auto_round}  "
            f"brief={'yes' if run.has_brief else 'no'}  "
            f"report={'yes' if run.has_report else 'no'}  "
            f"pending_handoff={'yes' if run.pending_handoff else 'no'}"
        )


@app.command()
def status(slug: str) -> None:
    """Show status of a research run."""
    from deepresearch.config import get_config
    from deepresearch.run_state import load_run

    cfg = get_config()
    summary = load_run(cfg.state_dir, cfg.output_dir, slug)
    if summary is None:
        typer.echo(f"No run found for slug: {slug}")
        raise typer.Exit(1)
    for key, value in summary.model_dump().items():
        typer.echo(f"{key}: {value}")
    if summary.has_report:
        report_path = cfg.output_dir / summary.slug / "report.md"
        typer.echo("\nPreview:")
        lines = report_path.read_text(encoding="utf-8").splitlines()[:20]
        for line in lines:
            typer.echo(line)


if __name__ == "__main__":
    app()
