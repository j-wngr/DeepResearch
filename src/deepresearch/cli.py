"""Typer CLI entry point."""

import typer

app = typer.Typer()


def _build_configurable(cfg):
    """Build the configurable dict for graph invocation."""
    from langchain_ollama import OllamaEmbeddings as LCOllamaEmbeddings

    from deepresearch.llm import chat
    from deepresearch.rag.store import ChromaStore

    embeddings = LCOllamaEmbeddings(
        model=cfg.embed_model,
        base_url=cfg.embed_base_url,
    )
    store = ChromaStore(cfg.state_dir, embeddings.embed_query)

    return {
        "chat_fn": chat,
        "embeddings": embeddings,
        "store": store,
        "bibliography_dir": str(cfg.bibliography_dir),
        "state_dir": str(cfg.state_dir),
        "output_dir": str(cfg.output_dir),
    }


def _do_sync(cfg):
    """Reconcile inbox and index un-indexed sources."""
    from langchain_ollama import OllamaEmbeddings as LCOllamaEmbeddings

    from deepresearch.rag.index import reconcile
    from deepresearch.rag.store import ChromaStore

    embeddings = LCOllamaEmbeddings(
        model=cfg.embed_model,
        base_url=cfg.embed_base_url,
    )
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
    else:
        typer.echo(str(data))


def _collect_resume_input(current_state):
    """Display the pending interrupt and collect user input."""
    interrupts = current_state.interrupts if hasattr(current_state, "interrupts") else []
    for interrupt_data in interrupts:
        _display_interrupt(interrupt_data.value)
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
        user_input = typer.prompt("\nYour response")
        current_input = Command(resume=user_input)


@app.command()
def run(question: str) -> None:
    """Start a new research run."""
    from deepresearch.config import get_config
    from deepresearch.graph import build_graph
    from deepresearch.paths import slug
    from deepresearch.persistence import create_checkpointer
    from deepresearch.state import ResearchState

    cfg = get_config()
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
    from deepresearch.graph import build_graph
    from deepresearch.persistence import create_checkpointer

    cfg = get_config()

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
    typer.echo("list (not yet implemented)")


@app.command()
def status(slug: str) -> None:
    """Show status of a research run."""
    typer.echo(f"status: {slug} (not yet implemented)")


if __name__ == "__main__":
    app()
