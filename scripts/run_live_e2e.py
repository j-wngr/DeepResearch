#!/usr/bin/env python3
"""Real-world end-to-end test for the DeepResearch framework.

This script runs the full research pipeline against real Ollama and Tavily
services, auto-approving all human-in-the-loop interrupts. It is intended for
manual smoke testing of a deployed environment, not for the hermetic test
suite.

Requires:
  - Ollama running and reachable (OLLAMA_BASE_URL / EMBED_BASE_URL)
  - Tavily API key exported as TAVILY_API_KEY
  - The configured chat and embedding models pulled in Ollama

Usage:
  uv run python scripts/run_live_e2e.py
  uv run python scripts/run_live_e2e.py "What are the health benefits of coffee?"
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
import time
import urllib.parse
from pathlib import Path

import httpx
from langchain_ollama import OllamaEmbeddings as LCOllamaEmbeddings
from langgraph.types import Command
from tavily import TavilyClient

from deepresearch.config import get_config, reset_config
from deepresearch.endpoints import resolve_ollama_endpoints
from deepresearch.graph import build_graph
from deepresearch.llm import chat
from deepresearch.models import AcquisitionResponse
from deepresearch.paths import slug
from deepresearch.persistence import create_checkpointer
from deepresearch.rag.index import reconcile
from deepresearch.rag.store import ChromaStore
from deepresearch.sources.pdf import HttpxPdfClient, is_login_wall
from deepresearch.state import ResearchState

DEFAULT_QUESTION = "What are the health benefits of green tea?"
MAX_ROUNDS = 20


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a real-world end-to-end DeepResearch pipeline."
    )
    parser.add_argument(
        "question",
        nargs="?",
        default=DEFAULT_QUESTION,
        help="Research question to run (defaults to a simple question).",
    )
    parser.add_argument(
        "--keep-output",
        action="store_true",
        help="Keep the temp output directory after the run (default behavior).",
    )
    parser.add_argument(
        "--cleanup",
        action="store_true",
        help="Delete the temp output directory after verification passes.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Use a specific output directory instead of a temp one.",
    )
    return parser.parse_args(argv)


def _check_ollama(base_url: str, model: str) -> tuple[bool, bool]:
    """Return (server_reachable, model_present).

    Models served by cloud endpoints or pulled lazily may not appear in
    /api/tags exactly as configured, so we only hard-fail on unreachable
    servers and warn when a model is not listed. Retries on transient
    network errors because LAN Ollama hosts can flake on first connect.
    """
    parsed = urllib.parse.urlparse(base_url)
    last_exc: Exception | None = None
    for attempt in range(3):
        try:
            resp = httpx.get(f"{parsed.scheme}://{parsed.netloc}/api/tags", timeout=30.0)
            resp.raise_for_status()
            models = {m.get("name") or m.get("model") for m in resp.json().get("models", [])}
            return True, model in models
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if attempt < 2:
                time.sleep(2.0 * (attempt + 1))
    print(f"WARNING: Could not reach Ollama server at {base_url}: {last_exc}")
    return False, False


def _validate_prereqs(cfg) -> list[str]:
    """Return a list of missing prerequisites."""
    errors: list[str] = []

    if not cfg.tavily_api_key:
        errors.append("TAVILY_API_KEY is not set in environment or .env")

    for purpose, base_url, model in [
        ("chat", cfg.ollama_base_url, cfg.model_fast),
        ("chat", cfg.ollama_base_url, cfg.model_long),
        ("chat", cfg.ollama_base_url, cfg.model_writer),
        ("embed", cfg.embed_base_url, cfg.embed_model),
    ]:
        reachable, present = _check_ollama(base_url, model)
        if not reachable:
            errors.append(f"Ollama server unreachable at {base_url} for {purpose} model '{model}'")
        elif not present:
            print(
                f"WARNING: Ollama {purpose} model '{model}' not listed at {base_url}; "
                "proceeding anyway (it may be a cloud alias or lazily loaded)."
            )

    return errors


def _auto_respond(interrupts: list) -> object:
    """Generate an auto-response for the first interrupt in the list.

    LangGraph currently surfaces one interrupt per invocation in this workflow,
    so we resume with a single value. The acquire interrupt expects a list of
    AcquisitionResponse-like dicts; everything else expects a string.
    """
    if not interrupts:
        return "approved"

    data = interrupts[0].value
    interrupt_type = data.get("type", "unknown")

    if interrupt_type == "clarify":
        return "No clarification needed. Proceed as-is."

    if interrupt_type == "approve":
        return "approved"

    if interrupt_type == "acquire":
        requests = data.get("requests", [])
        responses = []
        print(f"  Attempting to resolve {len(requests)} blocked source(s) directly...")
        for request in requests:
            source_id = request["source_id"]
            url = request.get("url", "")
            save_path = Path(request.get("save_path", ""))
            resolved = _try_download_acquisition(url, save_path)
            if resolved:
                print(f"    downloaded {url} -> {save_path}")
                responses.append(
                    AcquisitionResponse(
                        source_id=source_id, kind="saved", save_path=str(save_path)
                    ).model_dump()
                )
            else:
                print(f"    could not download {url}; marking unobtainable")
                responses.append(
                    AcquisitionResponse(source_id=source_id, kind="unobtainable").model_dump()
                )
        return responses

    if interrupt_type == "evaluate":
        return "approved"

    return "approved"


def _try_download_acquisition(url: str, save_path: Path) -> bool:
    """Attempt a direct download of a previously blocked URL.

    Returns True when bytes were successfully written to ``save_path``.
    """
    if not url or not save_path:
        return False
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"
        )
    }
    try:
        response = httpx.get(url, headers=headers, follow_redirects=True, timeout=30)
        if response.status_code != 200:
            return False
        content_type = response.headers.get("content-type", "")
        text = response.text if not content_type.startswith("application/pdf") else ""
        if text and is_login_wall(text):
            return False
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_path.write_bytes(response.content)
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"    download error for {url}: {exc}")
        return False


def _display_interrupt(interrupts: list) -> None:
    """Print a concise description of each interrupt."""
    for idx, interrupt_data in enumerate(interrupts, start=1):
        data = interrupt_data.value
        itype = data.get("type", "unknown")
        extra = ""
        if itype == "approve":
            brief = data.get("brief", {})
            extra = f" ({len(brief.get('subtopics', []))} sub-topics)"
        elif itype == "acquire":
            extra = f" ({len(data.get('requests', []))} blocked source(s))"
        print(f"  [interrupt {idx}] type={itype}{extra}")


def run_pipeline(question: str, cfg) -> ResearchState:
    """Run the compiled graph with real services and auto-approve interrupts."""
    run_slug = slug(question)

    embeddings_kwargs = {"model": cfg.embed_model, "base_url": cfg.embed_base_url}
    if cfg.ollama_api_key:
        embeddings_kwargs["client_kwargs"] = {
            "headers": {"Authorization": f"Bearer {cfg.ollama_api_key}"}
        }
    embeddings = LCOllamaEmbeddings(**embeddings_kwargs)
    store = ChromaStore(cfg.state_dir, embeddings.embed_query)

    print("Syncing bibliography (inbox + index)...")
    reconcile(cfg.bibliography_dir, store, embeddings)

    print(f"Building graph for run slug: {run_slug}")
    with create_checkpointer(cfg.state_dir) as checkpointer:
        graph = build_graph(checkpointer=checkpointer)

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
            "configurable": {
                "chat_fn": chat,
                "embeddings": embeddings,
                "store": store,
                "tavily_client": TavilyClient(api_key=cfg.tavily_api_key),
                "pdf_client": HttpxPdfClient(),
                "bibliography_dir": str(cfg.bibliography_dir),
                "state_dir": str(cfg.state_dir),
                "output_dir": str(cfg.output_dir),
            },
            "thread_id": run_slug,
        }

        current_input: object = initial_state
        for round_num in range(1, MAX_ROUNDS + 1):
            print(f"\n-- Round {round_num} --")
            start = time.perf_counter()
            result = graph.invoke(current_input, config)
            elapsed = time.perf_counter() - start
            print(f"  invocation took {elapsed:.2f}s")

            interrupts = result.get("__interrupt__", [])
            if not interrupts:
                print("\nRun completed (no interrupts).")
                return result

            _display_interrupt(interrupts)
            response = _auto_respond(interrupts)
            current_input = Command(resume=response)

        raise RuntimeError(f"Exceeded maximum of {MAX_ROUNDS} graph invocations")


def _verify_outputs(cfg, final_state: ResearchState) -> tuple[bool, list[str]]:
    """Verify that expected deliverables exist and invariants hold."""
    errors: list[str] = []
    run_slug = final_state["slug"]
    output_root = cfg.output_dir / run_slug

    print(f"\nVerifying outputs in {output_root}...")

    for name in ("brief.md", "report.md", "references.json"):
        path = output_root / name
        if not path.exists():
            errors.append(f"Missing: {path}")
        else:
            size = path.stat().st_size
            print(f"  OK {name} ({size} bytes)")

    if (output_root / "report.md").exists():
        report_text = (output_root / "report.md").read_text(encoding="utf-8")
        if not report_text.strip():
            errors.append("report.md is empty")

    brief = final_state.get("brief")
    if brief:
        for subtopic in brief.subtopics:
            sub_path = output_root / subtopic.slug / "report.md"
            if not sub_path.exists():
                errors.append(f"Missing sub-report: {sub_path}")
            else:
                print(f"  OK {subtopic.slug}/report.md")

    if not final_state.get("verify_ok"):
        errors.append("verify_ok is False")
    else:
        print("  OK verify_ok = True")

    bib_reports = list(cfg.bibliography_dir.glob("**/report.md"))
    if bib_reports:
        errors.append(f"Report files found in Bibliography: {bib_reports}")
    else:
        print("  OK no reports leaked into Bibliography")

    return not errors, errors


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    if args.output_dir is not None:
        output_dir = args.output_dir.resolve()
        os.environ["OUTPUT_DIR"] = str(output_dir)
        os.environ.setdefault("STATE_DIR", str(output_dir / ".deepresearch"))
    else:
        output_dir = Path(tempfile.mkdtemp(prefix="deepresearch-live-"))
        os.environ["OUTPUT_DIR"] = str(output_dir)
        os.environ["STATE_DIR"] = str(output_dir / ".deepresearch")

    # Force the config singleton to reload with the new paths.
    reset_config()
    cfg = get_config(env_file=".env")
    # Fall back to a local Ollama when the configured hosts are unreachable.
    resolve_ollama_endpoints(cfg)

    print(f"Research question: {args.question}")
    print(f"Output directory:  {output_dir}")
    print(f"Ollama base URL:   {cfg.ollama_base_url}")
    print(f"Embed base URL:    {cfg.embed_base_url}")
    print(f"Embed model:       {cfg.embed_model}")
    print(f"Tavily key:        {'***' if cfg.tavily_api_key else 'MISSING'}")

    prereq_errors = _validate_prereqs(cfg)
    if prereq_errors:
        print("\nPrerequisite check failed:")
        for err in prereq_errors:
            print(f"  - {err}")
        print("\nAborting.")
        return 1

    print("\nAll prerequisites passed.")

    try:
        final_state = run_pipeline(args.question, cfg)
    except Exception as exc:  # noqa: BLE001
        import traceback

        print(f"\nPipeline failed with {type(exc).__name__}: {exc}")
        traceback.print_exc()
        print(f"Output left at {output_dir} for inspection.")
        return 1

    ok, errors = _verify_outputs(cfg, final_state)

    if not ok:
        print("\nVerification failed:")
        for err in errors:
            print(f"  - {err}")
        print(f"\nOutput left at {output_dir} for inspection.")
        return 1

    print("\nVerification passed.")
    print(f"Output left at {output_dir} for inspection.")

    if args.cleanup and args.output_dir is None:
        import shutil

        shutil.rmtree(output_dir, ignore_errors=True)
        print(f"Cleaned up {output_dir}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
