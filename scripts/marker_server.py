#!/usr/bin/env python3
"""Marker PDF conversion server.

Runs marker-pdf as a lightweight HTTP service so DeepResearch (or any client)
can offload PDF conversion to a more powerful machine.

Setup on the server machine
----------------------------
Install dependencies (marker-pdf pulls torch/safetensors — that's fine here):

    pip install marker-pdf fastapi "uvicorn[standard]" python-multipart

Or, if you have this repo checked out on the server:

    uv sync --extra server

Run the server:

    python scripts/marker_server.py

Environment variables
---------------------
    MARKER_SERVER_HOST      Bind address (default: 0.0.0.0)
    MARKER_SERVER_PORT      Bind port    (default: 8080)
    MARKER_SERVER_API_KEY   If set, require ``Authorization: Bearer <key>`` on
                            every request. Leave unset to disable auth (LAN only).

Client configuration (in DeepResearch .env)
---------------------------------------------
    PDF_CONVERTER=remote
    PDF_CONVERTER_URL=http://<server-ip>:8080/convert
    PDF_CONVERTER_API_KEY=<same key as MARKER_SERVER_API_KEY, if set>

API
---
    POST /convert
        Multipart form upload, field name ``file``.
        Returns: {"markdown": "<converted text>"}

    GET /health
        Returns: {"status": "ok"}
"""

import logging
import os
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Marker model — loaded once at startup, reused for every request
# ---------------------------------------------------------------------------

def _load_marker():
    from marker.config.parser import ConfigParser
    from marker.converters.pdf import PdfConverter
    from marker.models import create_model_dict

    config = {
        "use_llm": False,
        "force_ocr": False,
        "languages": "en",
        "output_format": "markdown",
        "output_dir": tempfile.gettempdir(),
        "processors": None,
        "config_json": None,
        "disable_multiprocessing": False,
        "disable_image_extraction": True,
        "page_range": None,
        "converter_cls": None,
        "llm_service": None,
    }
    config_parser = ConfigParser(config)
    artifact_dict = create_model_dict()
    return PdfConverter(
        config=config_parser.generate_config_dict(),
        artifact_dict=artifact_dict,
        processor_list=config_parser.get_processors(),
        renderer=config_parser.get_renderer(),
        llm_service=config_parser.get_llm_service(),
    )


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

def create_app():
    try:
        from fastapi import Depends, FastAPI, HTTPException, UploadFile
        from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
    except ImportError as exc:
        raise ImportError(
            "Server dependencies are not installed.\n"
            "Run:  pip install fastapi 'uvicorn[standard]' python-multipart\n"
            "Or:   uv sync --extra server"
        ) from exc

    _converter = None

    @asynccontextmanager
    async def lifespan(app):
        nonlocal _converter
        log.info("Loading marker models — this may take a minute on first run...")
        _converter = _load_marker()
        log.info("Marker ready.")
        yield
        _converter = None

    app = FastAPI(title="Marker PDF Converter", lifespan=lifespan)
    _security = HTTPBearer(auto_error=False)

    def _check_auth(credentials: HTTPAuthorizationCredentials | None = Depends(_security)):
        api_key = os.environ.get("MARKER_SERVER_API_KEY", "")
        if not api_key:
            return  # auth not configured — allow all (use on trusted LAN only)
        if credentials is None or credentials.credentials != api_key:
            raise HTTPException(status_code=401, detail="Invalid or missing API key")

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.post("/convert")
    async def convert(file: UploadFile, _: None = Depends(_check_auth)):
        pdf_bytes = await file.read()
        if not pdf_bytes or not pdf_bytes.startswith(b"%PDF-"):
            raise HTTPException(status_code=400, detail="Invalid or empty PDF content")

        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(pdf_bytes)
            tmp_path = Path(tmp.name)

        try:
            rendered = _converter(str(tmp_path))
        finally:
            tmp_path.unlink(missing_ok=True)

        markdown = rendered.markdown if hasattr(rendered, "markdown") else str(rendered)
        return {"markdown": markdown}

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    logging.basicConfig(level=logging.INFO)
    uvicorn.run(
        app,
        host=os.environ.get("MARKER_SERVER_HOST", "0.0.0.0"),
        port=int(os.environ.get("MARKER_SERVER_PORT", "8080")),
    )
