"""Ollama chat client with role-tiered model resolution."""

import httpx
from langchain_ollama import ChatOllama

from deepresearch.config import get_config
from deepresearch.retry import OllamaTransientError, retry_call


def chat(role: str, messages: list) -> str:
    """Send messages to the Ollama host using the model assigned to `role`.

    Roles:
      - clarify  > fast model
      - gate, synth  > large-context model
      - writer, eval  > strong model
    """
    cfg = get_config()
    model_name = {
        "clarify": cfg.model_fast,
        "gate": cfg.model_long,
        "synth": cfg.model_long,
        "writer": cfg.model_writer,
        "eval": cfg.model_writer,
    }.get(role, cfg.model_fast)

    llm = ChatOllama(model=model_name, base_url=cfg.ollama_base_url)

    def _invoke():
        """Invoke Ollama; retry only transport-like transient failures.

        Retryable errors are deliberately narrow: httpx connect/timeout errors
        from the Ollama client plus built-in ConnectionError/TimeoutError.
        Application/model errors are re-raised unchanged.
        """
        try:
            return llm.invoke(messages)
        except (httpx.ConnectError, httpx.TimeoutException, ConnectionError, TimeoutError) as exc:
            raise OllamaTransientError(str(exc)) from exc

    response = retry_call(
        _invoke,
        max_attempts=cfg.llm_max_retries,
        initial_interval=cfg.llm_retry_initial_interval,
        backoff_factor=2.0,
        max_interval=4.0,
    )
    return str(response.content)
