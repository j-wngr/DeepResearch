"""Ollama chat client with role-tiered model resolution."""

from langchain_ollama import ChatOllama

from deepresearch.config import get_config


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
    response = llm.invoke(messages)
    return str(response.content)
