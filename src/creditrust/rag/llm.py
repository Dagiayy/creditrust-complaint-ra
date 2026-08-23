"""LLM provider abstraction.

The RAG pipeline talks to an `LLMProvider` interface rather than a concrete
HuggingFace pipeline. This keeps `RAGPipeline` testable without downloading
multi-hundred-MB-to-multi-GB model weights, and makes it possible to swap the
generation backend (e.g. a stronger instruction-tuned model, or a hosted API)
via configuration instead of code changes.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from creditrust.config import Settings, get_settings
from creditrust.logging_config import get_logger

logger = get_logger(__name__)


class LLMProvider(ABC):
    @abstractmethod
    def generate(self, prompt: str) -> str:
        """Generate a completion for `prompt`."""


class LocalHFProvider(LLMProvider):
    """Runs a local HuggingFace causal/seq2seq LM via `transformers.pipeline`.

    The model is loaded lazily on first use (not at import time) so importing
    this module — e.g. from the API or test suite — never triggers a model
    download or multi-second load.
    """

    def __init__(self, settings: Settings | None = None):
        self._settings = settings or get_settings()
        self._pipe = None

    def _load(self):
        if self._pipe is not None:
            return self._pipe

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

        model_name = self._settings.llm_model_name
        logger.info("Loading local LLM %s ...", model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name, token=self._settings.hf_token)
        model = AutoModelForCausalLM.from_pretrained(model_name, token=self._settings.hf_token)
        self._pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=self._settings.llm_max_new_tokens,
            do_sample=self._settings.llm_temperature > 0,
            temperature=max(self._settings.llm_temperature, 1e-4),
            device=torch.device("cpu"),
        )
        logger.info("LLM %s loaded.", model_name)
        return self._pipe

    def generate(self, prompt: str) -> str:
        pipe = self._load()
        output = pipe(prompt, return_full_text=False)
        return output[0]["generated_text"]


class MockLLMProvider(LLMProvider):
    """Deterministic provider used in tests and offline demos — no model weights."""

    def __init__(self, canned_response: str | None = None):
        self._canned_response = canned_response

    def generate(self, prompt: str) -> str:
        if self._canned_response is not None:
            return self._canned_response
        return "Answer: [mock] Based on the provided context, this is a placeholder summary."


def get_llm_provider(settings: Settings | None = None) -> LLMProvider:
    settings = settings or get_settings()
    if settings.llm_provider == "mock":
        return MockLLMProvider()
    return LocalHFProvider(settings)
