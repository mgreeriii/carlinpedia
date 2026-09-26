"""A narrow LLM interface: one structured-output call. Real calls go to Anthropic; tests use FakeLLM."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from typing import Any, Protocol, TypeVar

import anthropic
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)

# Thinking tokens count against max_tokens, and a full special's segmentation is a large JSON answer on top.
# Requests this large must stream: the SDK refuses non-streaming calls it expects to run over ten minutes.
DEFAULT_MAX_TOKENS = 64000


class LLMError(Exception):
    """The model refused, was truncated, or returned nothing parseable."""


class LLM(Protocol):
    model: str

    def parse(self, *, prompt_name: str, key: str, system: str, user: str, output_model: type[T]) -> T: ...


def load_prompt(name: str) -> str:
    return resources.files("carlinpedia.llm").joinpath("prompts", f"{name}.md").read_text(encoding="utf-8")


class AnthropicLLM:
    def __init__(self, model: str, client: anthropic.Anthropic | None = None, max_tokens: int = DEFAULT_MAX_TOKENS):
        self.model = model
        self._client = client or anthropic.Anthropic()
        self._max_tokens = max_tokens

    def parse(self, *, prompt_name: str, key: str, system: str, user: str, output_model: type[T]) -> T:
        with self._client.messages.stream(
            model=self.model,
            max_tokens=self._max_tokens,
            # The system prompt is identical across every call for a prompt, so cache it.
            system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            messages=[{"role": "user", "content": user}],
            thinking={"type": "adaptive"},
            output_format=output_model,
        ) as stream:
            response = stream.get_final_message()
        if response.stop_reason == "refusal":
            raise LLMError(f"{prompt_name} [{key}]: the model declined the request")
        if response.stop_reason == "max_tokens":
            raise LLMError(f"{prompt_name} [{key}]: the response hit max_tokens ({self._max_tokens}) and was cut off")
        if response.parsed_output is None:
            raise LLMError(f"{prompt_name} [{key}]: no structured output in the response")
        return response.parsed_output

    def count_tokens(self, *, system: str, user: str) -> int:
        result = self._client.messages.count_tokens(
            model=self.model, system=system, messages=[{"role": "user", "content": user}]
        )
        return result.input_tokens


@dataclass
class FakeLLM:
    """Replays canned responses keyed by "<prompt_name>:<key>". Each key holds a list consumed in order."""

    responses: dict[str, list[Any]]
    model: str = "fake-model"
    calls: list[dict] = field(default_factory=list)

    @classmethod
    def from_file(cls, path: Path) -> "FakeLLM":
        return cls(responses=json.loads(path.read_text(encoding="utf-8")))

    def parse(self, *, prompt_name: str, key: str, system: str, user: str, output_model: type[T]) -> T:
        full_key = f"{prompt_name}:{key}"
        self.calls.append({"key": full_key, "system": system, "user": user})
        queue = self.responses.get(full_key)
        if not queue:
            raise LLMError(f"FakeLLM has no response left for {full_key}")
        item = queue.pop(0) if len(queue) > 1 else queue[0]
        if isinstance(item, Exception):
            raise item
        return output_model.model_validate(item)
