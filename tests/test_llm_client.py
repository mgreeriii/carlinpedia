from types import SimpleNamespace

import pytest
from pydantic import BaseModel

from carlinpedia.llm.client import AnthropicLLM, FakeLLM, LLMError, load_prompt


class Answer(BaseModel):
    value: int


class RecordingStream:
    def __init__(self, response):
        self.response = response

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def get_final_message(self):
        return self.response


class RecordingMessages:
    """Only streaming is offered: long thinking plus a large JSON answer needs a high max_tokens,
    and the SDK refuses non-streaming requests that could run that long."""

    def __init__(self, response):
        self.response = response
        self.kwargs = None

    def stream(self, **kwargs):
        self.kwargs = kwargs
        return RecordingStream(self.response)


def make_llm(stop_reason="end_turn", parsed=Answer(value=3)):
    messages = RecordingMessages(SimpleNamespace(stop_reason=stop_reason, parsed_output=parsed))
    return AnthropicLLM("claude-sonnet-5", client=SimpleNamespace(messages=messages)), messages


def test_anthropic_llm_builds_the_request():
    llm, messages = make_llm()
    result = llm.parse(prompt_name="p", key="k", system="SYS", user="USER", output_model=Answer)
    assert result == Answer(value=3)
    kw = messages.kwargs
    assert kw["model"] == "claude-sonnet-5"
    assert kw["max_tokens"] == 64000
    assert kw["system"] == [{"type": "text", "text": "SYS", "cache_control": {"type": "ephemeral"}}]
    assert kw["messages"] == [{"role": "user", "content": "USER"}]
    assert kw["thinking"] == {"type": "adaptive"}
    assert kw["output_format"] is Answer


@pytest.mark.parametrize("stop_reason,parsed,message", [
    ("refusal", None, "declined"),
    ("max_tokens", None, "max_tokens"),
    ("end_turn", None, "no structured output"),
])
def test_anthropic_llm_raises_on_unusable_responses(stop_reason, parsed, message):
    llm, _ = make_llm(stop_reason, parsed)
    with pytest.raises(LLMError, match=message):
        llm.parse(prompt_name="p", key="k", system="s", user="u", output_model=Answer)


def test_fake_llm_replays_in_order_and_repeats_the_last():
    fake = FakeLLM({"p:k": [{"value": 1}, {"value": 2}]})
    values = [fake.parse(prompt_name="p", key="k", system="", user="", output_model=Answer).value for _ in range(3)]
    assert values == [1, 2, 2]
    assert [c["key"] for c in fake.calls] == ["p:k"] * 3


def test_fake_llm_missing_key_and_exceptions():
    fake = FakeLLM({"p:boom": [LLMError("scripted failure")]})
    with pytest.raises(LLMError, match="no response left"):
        fake.parse(prompt_name="p", key="missing", system="", user="", output_model=Answer)
    with pytest.raises(LLMError, match="scripted failure"):
        fake.parse(prompt_name="p", key="boom", system="", user="", output_model=Answer)


def test_prompts_ship_with_the_package():
    assert "sentence unit" in load_prompt("segment_v1").lower()
    assert "thesis" in load_prompt("enrich_v1").lower()
