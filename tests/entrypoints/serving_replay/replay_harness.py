# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Data-driven replay harness for serving-layer tests.

Loads token sequences from JSONL fixtures and replays them through the
full serving pipeline (Chat Completions, Anthropic Messages, Responses
API) at different chunk sizes — without requiring a GPU.
"""

from __future__ import annotations

import json
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from typing import Any

# Re-use the mock tokenizer builder from the grammar parser tests.
from tests.parser.grammar.replay_harness import (
    DATA_DIR,
)
from tests.parser.grammar.replay_harness import (
    make_mock_tokenizer as _make_mock_tokenizer,
)
from vllm.entrypoints.openai.responses.context import SimpleContext
from vllm.outputs import CompletionOutput, RequestOutput
from vllm.parser import ParserManager
from vllm.parser.abstract_parser import Parser

CHUNK_SIZES = [1, 2, 3, 5, 11, 23, None]


@dataclass
class ServingSample:
    """One test sample loaded from a JSONL fixture file."""

    id: str
    description: str
    source: str
    vocab: dict[str, int]
    tokens: list[tuple[int, str]]
    # Parser expectations
    expected_reasoning: str | None
    expected_content: str | None
    expected_tool_calls: list[dict] | None
    # Serving context
    parser_name: str
    tool_choice: str
    tools: list[dict] | None
    include_reasoning: bool
    expected_finish_reason: str
    chat_template_kwargs: dict[str, Any] | None

    def __repr__(self) -> str:
        return f"ServingSample({self.id!r})"


def load_serving_samples(model: str) -> list[ServingSample]:
    """Load all samples with ``serving`` section from
    ``tests/parser/grammar/data/{model}.jsonl``."""
    path = DATA_DIR / f"{model}.jsonl"
    if not path.exists():
        return []

    samples = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        data = json.loads(line)
        serving = data.get("serving")
        if serving is None:
            continue
        tokens = [(t[0], t[1]) for t in data["tokens"]]
        expected = data.get("expected", {})
        samples.append(
            ServingSample(
                id=data["id"],
                description=data.get("description", ""),
                source=data.get("source", ""),
                vocab=data.get("vocab", {}),
                tokens=tokens,
                expected_reasoning=expected.get("reasoning"),
                expected_content=expected.get("content"),
                expected_tool_calls=expected.get("tool_calls"),
                parser_name=serving["parser_name"],
                tool_choice=serving.get("tool_choice", "auto"),
                tools=serving.get("tools"),
                include_reasoning=serving.get("include_reasoning", True),
                expected_finish_reason=serving.get("expected_finish_reason", "stop"),
                chat_template_kwargs=serving.get("chat_template_kwargs"),
            )
        )
    return samples


def make_mock_tokenizer(sample: ServingSample):
    """Build a mock tokenizer from a :class:`ServingSample`.

    Delegates to the grammar parser test harness's ``make_mock_tokenizer``.
    """
    from tests.parser.grammar.replay_harness import Sample as GrammarSample

    grammar_sample = GrammarSample(
        id=sample.id,
        description=sample.description,
        source=sample.source,
        vocab=sample.vocab,
        tokens=sample.tokens,
        expected_reasoning=sample.expected_reasoning,
        expected_content=sample.expected_content,
        expected_tool_calls=sample.expected_tool_calls,
    )
    return _make_mock_tokenizer(grammar_sample)


def get_parser_cls(parser_name: str) -> type[Parser]:
    """Look up a parser class by name from the ParserManager registry."""
    return ParserManager.get_parser_internal(parser_name)


# ---------------------------------------------------------------------------
# Token → engine output generators
# ---------------------------------------------------------------------------


async def tokens_to_request_outputs(
    tokens: list[tuple[int, str]],
    chunk_size: int | None = None,
) -> AsyncGenerator[RequestOutput, None]:
    """Yield ``RequestOutput`` objects from a token stream.

    Used by Chat Completions and Anthropic replay tests.
    """
    if chunk_size is None:
        chunk_size = len(tokens)

    all_ids = [tid for tid, _ in tokens]
    all_texts = [text for _, text in tokens]
    chunks = list(range(0, len(tokens), chunk_size))

    for i, start in enumerate(chunks):
        batch_end = min(start + chunk_size, len(tokens))
        batch_ids = all_ids[start:batch_end]
        delta_text = "".join(all_texts[start:batch_end])
        is_last = i == len(chunks) - 1

        yield RequestOutput(
            request_id="test-req",
            prompt=None,
            prompt_token_ids=[],
            prompt_logprobs=None,
            outputs=[
                CompletionOutput(
                    index=0,
                    text=delta_text,
                    token_ids=batch_ids,
                    cumulative_logprob=None,
                    logprobs=None,
                    finish_reason="stop" if is_last else None,
                )
            ],
            finished=is_last,
        )


async def tokens_to_simple_contexts(
    tokens: list[tuple[int, str]],
    chunk_size: int | None = None,
    context: SimpleContext | None = None,
) -> AsyncGenerator[SimpleContext, None]:
    """Yield ``SimpleContext`` objects from a token stream.

    Used by Responses API replay tests. Mirrors the engine's
    ``_generate_with_builtin_tools`` loop which calls
    ``context.append_output(req_output)`` then yields the same context.

    Pass an existing ``context`` to share state with the caller (required
    for ``responses_stream_generator`` which reads ``context.final_output``
    after the stream completes).
    """
    if context is None:
        context = SimpleContext()
    async for req_output in tokens_to_request_outputs(tokens, chunk_size):
        context.append_output(req_output)
        yield context


# ---------------------------------------------------------------------------
# Assertion helpers
# ---------------------------------------------------------------------------


def assert_chat_completion_response(response, sample: ServingSample) -> None:
    """Assert a ``ChatCompletionResponse`` matches expected values."""
    assert len(response.choices) == 1
    choice = response.choices[0]
    message = choice.message

    if sample.expected_content is not None:
        actual = message.content or ""
        assert actual == sample.expected_content, (
            f"Content mismatch:\n"
            f"  expected: {sample.expected_content!r}\n"
            f"  actual: {actual!r}"
        )

    if sample.expected_reasoning is not None:
        actual = message.reasoning or ""
        assert actual == sample.expected_reasoning, (
            f"Reasoning mismatch:\n"
            f"  expected: {sample.expected_reasoning!r}\n"
            f"  actual: {actual!r}"
        )

    if sample.expected_tool_calls is not None:
        actual_calls = message.tool_calls or []
        assert len(actual_calls) == len(sample.expected_tool_calls), (
            f"Tool call count: "
            f"expected {len(sample.expected_tool_calls)}, "
            f"got {len(actual_calls)}"
        )
        for i, (expected_tc, actual_tc) in enumerate(
            zip(sample.expected_tool_calls, actual_calls)
        ):
            assert actual_tc.function.name == expected_tc["name"], (
                f"Tool call {i} name: "
                f"expected {expected_tc['name']!r}, "
                f"got {actual_tc.function.name!r}"
            )
            if "arguments" in expected_tc:
                expected_args = expected_tc["arguments"]
                if isinstance(expected_args, dict):
                    actual_args = json.loads(actual_tc.function.arguments)
                    assert actual_args == expected_args, (
                        f"Tool call {i} args:\n"
                        f"  expected: {expected_args}\n"
                        f"  actual: {actual_args}"
                    )

    assert choice.finish_reason == sample.expected_finish_reason, (
        f"finish_reason: expected {sample.expected_finish_reason!r}, "
        f"got {choice.finish_reason!r}"
    )


def assert_anthropic_response(response, sample: ServingSample) -> None:
    """Assert an ``AnthropicMessagesResponse`` matches expected values."""
    finish_reason_map = {
        "stop": "end_turn",
        "tool_calls": "tool_use",
    }
    expected_stop_reason = finish_reason_map.get(
        sample.expected_finish_reason, sample.expected_finish_reason
    )
    assert response.stop_reason == expected_stop_reason, (
        f"stop_reason: expected {expected_stop_reason!r}, got {response.stop_reason!r}"
    )

    # Collect content blocks by type
    thinking_blocks = [b for b in response.content if b.type == "thinking"]
    text_blocks = [b for b in response.content if b.type == "text"]
    tool_use_blocks = [b for b in response.content if b.type == "tool_use"]

    if sample.expected_reasoning is not None:
        actual_reasoning = "".join(b.thinking for b in thinking_blocks)
        assert actual_reasoning == sample.expected_reasoning, (
            f"Reasoning mismatch:\n  expected: {sample.expected_reasoning!r}\n"
            f"  actual: {actual_reasoning!r}"
        )

    if sample.expected_content is not None:
        actual_content = "".join(b.text for b in text_blocks)
        assert actual_content == sample.expected_content, (
            f"Content mismatch:\n  expected: {sample.expected_content!r}\n"
            f"  actual: {actual_content!r}"
        )

    if sample.expected_tool_calls is not None:
        assert len(tool_use_blocks) == len(sample.expected_tool_calls), (
            f"Tool call count: expected {len(sample.expected_tool_calls)}, "
            f"got {len(tool_use_blocks)}"
        )
        for i, (expected_tc, actual_tc) in enumerate(
            zip(sample.expected_tool_calls, tool_use_blocks)
        ):
            assert actual_tc.name == expected_tc["name"], (
                f"Tool call {i} name: expected {expected_tc['name']!r}, "
                f"got {actual_tc.name!r}"
            )
            if "arguments" in expected_tc:
                expected_args = expected_tc["arguments"]
                if isinstance(expected_args, dict):
                    assert actual_tc.input == expected_args, (
                        f"Tool call {i} args:\n  expected: {expected_args}\n"
                        f"  actual: {actual_tc.input}"
                    )


def assert_responses_events(events: list, sample: ServingSample) -> None:
    """Assert Responses API streaming events match expected values.

    Reassembles text/reasoning/tool_call deltas from the event stream
    and compares against the sample's expected output.
    """
    accumulated_text = ""
    accumulated_reasoning = ""
    tool_calls: dict[int, dict] = {}  # output_index -> {name, arguments}
    completed_event = None

    for event in events:
        event_type = getattr(event, "type", "")
        if event_type == "response.output_text.delta":
            accumulated_text += event.delta
        elif event_type == "response.reasoning_text.delta":
            accumulated_reasoning += event.delta
        elif event_type == "response.function_call_arguments.delta":
            idx = event.output_index
            if idx not in tool_calls:
                tool_calls[idx] = {"name": "", "arguments": ""}
            tool_calls[idx]["arguments"] += event.delta
        elif event_type == "response.output_item.added":
            item = event.item
            if hasattr(item, "call_id"):
                idx = event.output_index
                tool_calls[idx] = {
                    "name": getattr(item, "name", ""),
                    "arguments": "",
                }
        elif event_type == "response.completed":
            completed_event = event

    if sample.expected_content is not None:
        assert accumulated_text == sample.expected_content, (
            f"Content mismatch:\n  expected: {sample.expected_content!r}\n"
            f"  actual: {accumulated_text!r}"
        )

    if sample.expected_reasoning is not None:
        assert accumulated_reasoning == sample.expected_reasoning, (
            f"Reasoning mismatch:\n  expected: {sample.expected_reasoning!r}\n"
            f"  actual: {accumulated_reasoning!r}"
        )

    if sample.expected_tool_calls is not None:
        actual_calls = list(tool_calls.values())
        assert len(actual_calls) == len(sample.expected_tool_calls), (
            f"Tool call count: expected {len(sample.expected_tool_calls)}, "
            f"got {len(actual_calls)}"
        )
        for i, (expected_tc, actual_tc) in enumerate(
            zip(sample.expected_tool_calls, actual_calls)
        ):
            assert actual_tc["name"] == expected_tc["name"], (
                f"Tool call {i} name: expected {expected_tc['name']!r}, "
                f"got {actual_tc['name']!r}"
            )
            if "arguments" in expected_tc:
                expected_args = expected_tc["arguments"]
                if isinstance(expected_args, dict):
                    actual_args = json.loads(actual_tc["arguments"])
                    assert actual_args == expected_args, (
                        f"Tool call {i} args:\n  expected: {expected_args}\n"
                        f"  actual: {actual_args}"
                    )

    assert completed_event is not None, "Missing response.completed event"


# ---------------------------------------------------------------------------
# Anthropic SSE accumulator
# ---------------------------------------------------------------------------


def accumulate_anthropic_sse(chunks: list[str]):
    """Parse Anthropic SSE chunks and return an AnthropicMessagesResponse-like
    object for assertion.

    Returns a simple namespace with the same fields the assertion helper uses.
    """
    from vllm.entrypoints.anthropic.protocol import (
        AnthropicContentBlock,
        AnthropicMessagesResponse,
        AnthropicUsage,
    )

    content_blocks: list[AnthropicContentBlock] = []
    current_block: dict | None = None
    stop_reason = None
    response_id = None
    model = None

    for chunk in chunks:
        chunk = chunk.strip()
        if not chunk:
            continue

        # Parse "event: type\ndata: {json}" format
        lines = chunk.split("\n")
        event_type = None
        data_str = None
        for line in lines:
            if line.startswith("event: "):
                event_type = line[7:].strip()
            elif line.startswith("data: "):
                data_str = line[6:].strip()

        if not data_str:
            continue
        try:
            data = json.loads(data_str)
        except json.JSONDecodeError:
            continue

        if event_type == "message_start":
            msg = data.get("message", {})
            response_id = msg.get("id")
            model = msg.get("model")
        elif event_type == "content_block_start":
            cb = data.get("content_block", {})
            current_block = {
                "type": cb.get("type", "text"),
                "text": cb.get("text", ""),
                "thinking": cb.get("thinking", ""),
                "id": cb.get("id"),
                "name": cb.get("name"),
                "input": cb.get("input", {}),
            }
        elif event_type == "content_block_delta":
            delta = data.get("delta", {})
            delta_type = delta.get("type", "")
            if current_block is not None:
                if delta_type == "text_delta":
                    current_block["text"] += delta.get("text", "")
                elif delta_type == "thinking_delta":
                    current_block["thinking"] += delta.get("thinking", "")
                elif delta_type == "input_json_delta":
                    if "input_str" not in current_block:
                        current_block["input_str"] = ""
                    current_block["input_str"] += delta.get("partial_json", "")
        elif event_type == "content_block_stop":
            if current_block is not None:
                block_type = current_block["type"]
                if block_type == "thinking":
                    content_blocks.append(
                        AnthropicContentBlock(
                            type="thinking",
                            thinking=current_block["thinking"],
                        )
                    )
                elif block_type == "text":
                    content_blocks.append(
                        AnthropicContentBlock(
                            type="text",
                            text=current_block["text"],
                        )
                    )
                elif block_type == "tool_use":
                    input_data = current_block.get("input", {})
                    if "input_str" in current_block and current_block["input_str"]:
                        try:
                            input_data = json.loads(current_block["input_str"])
                        except json.JSONDecodeError:
                            input_data = current_block["input_str"]
                    content_blocks.append(
                        AnthropicContentBlock(
                            type="tool_use",
                            id=current_block.get("id"),
                            name=current_block.get("name"),
                            input=input_data,
                        )
                    )
                current_block = None
        elif event_type == "message_delta":
            delta = data.get("delta", {})
            stop_reason = delta.get("stop_reason", stop_reason)

    return AnthropicMessagesResponse(
        id=response_id or "msg-test",
        content=content_blocks,
        model=model or "test-model",
        usage=AnthropicUsage(input_tokens=0, output_tokens=0),
        stop_reason=stop_reason,
    )
