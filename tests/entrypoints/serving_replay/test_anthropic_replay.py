# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Replay gold-dataset token streams through the Anthropic Messages
serving layer (streaming and non-streaming) and verify client-visible output.

The Anthropic layer wraps OpenAI Chat Completions internally:
- Streaming: OpenAI SSE → ``message_stream_converter()`` → Anthropic SSE
- Non-streaming: ``ChatCompletionResponse`` → ``messages_full_converter()``
"""

from __future__ import annotations

import pytest

from tests.entrypoints.serving_replay.mock_serving import (
    build_anthropic_serving,
    build_serving_chat,
)
from tests.entrypoints.serving_replay.replay_harness import (
    CHUNK_SIZES,
    DELEGATING_OLD_XFAIL_SAMPLES,
    accumulate_anthropic_sse,
    assert_anthropic_response,
    get_parser_cls,
    make_mock_tokenizer,
    tokens_to_request_outputs,
)
from tests.parser.engine.trace_builder import build_serving_samples
from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
    StreamOptions,
)
from vllm.entrypoints.openai.engine.protocol import RequestResponseMetadata

_all_samples = (
    build_serving_samples("qwen3")
    + build_serving_samples("gemma4")
    + build_serving_samples("deepseek_v4")
)


def _build_request(sample, *, stream: bool = True) -> ChatCompletionRequest:
    req = ChatCompletionRequest(
        model="test-model",
        messages=[{"role": "user", "content": "test"}],
        stream=stream,
        tools=sample.tools,
        tool_choice=sample.tool_choice if sample.tools else None,
    )
    if stream:
        req.stream_options = StreamOptions.model_validate(
            {"include_usage": True, "continuous_usage_stats": True}
        )
    return req


async def _generate_openai_sse_stream(sample, chunk_size, parser_mode="engine"):
    """Generate OpenAI SSE chunks from a sample's tokens."""
    parser_cls = get_parser_cls(sample.parser_name, mode=parser_mode)
    serving_chat = build_serving_chat(parser_cls)
    tokenizer = make_mock_tokenizer(sample)
    request = _build_request(sample, stream=True)
    result_gen = tokens_to_request_outputs(sample.tokens, chunk_size)

    async for chunk in serving_chat.chat_completion_stream_generator(
        request=request,
        result_generator=result_gen,
        request_id="test-req",
        model_name="test-model",
        conversation=[],
        tokenizer=tokenizer,
        request_metadata=RequestResponseMetadata(request_id="test-req"),
        chat_template_kwargs=sample.chat_template_kwargs,
    ):
        yield chunk


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "parser_mode",
    ["engine", "delegating", "delegating_engine"],
    ids=lambda m: f"mode={m}",
)
@pytest.mark.parametrize("chunk_size", CHUNK_SIZES, ids=lambda c: f"chunk={c}")
@pytest.mark.parametrize("sample", _all_samples, ids=lambda s: s.id)
async def test_anthropic_streaming(sample, chunk_size, parser_mode):
    if parser_mode == "delegating" and sample.id in DELEGATING_OLD_XFAIL_SAMPLES:
        pytest.xfail("old delegating parser has streaming differences")
    anthropic_serving = build_anthropic_serving(
        get_parser_cls(sample.parser_name, mode=parser_mode)
    )
    openai_sse_gen = _generate_openai_sse_stream(sample, chunk_size, parser_mode)

    anthropic_chunks = []
    async for chunk in anthropic_serving.message_stream_converter(openai_sse_gen):
        anthropic_chunks.append(chunk)

    response = accumulate_anthropic_sse(anthropic_chunks)
    assert_anthropic_response(response, sample)
