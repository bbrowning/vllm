# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Replay gold-dataset token streams through the OpenAI Responses API
serving layer (streaming and non-streaming) and verify client-visible output.

The Responses API uses ``SimpleContext`` to wrap ``RequestOutput`` objects
and routes through ``_process_simple_streaming_events`` for streaming or
``responses_full_generator`` for non-streaming.
"""

from __future__ import annotations

from typing import Any

import pytest

from tests.entrypoints.serving_replay.mock_serving import build_serving_responses
from tests.entrypoints.serving_replay.replay_harness import (
    CHUNK_SIZES,
    assert_responses_events,
    get_parser_cls,
    load_serving_samples,
    make_mock_tokenizer,
    tokens_to_simple_contexts,
)
from vllm.entrypoints.openai.engine.protocol import RequestResponseMetadata
from vllm.entrypoints.openai.responses.context import SimpleContext
from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
from vllm.sampling_params import SamplingParams

_all_samples = (
    load_serving_samples("qwen3")
    + load_serving_samples("gemma4")
    + load_serving_samples("deepseek_v4")
)


def _convert_tools_to_responses_format(
    tools: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Convert Chat Completions tool dicts to Responses API format."""
    result = []
    for tool in tools:
        if tool.get("type") == "function" and "function" in tool:
            func = tool["function"]
            result.append(
                {
                    "type": "function",
                    "name": func["name"],
                    "parameters": func.get("parameters", {}),
                    "description": func.get("description"),
                }
            )
        else:
            result.append(tool)
    return result


def _build_responses_request(sample) -> ResponsesRequest:
    kwargs: dict[str, Any] = {
        "model": "test-model",
        "input": "test",
        "stream": True,
    }
    if sample.tools:
        kwargs["tools"] = _convert_tools_to_responses_format(sample.tools)
        kwargs["tool_choice"] = sample.tool_choice
    return ResponsesRequest(**kwargs)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "parser_mode", ["engine", "delegating"], ids=lambda m: f"mode={m}"
)
@pytest.mark.parametrize("chunk_size", CHUNK_SIZES, ids=lambda c: f"chunk={c}")
@pytest.mark.parametrize("sample", _all_samples, ids=lambda s: s.id)
async def test_responses_streaming(sample, chunk_size, parser_mode):
    tokenizer = make_mock_tokenizer(sample)
    parser_cls = get_parser_cls(sample.parser_name, mode=parser_mode)
    serving_responses = build_serving_responses(
        parser_cls, chat_template_kwargs=sample.chat_template_kwargs
    )
    request = _build_responses_request(sample)
    sampling_params = SamplingParams()
    context = SimpleContext()
    context_gen = tokens_to_simple_contexts(sample.tokens, chunk_size, context=context)

    events = []
    async for event in serving_responses.responses_stream_generator(
        request=request,
        sampling_params=sampling_params,
        result_generator=context_gen,
        context=context,
        model_name="test-model",
        tokenizer=tokenizer,
        request_metadata=RequestResponseMetadata(request_id="test-req"),
    ):
        events.append(event)

    assert_responses_events(events, sample)
