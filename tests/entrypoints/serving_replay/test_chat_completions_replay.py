# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Replay gold-dataset token streams through the OpenAI Chat Completions
serving layer (streaming and non-streaming) and verify client-visible output.

Each fixture is tested at multiple chunk sizes to verify chunk-size invariance
at the serving layer, not just the parser level.
"""

from __future__ import annotations

import pytest

from tests.entrypoints.openai.utils import accumulate_streaming_response
from tests.entrypoints.serving_replay.mock_serving import build_serving_chat
from tests.entrypoints.serving_replay.replay_harness import (
    CHUNK_SIZES,
    assert_chat_completion_response,
    get_parser_cls,
    load_serving_samples,
    make_mock_tokenizer,
    tokens_to_request_outputs,
)
from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
)
from vllm.entrypoints.openai.engine.protocol import RequestResponseMetadata

_all_samples = (
    load_serving_samples("qwen3")
    + load_serving_samples("gemma4")
    + load_serving_samples("deepseek_v4")
)


def _build_request(sample, *, stream: bool = True) -> ChatCompletionRequest:
    return ChatCompletionRequest(
        model="test-model",
        messages=[{"role": "user", "content": "test"}],
        stream=stream,
        tools=sample.tools,
        tool_choice=sample.tool_choice if sample.tools else None,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("chunk_size", CHUNK_SIZES, ids=lambda c: f"chunk={c}")
@pytest.mark.parametrize("sample", _all_samples, ids=lambda s: s.id)
async def test_streaming(sample, chunk_size):
    tokenizer = make_mock_tokenizer(sample)
    serving_chat = build_serving_chat(get_parser_cls(sample.parser_name))
    request = _build_request(sample, stream=True)
    result_gen = tokens_to_request_outputs(sample.tokens, chunk_size)

    response = await accumulate_streaming_response(
        serving_chat.chat_completion_stream_generator(
            request=request,
            result_generator=result_gen,
            request_id="test-req",
            model_name="test-model",
            conversation=[],
            tokenizer=tokenizer,
            request_metadata=RequestResponseMetadata(request_id="test-req"),
            chat_template_kwargs=sample.chat_template_kwargs,
        )
    )
    assert_chat_completion_response(response, sample)
