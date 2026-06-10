# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Replay tests for DelegatingParser with engine adapters and old parsers.

Tests the same scenarios and chunk sizes as the former serving_replay tests,
but asserts on DeltaMessage/ParseOutput directly instead of routing through
the serving layer's HTTP response formatting.
"""

from __future__ import annotations

from functools import lru_cache

import pytest
from pydantic import TypeAdapter

from tests.parser.engine.replay_harness import (
    assert_parse_output,
    collect_output,
    make_mock_tokenizer,
    replay_streaming,
)
from tests.parser.engine.trace_builder import build_samples
from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionToolsParam,
)
from vllm.parser.abstract_parser import Parser
from vllm.parser.parser_manager import ParserManager

_TOOLS_VALIDATOR = TypeAdapter(list[ChatCompletionToolsParam])

_DELEGATING_OLD_PAIRINGS: dict[str, tuple[str, str]] = {
    "deepseek_v4_engine": ("deepseek_v4", "deepseek_v4"),
    "qwen3_engine": ("qwen3_xml", "qwen3"),
    "gemma4_engine": ("gemma4", "gemma4"),
}

_DELEGATING_ENGINE_PAIRINGS: dict[str, tuple[str, str]] = {
    "deepseek_v4_engine": ("deepseek_v4_engine", "deepseek_v4_engine"),
    "qwen3_engine": ("qwen3_xml_engine", "qwen3_engine"),
    "gemma4_engine": ("gemma4_engine", "gemma4_engine"),
}

CHUNK_SIZES = [1, 2, 3, 5, 11, 23, None]

DELEGATING_OLD_XFAIL_SAMPLES: frozenset[str] = frozenset(
    {
        "deepseek_v4-think-whitespace-tool",
        "deepseek_v4-whitespace-before-tool",
        "gemma4-complex-json-args",
        "gemma4-empty-reasoning-content",
        "gemma4-think-content-tool",
        "gemma4-think-then-content",
        "gemma4-think-then-parallel-tools",
        "gemma4-think-then-tool",
        "gemma4-think-whitespace-tool",
        "gemma4-tool-only",
        "gemma4-whitespace-before-tool",
        "qwen3-think-then-parallel-tools",
        "qwen3-think-whitespace-tool",
        "qwen3-whitespace-before-tool",
    }
)


@lru_cache
def _get_delegating_parser_cls(parser_name: str, pairings: str = "old") -> type[Parser]:
    table = (
        _DELEGATING_ENGINE_PAIRINGS
        if pairings == "engine"
        else _DELEGATING_OLD_PAIRINGS
    )
    tool_name, reasoning_name = table[parser_name]
    parser_cls = ParserManager.get_parser(
        tool_parser_name=tool_name,
        reasoning_parser_name=reasoning_name,
        enable_auto_tools=True,
    )
    assert parser_cls is not None
    return parser_cls


_all_samples = (
    build_samples("qwen3") + build_samples("gemma4") + build_samples("deepseek_v4")
)


@pytest.mark.parametrize(
    "pairings",
    ["old", "engine"],
    ids=lambda p: f"mode={p}",
)
@pytest.mark.parametrize("chunk_size", CHUNK_SIZES, ids=lambda c: f"chunk={c}")
@pytest.mark.parametrize("sample", _all_samples, ids=lambda s: s.id)
def test_delegating_replay(sample, chunk_size, pairings):
    if pairings == "old" and sample.id in DELEGATING_OLD_XFAIL_SAMPLES:
        pytest.xfail("old delegating parser has streaming differences")

    parser_name = f"{sample.id.split('-', 1)[0]}_engine"
    parser_cls = _get_delegating_parser_cls(parser_name, pairings=pairings)

    tokenizer = make_mock_tokenizer(sample)
    validated_tools = (
        _TOOLS_VALIDATOR.validate_python(sample.tools) if sample.tools else None
    )
    parser = parser_cls(
        tokenizer,
        validated_tools,
        chat_template_kwargs=sample.chat_template_kwargs,
    )

    deltas = replay_streaming(
        parser,
        sample.tokens,
        chunk_size=chunk_size,
        finished_on_last=True,
        tools=sample.tools,
    )
    output = collect_output(deltas)
    assert_parse_output(output, sample)
