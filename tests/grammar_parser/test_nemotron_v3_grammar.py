# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the grammar-based Nemotron V3 parser.

Validates that ``NemotronV3GrammarParser`` correctly handles:
- ``<think>``/``</think>`` reasoning with ``<tool_call>`` XML tool calls
  (same format as Qwen3)
- Nemotron-specific reasoning/content swap when ``enable_thinking=False``
  or ``force_nonempty_content=True``
"""

import json
from unittest.mock import MagicMock

import pytest

from tests.grammar_parser.streaming_helpers import (
    collect_function_name,
    collect_tool_arguments,
    simulate_reasoning_streaming,
    simulate_tool_streaming,
)
from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
)
from vllm.grammar_parser.parsers import NemotronV3GrammarParser

_THINK_START_ID = 50
_THINK_END_ID = 51
_TOOL_CALL_ID = 60
_TOOL_CALL_END_ID = 61
_TEXT_ID = 100

_SPECIAL_TOKEN_TEXT = {
    _THINK_START_ID: "<think>",
    _THINK_END_ID: "</think>",
    _TOOL_CALL_ID: "<tool_call>",
    _TOOL_CALL_END_ID: "</tool_call>",
}


def _make_tokenizer():
    tokenizer = MagicMock()
    tokenizer.encode.return_value = [1, 2, 3]
    tokenizer.get_vocab.return_value = {
        "<think>": _THINK_START_ID,
        "</think>": _THINK_END_ID,
        "<tool_call>": _TOOL_CALL_ID,
        "</tool_call>": _TOOL_CALL_END_ID,
    }
    tokenizer.decode.side_effect = lambda ids: "".join(
        _SPECIAL_TOKEN_TEXT.get(i, chr(i) if i < 128 else f"<{i}>") for i in ids
    )
    return tokenizer


def _make_request(**chat_template_kwargs):
    request = MagicMock(spec=ChatCompletionRequest)
    request.tools = []
    request.tool_choice = "auto"
    request.chat_template_kwargs = chat_template_kwargs or None
    return request


@pytest.fixture
def parser():
    return NemotronV3GrammarParser(_make_tokenizer())


class TestNonStreamingReasoning:
    def test_reasoning_then_content(self, parser):
        text = "Let me analyze.</think>The answer is 42."
        request = _make_request()
        reasoning, content = parser.extract_reasoning(text, request)
        assert reasoning == "Let me analyze."
        assert content == "The answer is 42."

    def test_reasoning_only(self, parser):
        text = "Still thinking..."
        request = _make_request()
        reasoning, content = parser.extract_reasoning(text, request)
        assert reasoning == "Still thinking..."
        assert content is None

    def test_with_think_tags(self, parser):
        text = "<think>Let me analyze.</think>The answer is 42."
        request = _make_request()
        reasoning, content = parser.extract_reasoning(text, request)
        assert reasoning == "Let me analyze."
        assert content == "The answer is 42."

    def test_tool_call_implicit_end(self, parser):
        text = (
            "I need to read the file.\n\n"
            "<tool_call>\n<function=bash>\n"
            "<parameter=cmd>ls</parameter>\n"
            "</function>\n</tool_call>"
        )
        request = _make_request()
        reasoning, content = parser.extract_reasoning(text, request)
        assert reasoning == "I need to read the file.\n\n"
        assert "<tool_call>" not in reasoning


class TestNemotronSwap:
    def test_enable_thinking_false_swaps(self, parser):
        """When enable_thinking=False, model output without think tags
        should have reasoning swapped to content."""
        text = "The answer is 42."
        request = _make_request(enable_thinking=False)
        reasoning, content = parser.extract_reasoning(text, request)
        assert content == "The answer is 42."
        assert reasoning is None

    def test_force_nonempty_content_swaps(self, parser):
        """force_nonempty_content=True triggers swap when content empty."""
        text = "The answer is 42."
        request = _make_request(force_nonempty_content=True)
        reasoning, content = parser.extract_reasoning(text, request)
        assert content == "The answer is 42."
        assert reasoning is None

    def test_no_swap_when_content_exists(self, parser):
        """With enable_thinking=False but real </think> giving content,
        no swap occurs."""
        text = "Some reasoning.</think>Actual content here."
        request = _make_request(enable_thinking=False)
        reasoning, content = parser.extract_reasoning(text, request)
        assert reasoning == "Some reasoning."
        assert content == "Actual content here."

    def test_no_swap_when_enable_thinking_true(self, parser):
        """Normal thinking mode: no swap, even when content is empty."""
        text = "Still thinking..."
        request = _make_request(enable_thinking=True)
        reasoning, content = parser.extract_reasoning(text, request)
        assert reasoning == "Still thinking..."
        assert content is None

    def test_no_swap_with_none_request(self, parser):
        """Graceful handling when request is None."""
        text = "Some text."
        reasoning, content = parser.extract_reasoning(text, None)
        assert reasoning == "Some text."
        assert content is None

    def test_no_swap_with_no_kwargs(self, parser):
        """No swap when chat_template_kwargs is absent."""
        text = "Some text."
        request = _make_request()
        reasoning, content = parser.extract_reasoning(text, request)
        assert reasoning == "Some text."
        assert content is None

    def test_swap_with_whitespace_only_content(self, parser):
        """Swap occurs when content is whitespace-only."""
        text = "The answer.</think>   "
        request = _make_request(enable_thinking=False)
        reasoning, content = parser.extract_reasoning(text, request)
        assert content == "The answer."
        assert reasoning == "   "


class TestIsReasoningEnd:
    def test_think_end_token(self, parser):
        assert parser.is_reasoning_end([_THINK_START_ID, 1, _THINK_END_ID])

    def test_no_end_token(self, parser):
        assert not parser.is_reasoning_end([_THINK_START_ID, 1, 2])

    def test_tool_call_as_implicit_end(self, parser):
        assert parser.is_reasoning_end([_THINK_START_ID, 1, _TOOL_CALL_ID])

    def test_paired_tool_call_not_end(self, parser):
        assert not parser.is_reasoning_end(
            [_THINK_START_ID, 1, _TOOL_CALL_ID, 2, _TOOL_CALL_END_ID]
        )

    def test_empty_ids(self, parser):
        assert not parser.is_reasoning_end([])


class TestNonStreamingToolCalls:
    def test_single_tool_call(self, parser):
        text = (
            "<tool_call>\n"
            "<function=get_weather>\n"
            "<parameter=city>Tokyo</parameter>\n"
            "</function>\n"
            "</tool_call>"
        )
        request = _make_request()
        result = parser.extract_tool_calls(text, request)
        assert result.tools_called is True
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].function.name == "get_weather"
        args = json.loads(result.tool_calls[0].function.arguments)
        assert args == {"city": "Tokyo"}

    def test_parallel_tool_calls(self, parser):
        text = (
            "<tool_call>\n"
            "<function=get_weather>\n"
            "<parameter=city>Tokyo</parameter>\n"
            "</function>\n"
            "</tool_call>"
            "<tool_call>\n"
            "<function=get_time>\n"
            "<parameter=timezone>Asia/Tokyo</parameter>\n"
            "</function>\n"
            "</tool_call>"
        )
        request = _make_request()
        result = parser.extract_tool_calls(text, request)
        assert result.tools_called is True
        assert len(result.tool_calls) == 2
        assert result.tool_calls[0].function.name == "get_weather"
        assert result.tool_calls[1].function.name == "get_time"

    def test_no_tool_calls(self, parser):
        request = _make_request()
        result = parser.extract_tool_calls("Hello, how can I help?", request)
        assert result.tools_called is False
        # Parser starts in REASONING state, so plain text is classified
        # as reasoning (not content) when there are no tool calls.
        assert result.content is None


class TestStreaming:
    def test_basic_streaming_reasoning(self, parser):
        reasoning, content = simulate_reasoning_streaming(
            parser,
            ["thinking", " hard", "</think>", "done"],
            [
                (1,),
                (2,),
                (_THINK_END_ID,),
                (3,),
            ],
        )
        assert reasoning == "thinking hard"
        assert content == "done"

    def test_streaming_tool_call_implicit_end(self, parser):
        reasoning, content = simulate_reasoning_streaming(
            parser,
            ["I need to check.", "<tool_call>", "\n<function=test>"],
            [
                (1,),
                (_TOOL_CALL_ID,),
                (2,),
            ],
        )
        assert reasoning == "I need to check."
        assert "<tool_call>" not in reasoning

    def test_streaming_tool_calls(self, parser):
        request = _make_request()
        chunks = [
            "<tool_call>\n",
            "<function=get_weather>\n",
            "<parameter=city>Tokyo",
            "</parameter>\n",
            "</function>\n",
            "</tool_call>",
        ]
        results = simulate_tool_streaming(parser, request, chunks)
        name = collect_function_name(results)
        assert name == "get_weather"
        args_text = collect_tool_arguments(results)
        assert args_text
        parsed = json.loads(args_text)
        assert parsed == {"city": "Tokyo"}
