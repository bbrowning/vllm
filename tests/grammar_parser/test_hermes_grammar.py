# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the grammar-based Hermes JSON tool call parser."""

import json
from unittest.mock import MagicMock

import pytest

from tests.grammar_parser.streaming_helpers import (
    collect_function_name,
    collect_tool_arguments,
    simulate_tool_streaming,
)
from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
)
from vllm.grammar_parser.grammars.hermes import hermes_config
from vllm.grammar_parser.unified_parser import GrammarParser

_SPECIAL_DECODE = {100: "<tool_call>", 101: "</tool_call>"}


@pytest.fixture
def mock_tokenizer():
    tokenizer = MagicMock()
    tokenizer.encode.return_value = [1, 2, 3]
    tokenizer.get_vocab.return_value = {
        "<tool_call>": 100,
        "</tool_call>": 101,
    }
    tokenizer.decode.side_effect = lambda ids: "".join(
        _SPECIAL_DECODE.get(i, chr(i) if i < 128 else f"<{i}>") for i in ids
    )
    return tokenizer


@pytest.fixture
def parser(mock_tokenizer):
    return GrammarParser(
        mock_tokenizer,
        grammar_config=hermes_config(),
    )


@pytest.fixture
def mock_request():
    request = MagicMock(spec=ChatCompletionRequest)
    request.tools = []
    request.tool_choice = "auto"
    return request


class TestNonStreaming:
    def test_no_tool_calls(self, parser, mock_request):
        result = parser.extract_tool_calls(
            "Hello, how can I help?",
            mock_request,
        )
        assert result.tools_called is False
        assert result.content == "Hello, how can I help?"

    def test_single_tool_call(self, parser, mock_request):
        text = (
            '<tool_call>{"name": "get_weather", '
            '"arguments": {"city": "Tokyo"}}</tool_call>'
        )
        result = parser.extract_tool_calls(text, mock_request)

        assert result.tools_called is True
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].function.name == "get_weather"
        args = json.loads(result.tool_calls[0].function.arguments)
        assert args == {"city": "Tokyo"}

    def test_parameters_key(self, parser, mock_request):
        text = (
            '<tool_call>{"name": "search", "parameters": {"query": "test"}}</tool_call>'
        )
        result = parser.extract_tool_calls(text, mock_request)

        assert result.tools_called is True
        assert result.tool_calls[0].function.name == "search"
        args = json.loads(result.tool_calls[0].function.arguments)
        assert args == {"query": "test"}

    def test_multiple_tool_calls(self, parser, mock_request):
        text = (
            '<tool_call>{"name": "get_weather", '
            '"arguments": {"city": "Tokyo"}}</tool_call>'
            '<tool_call>{"name": "get_time", '
            '"arguments": {"tz": "JST"}}</tool_call>'
        )
        result = parser.extract_tool_calls(text, mock_request)

        assert result.tools_called is True
        assert len(result.tool_calls) == 2
        assert result.tool_calls[0].function.name == "get_weather"
        assert result.tool_calls[1].function.name == "get_time"

    def test_text_before_tool_call(self, parser, mock_request):
        text = (
            "Let me check the weather. "
            '<tool_call>{"name": "get_weather", '
            '"arguments": {"city": "Paris"}}</tool_call>'
        )
        result = parser.extract_tool_calls(text, mock_request)

        assert result.tools_called is True
        assert result.content is not None
        assert "Let me check" in result.content
        assert result.tool_calls[0].function.name == "get_weather"

    def test_nested_arguments(self, parser, mock_request):
        text = (
            '<tool_call>{"name": "complex", "arguments": '
            '{"nested": {"a": 1}, "list": [1, 2, 3]}}</tool_call>'
        )
        result = parser.extract_tool_calls(text, mock_request)

        assert result.tools_called is True
        args = json.loads(result.tool_calls[0].function.arguments)
        assert args == {"nested": {"a": 1}, "list": [1, 2, 3]}

    def test_empty_arguments(self, parser, mock_request):
        text = '<tool_call>{"name": "refresh", "arguments": {}}</tool_call>'
        result = parser.extract_tool_calls(text, mock_request)

        assert result.tools_called is True
        assert result.tool_calls[0].function.name == "refresh"
        args = json.loads(result.tool_calls[0].function.arguments)
        assert args == {}

    def test_various_data_types(self, parser, mock_request):
        text = (
            '<tool_call>{"name": "multi_type", "arguments": '
            '{"s": "hello", "i": 42, "f": 3.14, "b": true, '
            '"n": null, "a": [1, "two"], '
            '"o": {"nested": true}}}</tool_call>'
        )
        result = parser.extract_tool_calls(text, mock_request)

        assert result.tools_called is True
        args = json.loads(result.tool_calls[0].function.arguments)
        assert args == {
            "s": "hello",
            "i": 42,
            "f": 3.14,
            "b": True,
            "n": None,
            "a": [1, "two"],
            "o": {"nested": True},
        }

    def test_escaped_strings(self, parser, mock_request):
        text = (
            '<tool_call>{"name": "write", "arguments": '
            '{"path": "C:\\\\Users\\\\test\\\\file.txt", '
            '"content": "line1\\nline2\\n", '
            '"label": "say \\"hello\\""}}</tool_call>'
        )
        result = parser.extract_tool_calls(text, mock_request)

        assert result.tools_called is True
        args = json.loads(result.tool_calls[0].function.arguments)
        assert args["path"] == "C:\\Users\\test\\file.txt"
        assert args["content"] == "line1\nline2\n"
        assert args["label"] == 'say "hello"'


class TestStreaming:
    def test_basic_streaming(self, parser, mock_request):
        chunks = [
            "<tool_call>",
            '{"name": "get_weather",',
            ' "arguments": {"city":',
            ' "Tokyo"',
            "}}",
            "</tool_call>",
        ]

        results = simulate_tool_streaming(parser, mock_request, chunks)

        name = collect_function_name(results)
        assert name == "get_weather"

        args_text = collect_tool_arguments(results)
        assert args_text
        parsed = json.loads(args_text)
        assert parsed == {"city": "Tokyo"}

    def test_streaming_text_before_tool(self, parser, mock_request):
        chunks = [
            "Let me check. ",
            "<tool_call>",
            '{"name": "search",',
            ' "arguments": {"q": "test"}}',
            "</tool_call>",
        ]

        results = simulate_tool_streaming(parser, mock_request, chunks)

        content_parts = []
        for delta, _ in results:
            if delta and delta.content:
                content_parts.append(delta.content)

        assert "".join(content_parts).strip().startswith("Let me check")

        name = collect_function_name(results)
        assert name == "search"

    def test_streaming_multiple_calls(self, parser, mock_request):
        chunks = [
            "<tool_call>",
            '{"name": "a", "arguments": {"x": 1}}',
            "</tool_call>",
            "<tool_call>",
            '{"name": "b", "arguments": {"y": 2}}',
            "</tool_call>",
        ]

        results = simulate_tool_streaming(parser, mock_request, chunks)

        names = []
        for delta, _ in results:
            if delta and delta.tool_calls:
                for tc in delta.tool_calls:
                    if tc.function and tc.function.name:
                        names.append(tc.function.name)

        assert "a" in names
        assert "b" in names

    def test_streaming_incremental_json(self, parser, mock_request):
        chunks = [
            "<tool_call>",
            '{"na',
            'me": "func",',
            ' "arguments": {',
            '"key": "val',
            'ue"}}',
            "</tool_call>",
        ]

        results = simulate_tool_streaming(parser, mock_request, chunks)

        name = collect_function_name(results)
        assert name == "func"

        args_text = collect_tool_arguments(results)
        assert args_text
        parsed = json.loads(args_text)
        assert parsed == {"key": "value"}
