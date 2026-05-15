# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the grammar-based Hermes JSON tool call parser."""

import json
from typing import Any
from unittest.mock import MagicMock

import pytest

from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
)
from vllm.grammar_parser.adapter import GrammarToolParser
from vllm.grammar_parser.grammars.hermes import hermes_config


@pytest.fixture
def mock_tokenizer():
    tokenizer = MagicMock()
    tokenizer.encode.return_value = [1, 2, 3]
    tokenizer.get_vocab.return_value = {
        "<tool_call>": 100,
        "</tool_call>": 101,
    }
    tokenizer.decode.side_effect = lambda ids: "".join(
        chr(i) if i < 128 else f"<{i}>" for i in ids
    )
    return tokenizer


@pytest.fixture
def parser(mock_tokenizer):
    return GrammarToolParser(
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


class TestStreaming:
    def _simulate_streaming(
        self,
        parser: GrammarToolParser,
        mock_request,
        chunks: list[str],
    ) -> list[tuple[Any, str]]:
        results: list[tuple[Any, str]] = []
        previous_text = ""
        previous_token_ids: list[int] = []

        for chunk in chunks:
            current_text = previous_text + chunk
            delta_token_ids: list[int] = [0]
            current_token_ids = previous_token_ids + delta_token_ids

            delta = parser.extract_tool_calls_streaming(
                previous_text=previous_text,
                current_text=current_text,
                delta_text=chunk,
                previous_token_ids=tuple(previous_token_ids),
                current_token_ids=tuple(current_token_ids),
                delta_token_ids=tuple(delta_token_ids),
                request=mock_request,
            )
            results.append((delta, current_text))
            previous_text = current_text
            previous_token_ids = list(current_token_ids)

        return results

    def _collect_arguments(self, results):
        args_text = ""
        for delta, _ in results:
            if delta and delta.tool_calls:
                for tc in delta.tool_calls:
                    if tc.function and tc.function.arguments:
                        args_text += tc.function.arguments
        return args_text

    def _collect_function_name(self, results):
        for delta, _ in results:
            if delta and delta.tool_calls:
                for tc in delta.tool_calls:
                    if tc.function and tc.function.name:
                        return tc.function.name
        return None

    def test_basic_streaming(self, parser, mock_request):
        chunks = [
            "<tool_call>",
            '{"name": "get_weather",',
            ' "arguments": {"city":',
            ' "Tokyo"',
            "}}",
            "</tool_call>",
        ]

        results = self._simulate_streaming(parser, mock_request, chunks)

        name = self._collect_function_name(results)
        assert name == "get_weather"

        args_text = self._collect_arguments(results)
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

        results = self._simulate_streaming(parser, mock_request, chunks)

        content_parts = []
        for delta, _ in results:
            if delta and delta.content:
                content_parts.append(delta.content)

        assert "".join(content_parts).strip().startswith("Let me check")

        name = self._collect_function_name(results)
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

        results = self._simulate_streaming(parser, mock_request, chunks)

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

        results = self._simulate_streaming(parser, mock_request, chunks)

        name = self._collect_function_name(results)
        assert name == "func"

        args_text = self._collect_arguments(results)
        assert args_text
        parsed = json.loads(args_text)
        assert parsed == {"key": "value"}
