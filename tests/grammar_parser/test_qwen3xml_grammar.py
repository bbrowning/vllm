# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the grammar-based Qwen3 XML tool call parser.

These mirror the test cases from test_qwen3xml_tool_parser.py to validate
that the grammar-driven parser produces identical results.
"""

import json
from typing import Any
from unittest.mock import MagicMock

import pytest

from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
)
from vllm.grammar_parser.adapter import GrammarToolParser
from vllm.grammar_parser.grammars.qwen3xml import (
    TOOL_CALL_END,
    TOOL_CALL_START,
    qwen3xml_config,
)


@pytest.fixture
def mock_tokenizer():
    tokenizer = MagicMock()
    tokenizer.encode.return_value = [1, 2, 3]
    tokenizer.get_vocab.return_value = {
        TOOL_CALL_START: 100,
        TOOL_CALL_END: 101,
    }
    tokenizer.decode.side_effect = lambda ids: "".join(
        chr(i) if i < 128 else f"<{i}>" for i in ids
    )
    return tokenizer


@pytest.fixture
def parser(mock_tokenizer):
    return GrammarToolParser(
        mock_tokenizer,
        grammar_config=qwen3xml_config(),
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
            "This is a regular response without any tool calls.",
            mock_request,
        )
        assert result.tools_called is False
        assert result.tool_calls == []
        assert result.content == ("This is a regular response without any tool calls.")

    def test_single_tool_call(self, parser, mock_request):
        text = (
            "<tool_call>\n"
            "<function=get_weather>\n"
            "<parameter=city>Tokyo</parameter>\n"
            "</function>\n"
            "</tool_call>"
        )
        result = parser.extract_tool_calls(text, mock_request)

        assert result.tools_called is True
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].function.name == "get_weather"
        args = json.loads(result.tool_calls[0].function.arguments)
        assert args == {"city": "Tokyo"}

    def test_parallel_tool_calls(self, parser, mock_request):
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
        result = parser.extract_tool_calls(text, mock_request)

        assert result.tools_called is True
        assert len(result.tool_calls) == 2
        assert result.tool_calls[0].function.name == "get_weather"
        assert result.tool_calls[1].function.name == "get_time"

        args0 = json.loads(result.tool_calls[0].function.arguments)
        assert args0 == {"city": "Tokyo"}
        args1 = json.loads(result.tool_calls[1].function.arguments)
        assert args1 == {"timezone": "Asia/Tokyo"}

    def test_various_data_types(self, parser, mock_request):
        text = (
            "<tool_call>\n<function=test_function>\n"
            "<parameter=string_field>hello</parameter>\n"
            "<parameter=int_field>42</parameter>\n"
            "<parameter=float_field>3.14</parameter>\n"
            "<parameter=bool_field>true</parameter>\n"
            "<parameter=null_field>null</parameter>\n"
            '<parameter=array_field>["a", "b", "c"]</parameter>\n'
            '<parameter=object_field>{"nested": "value"}</parameter>\n'
            "</function>\n</tool_call>"
        )
        result = parser.extract_tool_calls(text, mock_request)

        assert result.tools_called is True
        args = json.loads(result.tool_calls[0].function.arguments)
        assert args["string_field"] == "hello"
        assert args["int_field"] == 42
        assert args["float_field"] == 3.14
        assert args["bool_field"] is True
        assert args["null_field"] is None
        assert args["array_field"] == ["a", "b", "c"]
        assert args["object_field"] == {"nested": "value"}

    def test_empty_arguments(self, parser, mock_request):
        text = "<tool_call>\n<function=refresh>\n</function>\n</tool_call>"
        result = parser.extract_tool_calls(text, mock_request)

        assert result.tools_called is True
        assert result.tool_calls[0].function.name == "refresh"
        args = json.loads(result.tool_calls[0].function.arguments)
        assert args == {}

    def test_surrounding_text(self, parser, mock_request):
        text = (
            "Let me check the weather for you.\n\n"
            "<tool_call>\n<function=get_weather>\n"
            "<parameter=city>Tokyo</parameter>\n"
            "</function>\n</tool_call>\n\n"
            "I will get that information."
        )
        result = parser.extract_tool_calls(text, mock_request)

        assert result.tools_called is True
        assert result.content is not None
        assert "Let me check the weather" in result.content
        assert result.tool_calls[0].function.name == "get_weather"

    def test_escaped_strings(self, parser, mock_request):
        text = (
            "<tool_call>\n<function=test_function>\n"
            '<parameter=quoted>He said "hello"</parameter>\n'
            "<parameter=path>C:\\Users\\file.txt</parameter>\n"
            "<parameter=newline>line1\nline2</parameter>\n"
            "</function>\n</tool_call>"
        )
        result = parser.extract_tool_calls(text, mock_request)

        assert result.tools_called is True
        args = json.loads(result.tool_calls[0].function.arguments)
        assert args["quoted"] == 'He said "hello"'
        assert args["path"] == "C:\\Users\\file.txt"
        assert args["newline"] == "line1\nline2"

    def test_multiple_parameters(self, parser, mock_request):
        text = (
            "<tool_call>\n<function=search>\n"
            "<parameter=query>vllm parsing</parameter>\n"
            "<parameter=limit>10</parameter>\n"
            "<parameter=exact_match>false</parameter>\n"
            "</function>\n</tool_call>"
        )
        result = parser.extract_tool_calls(text, mock_request)

        assert result.tools_called is True
        args = json.loads(result.tool_calls[0].function.arguments)
        assert args == {
            "query": "vllm parsing",
            "limit": 10,
            "exact_match": False,
        }


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
            "<tool_call>\n",
            "<function=get_weather>\n",
            "<parameter=city>Tokyo",
            "</parameter>\n",
            "</function>\n",
            "</tool_call>",
        ]

        results = self._simulate_streaming(parser, mock_request, chunks)

        name = self._collect_function_name(results)
        assert name == "get_weather"

        args_text = self._collect_arguments(results)
        assert args_text
        parsed = json.loads(args_text)
        assert parsed == {"city": "Tokyo"}

    def test_streaming_multi_param(self, parser, mock_request):
        chunks = [
            "<tool_call>\n",
            "<function=get_weather>\n",
            "<parameter=city>Tokyo</parameter>\n",
            "<parameter=unit>celsius</parameter>\n",
            "</function>\n",
            "</tool_call>",
        ]

        results = self._simulate_streaming(parser, mock_request, chunks)

        name = self._collect_function_name(results)
        assert name == "get_weather"

        args_text = self._collect_arguments(results)
        assert args_text
        parsed = json.loads(args_text)
        assert parsed == {"city": "Tokyo", "unit": "celsius"}

    def test_streaming_text_before_tool(self, parser, mock_request):
        chunks = [
            "Let me check ",
            "the weather. ",
            "<tool_call>\n",
            "<function=get_weather>\n",
            "<parameter=city>Tokyo</parameter>\n",
            "</function>\n",
            "</tool_call>",
        ]

        results = self._simulate_streaming(parser, mock_request, chunks)

        content_parts = []
        for delta, _ in results:
            if delta and delta.content:
                content_parts.append(delta.content)

        assert "".join(content_parts).strip().startswith("Let me check")

    def test_streaming_empty_args(self, parser, mock_request):
        chunks = [
            "<tool_call>\n",
            "<function=refresh>\n",
            "</function>\n",
            "</tool_call>",
        ]

        results = self._simulate_streaming(parser, mock_request, chunks)

        name = self._collect_function_name(results)
        assert name == "refresh"

    def test_streaming_split_parameter_tag(self, parser, mock_request):
        """Parameter tag split across chunks."""
        chunks = [
            "<tool_call>\n",
            "<function=test>\n",
            "<parameter=",
            "name>Alice",
            "</parameter>\n",
            "</function>\n",
            "</tool_call>",
        ]

        results = self._simulate_streaming(parser, mock_request, chunks)

        name = self._collect_function_name(results)
        assert name == "test"

        args_text = self._collect_arguments(results)
        assert args_text
        parsed = json.loads(args_text)
        assert parsed["name"] == "Alice"

    def test_streaming_numeric_values(self, parser, mock_request):
        chunks = [
            "<tool_call>\n",
            "<function=set_config>\n",
            "<parameter=count>42</parameter>\n",
            "<parameter=active>true</parameter>\n",
            "</function>\n",
            "</tool_call>",
        ]

        results = self._simulate_streaming(parser, mock_request, chunks)
        args_text = self._collect_arguments(results)
        if args_text:
            parsed = json.loads(args_text)
            assert parsed["count"] == 42
            assert parsed["active"] is True

    def test_streaming_parallel_calls(self, parser, mock_request):
        chunks = [
            "<tool_call>\n",
            "<function=get_weather>\n",
            "<parameter=city>Tokyo</parameter>\n",
            "</function>\n",
            "</tool_call>",
            "<tool_call>\n",
            "<function=get_time>\n",
            "<parameter=tz>JST</parameter>\n",
            "</function>\n",
            "</tool_call>",
        ]

        results = self._simulate_streaming(parser, mock_request, chunks)

        names = []
        for delta, _ in results:
            if delta and delta.tool_calls:
                for tc in delta.tool_calls:
                    if tc.function and tc.function.name:
                        names.append(tc.function.name)

        assert "get_weather" in names
        assert "get_time" in names

    def test_streaming_value_split_across_chunks(self, parser, mock_request):
        """Parameter value split across multiple chunks."""
        chunks = [
            "<tool_call>\n",
            "<function=search>\n",
            "<parameter=query>hello ",
            "world",
            " test</parameter>\n",
            "</function>\n",
            "</tool_call>",
        ]

        results = self._simulate_streaming(parser, mock_request, chunks)

        args_text = self._collect_arguments(results)
        assert args_text
        parsed = json.loads(args_text)
        assert parsed["query"] == "hello world test"
