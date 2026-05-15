# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Run the grammar-based Gemma4 tool parser through the existing Gemma4
test cases.

Mirrors the test structure from tests/tool_parsers/test_gemma4_tool_parser.py
but uses the grammar-based ``GrammarGemma4ToolParser``.
"""

import json
from typing import Any
from unittest.mock import MagicMock

import pytest

from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
)
from vllm.grammar_parser.grammars.gemma4 import TOOL_CALL_END, TOOL_CALL_START
from vllm.grammar_parser.registered_parsers import GrammarGemma4ToolParser


@pytest.fixture
def mock_tokenizer():
    tokenizer = MagicMock()
    tokenizer.encode.return_value = [1, 2, 3]
    tokenizer.get_vocab.return_value = {
        TOOL_CALL_START: 48,
        TOOL_CALL_END: 49,
    }
    return tokenizer


@pytest.fixture
def parser(mock_tokenizer):
    return GrammarGemma4ToolParser(mock_tokenizer)


@pytest.fixture
def mock_request():
    request = MagicMock(spec=ChatCompletionRequest)
    request.tools = []
    request.tool_choice = "auto"
    return request


class TestExtractToolCalls:
    def test_no_tool_calls(self, parser, mock_request):
        model_output = "Hello, how can I help you today?"
        result = parser.extract_tool_calls(model_output, mock_request)

        assert result.tools_called is False
        assert result.tool_calls == []
        assert result.content == model_output

    def test_single_tool_call(self, parser, mock_request):
        model_output = (
            '<|tool_call>call:get_weather{location:<|"|>London<|"|>}<tool_call|>'
        )
        result = parser.extract_tool_calls(model_output, mock_request)

        assert result.tools_called is True
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].function.name == "get_weather"
        args = json.loads(result.tool_calls[0].function.arguments)
        assert args == {"location": "London"}

    def test_multiple_arguments(self, parser, mock_request):
        model_output = (
            "<|tool_call>call:get_weather{"
            'location:<|"|>San Francisco<|"|>,'
            'unit:<|"|>celsius<|"|>}'
            "<tool_call|>"
        )
        result = parser.extract_tool_calls(model_output, mock_request)

        assert result.tools_called is True
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].function.name == "get_weather"
        args = json.loads(result.tool_calls[0].function.arguments)
        assert args == {"location": "San Francisco", "unit": "celsius"}

    def test_text_before_tool_call(self, parser, mock_request):
        model_output = (
            "Let me check the weather for you. "
            '<|tool_call>call:get_weather{location:<|"|>Paris<|"|>}'
            "<tool_call|>"
        )
        result = parser.extract_tool_calls(model_output, mock_request)

        assert result.tools_called is True
        assert result.content == "Let me check the weather for you."
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].function.name == "get_weather"

    def test_multiple_tool_calls(self, parser, mock_request):
        model_output = (
            '<|tool_call>call:get_weather{location:<|"|>London<|"|>}'
            "<tool_call|>"
            '<|tool_call>call:get_time{location:<|"|>London<|"|>}'
            "<tool_call|>"
        )
        result = parser.extract_tool_calls(model_output, mock_request)

        assert result.tools_called is True
        assert len(result.tool_calls) == 2
        assert result.tool_calls[0].function.name == "get_weather"
        assert result.tool_calls[1].function.name == "get_time"

    def test_nested_arguments(self, parser, mock_request):
        model_output = (
            "<|tool_call>call:complex_function{"
            'nested:{inner:<|"|>value<|"|>},'
            'list:[<|"|>a<|"|>,<|"|>b<|"|>]}'
            "<tool_call|>"
        )
        result = parser.extract_tool_calls(model_output, mock_request)

        assert result.tools_called is True
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].function.name == "complex_function"
        args = json.loads(result.tool_calls[0].function.arguments)
        assert args == {"nested": {"inner": "value"}, "list": ["a", "b"]}

    def test_tool_call_with_number_and_boolean(self, parser, mock_request):
        model_output = (
            "<|tool_call>call:set_status{"
            "is_active:true,"
            "count:42,"
            "score:3.14}"
            "<tool_call|>"
        )
        result = parser.extract_tool_calls(model_output, mock_request)

        assert result.tools_called is True
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].function.name == "set_status"
        args = json.loads(result.tool_calls[0].function.arguments)
        assert args == {"is_active": True, "count": 42, "score": 3.14}

    def test_no_arguments(self, parser, mock_request):
        model_output = "<|tool_call>call:get_status{}<tool_call|>"
        result = parser.extract_tool_calls(model_output, mock_request)

        assert result.tools_called is True
        assert result.tool_calls[0].function.name == "get_status"
        args = json.loads(result.tool_calls[0].function.arguments)
        assert args == {}


class TestStreamingExtraction:
    def _simulate_streaming(
        self,
        parser: GrammarGemma4ToolParser,
        mock_request: Any,
        chunks: list[str],
    ) -> list[tuple[Any, str]]:
        results: list[tuple[Any, str]] = []
        previous_text: str = ""
        previous_token_ids: list[int] = []

        for chunk in chunks:
            current_text = previous_text + chunk
            delta_token_ids: list[int] = []
            if TOOL_CALL_START in chunk:
                delta_token_ids.append(48)
            elif TOOL_CALL_END in chunk:
                delta_token_ids.append(49)
            else:
                delta_token_ids.append(0)

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
                    func = tc.function if isinstance(tc.function, dict) else tc.function
                    if isinstance(func, dict):
                        arg = func.get("arguments", "")
                    else:
                        arg = getattr(func, "arguments", "") or ""
                    if arg:
                        args_text += arg
        return args_text

    def _collect_function_name(self, results):
        for delta, _ in results:
            if delta and delta.tool_calls:
                for tc in delta.tool_calls:
                    func = tc.function if isinstance(tc.function, dict) else tc.function
                    if isinstance(func, dict):
                        name = func.get("name")
                    else:
                        name = getattr(func, "name", None)
                    if name:
                        return name
        return None

    def test_basic_streaming_single_tool(self, parser, mock_request):
        chunks = [
            "<|tool_call>",
            "call:get_weather{",
            'location:<|"|>Paris',
            ", France",
            '<|"|>}',
            "<tool_call|>",
        ]

        results = self._simulate_streaming(parser, mock_request, chunks)

        name = self._collect_function_name(results)
        assert name == "get_weather", f"Expected 'get_weather', got '{name}'"

        args_text = self._collect_arguments(results)
        assert args_text, "No arguments were streamed"
        parsed_args = json.loads(args_text)
        assert parsed_args == {"location": "Paris, France"}

    def test_streaming_multi_arg(self, parser, mock_request):
        chunks = [
            "<|tool_call>",
            "call:get_weather{",
            'location:<|"|>Tokyo<|"|>,',
            'unit:<|"|>celsius<|"|>}',
            "<tool_call|>",
        ]

        results = self._simulate_streaming(parser, mock_request, chunks)

        name = self._collect_function_name(results)
        assert name == "get_weather"

        args_text = self._collect_arguments(results)
        assert args_text
        parsed_args = json.loads(args_text)
        assert parsed_args == {"location": "Tokyo", "unit": "celsius"}

    def test_streaming_no_extra_brace(self, parser, mock_request):
        chunks = [
            "<|tool_call>",
            "call:get_weather{",
            'location:<|"|>London<|"|>}',
            "<tool_call|>",
        ]

        results = self._simulate_streaming(parser, mock_request, chunks)
        args_text = self._collect_arguments(results)
        assert args_text

        parsed = json.loads(args_text)
        assert parsed == {"location": "London"}
        assert args_text.count("}") <= 1

    def test_streaming_text_before_tool_call(self, parser, mock_request):
        chunks = [
            "Let me check ",
            "the weather. ",
            "<|tool_call>",
            "call:get_weather{",
            'location:<|"|>London<|"|>}',
            "<tool_call|>",
        ]

        results = self._simulate_streaming(parser, mock_request, chunks)

        content_parts = []
        for delta, _ in results:
            if delta and delta.content:
                content_parts.append(delta.content)

        assert "".join(content_parts).strip().startswith("Let me check")

    def test_streaming_numeric_args(self, parser, mock_request):
        chunks = [
            "<|tool_call>",
            "call:set_config{",
            "count:42,",
            "active:true}",
            "<tool_call|>",
        ]

        results = self._simulate_streaming(parser, mock_request, chunks)
        args_text = self._collect_arguments(results)
        if args_text:
            parsed_args = json.loads(args_text)
            assert parsed_args["count"] == 42
            assert parsed_args["active"] is True

    def test_streaming_empty_args(self, parser, mock_request):
        chunks = [
            "<|tool_call>",
            "call:get_status{}",
            "<tool_call|>",
        ]

        results = self._simulate_streaming(parser, mock_request, chunks)
        name = self._collect_function_name(results)
        assert name == "get_status"

    def test_streaming_split_delimiter_no_invalid_json(
        self,
        parser,
        mock_request,
    ):
        chunks = [
            "<|tool_call>",
            "call:todowrite{",
            'content:<|"|>Buy milk<|',
            '"|>}',
            "<tool_call|>",
        ]

        results = self._simulate_streaming(parser, mock_request, chunks)

        args_text = self._collect_arguments(results)
        assert args_text, "No arguments were streamed"

        parsed_args = json.loads(args_text)
        assert parsed_args["content"] == "Buy milk"

        assert "<|" not in args_text

    def test_streaming_trailing_bare_bool_not_duplicated(
        self,
        parser,
        mock_request,
    ):
        chunks = [
            "<|tool_call>",
            "call:Edit{",
            'file_path:<|"|>src/env.py<|"|>,',
            'old_string:<|"|>old_val<|"|>,',
            'new_string:<|"|>new_val<|"|>,',
            "replace_all:",
            "false}",
            "<tool_call|>",
        ]

        results = self._simulate_streaming(parser, mock_request, chunks)
        args_text = self._collect_arguments(results)
        assert args_text, "No arguments were streamed"

        parsed_args = json.loads(args_text)
        assert parsed_args == {
            "file_path": "src/env.py",
            "old_string": "old_val",
            "new_string": "new_val",
            "replace_all": False,
        }

        assert args_text.count("replace_all") == 1
