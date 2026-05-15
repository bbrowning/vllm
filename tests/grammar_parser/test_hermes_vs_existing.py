# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Run the grammar-based Hermes tool parser through the common test suite
and key Hermes-specific test cases from the existing test file."""

import json
from typing import Any
from unittest.mock import MagicMock

import pytest

from tests.tool_parsers.common_tests import ToolParserTestConfig, ToolParserTests
from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
)
from vllm.grammar_parser.registered_parsers import GrammarHermesToolParser


class TestGrammarHermesCommonSuite(ToolParserTests):
    @pytest.fixture
    def test_config(self) -> ToolParserTestConfig:
        return ToolParserTestConfig(
            parser_name="hermes_grammar",
            no_tool_calls_output=(
                "This is some prior text that has nothing to do with tool calling."
            ),
            single_tool_call_output=(
                '<tool_call>{"name": "get_weather", '
                '"arguments": {"city": "Tokyo"}}</tool_call>'
            ),
            parallel_tool_calls_output=(
                '<tool_call>{"name": "get_weather", '
                '"arguments": {"city": "Tokyo"}}</tool_call>'
                '<tool_call>{"name": "get_time", '
                '"arguments": {"timezone": "Asia/Tokyo"}}</tool_call>'
            ),
            various_data_types_output=(
                '<tool_call>{"name": "test_function", "arguments": '
                '{"string_field": "hello", "int_field": 42, '
                '"float_field": 3.14, "bool_field": true, '
                '"null_field": null, "array_field": ["a", "b", "c"], '
                '"object_field": {"nested": "value"}}}</tool_call>'
            ),
            empty_arguments_output=(
                '<tool_call>{"name": "refresh", "arguments": {}}</tool_call>'
            ),
            surrounding_text_output=(
                "Let me check the weather for you."
                '<tool_call>{"name": "get_weather", '
                '"arguments": {"city": "Tokyo"}}</tool_call>'
                "I will get that information."
            ),
            escaped_strings_output=(
                '<tool_call>{"name": "test_function", "arguments": '
                '{"quoted": "He said \\"hello\\"", '
                '"path": "C:\\\\Users\\\\file.txt", '
                '"newline": "line1\\nline2"}}</tool_call>'
            ),
            malformed_input_outputs=[
                '<tool_call>{"name": "func"',
                "<tool_call>not json</tool_call>",
            ],
            single_tool_call_expected_name="get_weather",
            single_tool_call_expected_args={"city": "Tokyo"},
            parallel_tool_calls_count=2,
            parallel_tool_calls_names=["get_weather", "get_time"],
        )


class TestGrammarHermesSpecific:
    """Hermes-specific tests mirroring the existing test file."""

    @pytest.fixture
    def mock_tokenizer(self):
        tokenizer = MagicMock()
        tokenizer.encode.return_value = [1, 2, 3]
        tokenizer.get_vocab.return_value = {
            "<tool_call>": 100,
            "</tool_call>": 101,
        }
        tokenizer.tokenize.return_value = []
        return tokenizer

    @pytest.fixture
    def parser(self, mock_tokenizer):
        return GrammarHermesToolParser(mock_tokenizer)

    @pytest.fixture
    def mock_request(self):
        request = MagicMock(spec=ChatCompletionRequest)
        request.tools = []
        request.tool_choice = "auto"
        return request

    def _simulate_streaming(
        self,
        parser: GrammarHermesToolParser,
        mock_request: Any,
        chunks: list[str],
    ) -> list:
        results = []
        previous_text = ""
        previous_token_ids: list[int] = []

        for chunk in chunks:
            current_text = previous_text + chunk
            delta_token_ids: list[int] = []
            if "<tool_call>" in chunk and "</tool_call>" not in chunk:
                delta_token_ids.append(100)
            elif "</tool_call>" in chunk:
                delta_token_ids.append(101)
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
            if delta is not None:
                results.append(delta)
            previous_text = current_text
            previous_token_ids = list(current_token_ids)

        return results

    def test_non_streaming_tool_call_between_tags(
        self,
        parser,
        mock_request,
    ):
        text = (
            "<tool_call>\n"
            '{"name": "final_answer", '
            '"arguments": {"trigger": true}}\n'
            "</tool_call>"
        )
        result = parser.extract_tool_calls(text, mock_request)

        assert result.tools_called
        assert result.tool_calls[0].function.name == "final_answer"
        assert result.tool_calls[0].function.arguments == '{"trigger": true}'

    def test_non_streaming_no_tool_call(self, parser, mock_request):
        text = "This is not a tool call."
        result = parser.extract_tool_calls(text, mock_request)

        assert not result.tools_called

    def test_streaming_basic(self, parser, mock_request):
        chunks = [
            "<tool_call>",
            '\n{"name": "get_current_temperature",',
            '"arguments": {"location":',
            '"San Francisco, California", "unit": "celsius"}}',
            "\n</tool_call>",
        ]

        deltas = self._simulate_streaming(parser, mock_request, chunks)

        tool_deltas = [tc for d in deltas if d.tool_calls for tc in d.tool_calls]
        assert tool_deltas
        assert tool_deltas[0].function.name == "get_current_temperature"

        args_str = "".join(tc.function.arguments or "" for tc in tool_deltas)
        assert json.loads(args_str) == {
            "location": "San Francisco, California",
            "unit": "celsius",
        }

    def test_streaming_content_then_tool_call(self, parser, mock_request):
        chunks = [
            "Sure, let me check the weather.",
            "<tool_call>",
            '{"name": "get_weather", ',
            '"arguments": {"city": "NYC"}}',
            "</tool_call>",
        ]

        deltas = self._simulate_streaming(parser, mock_request, chunks)

        content_parts = [d.content for d in deltas if d.content]
        tool_deltas = [tc for d in deltas if d.tool_calls for tc in d.tool_calls]

        content_str = "".join(content_parts)
        assert "Sure, let me check the weather." in content_str

        assert tool_deltas[0].function.name == "get_weather"
        args_str = "".join(tc.function.arguments or "" for tc in tool_deltas)
        assert json.loads(args_str) == {"city": "NYC"}

    def test_streaming_multiple_tool_calls(self, parser, mock_request):
        chunks = [
            "<tool_call>",
            '{"name": "search", ',
            '"arguments": {"q": "cats"}}',
            "</tool_call>",
            "<tool_call>",
            '{"name": "search", ',
            '"arguments": {"q": "dogs"}}',
            "</tool_call>",
        ]

        deltas = self._simulate_streaming(parser, mock_request, chunks)

        all_tool_calls = [tc for d in deltas if d.tool_calls for tc in d.tool_calls]

        tool0 = [tc for tc in all_tool_calls if tc.index == 0]
        tool1 = [tc for tc in all_tool_calls if tc.index == 1]

        assert tool0[0].function.name == "search"
        args0 = "".join(tc.function.arguments or "" for tc in tool0)
        assert json.loads(args0) == {"q": "cats"}

        assert tool1[0].function.name == "search"
        args1 = "".join(tc.function.arguments or "" for tc in tool1)
        assert json.loads(args1) == {"q": "dogs"}
