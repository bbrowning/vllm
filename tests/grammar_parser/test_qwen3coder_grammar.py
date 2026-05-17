# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the grammar-based Qwen3 Coder tool call parser.

The ``qwen3_coder`` tool call format is identical to ``qwen3_xml``;
these tests validate that the ``GrammarQwen3CoderToolParser`` wrapper
correctly instantiates with the shared grammar config and produces
correct results for the Qwen3 XML tool call format used by Qwen 3.6.
"""

import json
from typing import Any
from unittest.mock import MagicMock

import pytest

from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
)
from vllm.grammar_parser.grammars.qwen3xml import (
    TOOL_CALL_END,
    TOOL_CALL_START,
)
from vllm.grammar_parser.registered_parsers import GrammarQwen3CoderToolParser


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
    return GrammarQwen3CoderToolParser(mock_tokenizer)


@pytest.fixture
def mock_request():
    request = MagicMock(spec=ChatCompletionRequest)
    request.tools = []
    request.tool_choice = "auto"
    return request


class TestNonStreaming:
    def test_no_tool_calls(self, parser, mock_request):
        result = parser.extract_tool_calls(
            "Hello, how can I help you today?",
            mock_request,
        )
        assert result.tools_called is False
        assert result.tool_calls == []
        assert result.content == "Hello, how can I help you today?"

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

    def test_multiple_parameters(self, parser, mock_request):
        text = (
            "<tool_call>\n<function=search>\n"
            "<parameter=query>vllm tool parsing</parameter>\n"
            "<parameter=limit>10</parameter>\n"
            "<parameter=exact_match>false</parameter>\n"
            "</function>\n</tool_call>"
        )
        result = parser.extract_tool_calls(text, mock_request)

        assert result.tools_called is True
        args = json.loads(result.tool_calls[0].function.arguments)
        assert args == {
            "query": "vllm tool parsing",
            "limit": 10,
            "exact_match": False,
        }

    def test_various_data_types(self, parser, mock_request):
        text = (
            "<tool_call>\n<function=test_func>\n"
            "<parameter=string_val>hello</parameter>\n"
            "<parameter=int_val>42</parameter>\n"
            "<parameter=float_val>3.14</parameter>\n"
            "<parameter=bool_val>true</parameter>\n"
            "<parameter=null_val>null</parameter>\n"
            '<parameter=array_val>["a", "b"]</parameter>\n'
            '<parameter=obj_val>{"key": "val"}</parameter>\n'
            "</function>\n</tool_call>"
        )
        result = parser.extract_tool_calls(text, mock_request)

        assert result.tools_called is True
        args = json.loads(result.tool_calls[0].function.arguments)
        assert args["string_val"] == "hello"
        assert args["int_val"] == 42
        assert args["float_val"] == 3.14
        assert args["bool_val"] is True
        assert args["null_val"] is None
        assert args["array_val"] == ["a", "b"]
        assert args["obj_val"] == {"key": "val"}

    def test_empty_arguments(self, parser, mock_request):
        text = "<tool_call>\n<function=refresh>\n</function>\n</tool_call>"
        result = parser.extract_tool_calls(text, mock_request)

        assert result.tools_called is True
        assert result.tool_calls[0].function.name == "refresh"
        args = json.loads(result.tool_calls[0].function.arguments)
        assert args == {}

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

    def test_surrounding_text(self, parser, mock_request):
        text = (
            "Let me check the weather for you.\n\n"
            "<tool_call>\n<function=get_weather>\n"
            "<parameter=city>Dallas</parameter>\n"
            "<parameter=state>TX</parameter>\n"
            "</function>\n</tool_call>"
        )
        result = parser.extract_tool_calls(text, mock_request)

        assert result.tools_called is True
        assert result.content is not None
        assert "Let me check the weather" in result.content
        assert result.tool_calls[0].function.name == "get_weather"
        args = json.loads(result.tool_calls[0].function.arguments)
        assert args == {"city": "Dallas", "state": "TX"}

    def test_escaped_strings(self, parser, mock_request):
        text = (
            "<tool_call>\n<function=write_file>\n"
            '<parameter=content>He said "hello"</parameter>\n'
            "<parameter=path>C:\\Users\\file.txt</parameter>\n"
            "</function>\n</tool_call>"
        )
        result = parser.extract_tool_calls(text, mock_request)

        assert result.tools_called is True
        args = json.loads(result.tool_calls[0].function.arguments)
        assert args["content"] == 'He said "hello"'
        assert args["path"] == "C:\\Users\\file.txt"

    def test_multiline_param_values(self, parser, mock_request):
        """Parameter values spanning multiple lines."""
        text = (
            "<tool_call>\n"
            "<function=Bash>\n"
            "<parameter=command>\n"
            "ls -la /tmp\n"
            "</parameter>\n"
            "<parameter=description>\n"
            "List files in /tmp directory\n"
            "</parameter>\n"
            "</function>\n"
            "</tool_call>"
        )
        result = parser.extract_tool_calls(text, mock_request)

        assert result.tools_called is True
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].function.name == "Bash"
        args = json.loads(result.tool_calls[0].function.arguments)
        assert args["command"] == "ls -la /tmp"
        assert args["description"] == "List files in /tmp directory"


class TestStreaming:
    def _simulate_streaming(
        self,
        parser: GrammarQwen3CoderToolParser,
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
            "<parameter=city>Dallas</parameter>\n",
            "<parameter=state>TX</parameter>\n",
            "</function>\n",
            "</tool_call>",
        ]
        results = self._simulate_streaming(parser, mock_request, chunks)

        name = self._collect_function_name(results)
        assert name == "get_weather"

        args_text = self._collect_arguments(results)
        assert args_text
        parsed = json.loads(args_text)
        assert parsed == {"city": "Dallas", "state": "TX"}

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

    def test_streaming_split_tool_call_tag(self, parser, mock_request):
        """<tool_call> tag split across chunks."""
        chunks = [
            "<tool_",
            "call>\n",
            "<function=test>\n",
            "<parameter=x>1</parameter>\n",
            "</function>\n",
            "</tool_call>",
        ]
        results = self._simulate_streaming(parser, mock_request, chunks)

        name = self._collect_function_name(results)
        assert name == "test"

        args_text = self._collect_arguments(results)
        assert args_text
        parsed = json.loads(args_text)
        assert parsed["x"] == 1

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

    def test_streaming_value_split(self, parser, mock_request):
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

    def test_char_by_char_streaming(self, parser, mock_request):
        """Feed text character-by-character to test robustness."""
        full_text = (
            "<tool_call>\n"
            "<function=echo>\n"
            "<parameter=msg>hi</parameter>\n"
            "</function>\n"
            "</tool_call>"
        )
        chunks = list(full_text)
        results = self._simulate_streaming(parser, mock_request, chunks)

        name = self._collect_function_name(results)
        assert name == "echo"

        args_text = self._collect_arguments(results)
        assert args_text
        parsed = json.loads(args_text)
        assert parsed == {"msg": "hi"}

    def test_streaming_multiline_param_values(self, parser, mock_request):
        """Multi-line parameter values in streaming mode."""
        chunks = [
            "<tool_call>\n",
            "<function=Bash>\n",
            "<parameter=command>\n",
            "ls -la /tmp\n",
            "</parameter>\n",
            "<parameter=description>\n",
            "List files\n",
            "</parameter>\n",
            "</function>\n",
            "</tool_call>",
        ]
        results = self._simulate_streaming(parser, mock_request, chunks)

        name = self._collect_function_name(results)
        assert name == "Bash"

        args_text = self._collect_arguments(results)
        assert args_text
        parsed = json.loads(args_text)
        assert "ls -la /tmp" in parsed["command"]
        assert "List files" in parsed["description"]

    def test_streaming_multiline_two_tool_calls(self, parser, mock_request):
        """Two tool calls with multi-line values — matches bug report."""
        chunks = [
            "<tool_call>\n",
            "<function=Bash>\n",
            "<parameter=command>\n",
            "find /workspace -name '*.py' | head -20\n",
            "</parameter>\n",
            "<parameter=description>\n",
            "Find Python files\n",
            "</parameter>\n",
            "</function>\n",
            "</tool_call>",
            "<tool_call>\n",
            "<function=Read>\n",
            "<parameter=file_path>\n",
            "/workspace/main.py\n",
            "</parameter>\n",
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

        assert "Bash" in names
        assert "Read" in names


class TestStreamingWithSpecialTokenIDs:
    """Tests simulating skip_special_tokens=True stripping <tool_call>."""

    @pytest.fixture
    def special_tokenizer(self):
        special_tokens = {TOOL_CALL_START: 100, TOOL_CALL_END: 101}
        reverse = {v: k for k, v in special_tokens.items()}
        tokenizer = MagicMock()
        tokenizer.encode.return_value = [1, 2, 3]
        tokenizer.get_vocab.return_value = special_tokens
        tokenizer.decode.side_effect = lambda ids: "".join(
            reverse.get(i, chr(i) if i < 128 else f"<{i}>") for i in ids
        )
        return tokenizer

    @pytest.fixture
    def parser(self, special_tokenizer):
        return GrammarQwen3CoderToolParser(special_tokenizer)

    def _simulate_streaming_with_token_ids(
        self,
        parser: GrammarQwen3CoderToolParser,
        mock_request,
        deltas: list[tuple[str, list[int]]],
    ) -> list[tuple[Any, str]]:
        results: list[tuple[Any, str]] = []
        previous_text = ""
        previous_token_ids: list[int] = []

        for delta_text, delta_tids in deltas:
            current_text = previous_text + delta_text
            current_token_ids = previous_token_ids + delta_tids

            delta = parser.extract_tool_calls_streaming(
                previous_text=previous_text,
                current_text=current_text,
                delta_text=delta_text,
                previous_token_ids=tuple(previous_token_ids),
                current_token_ids=tuple(current_token_ids),
                delta_token_ids=tuple(delta_tids),
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

    def test_deferred_tool_call_with_function_in_same_delta(self, parser, mock_request):
        """<tool_call> token ID present but text stripped."""
        deltas = [
            ("\n<function=get_weather>\n", [100, 1, 2, 3, 4]),
            ("<parameter=city>Tokyo</parameter>\n", [5, 6, 7, 8]),
            ("</function>\n", [9, 10]),
            ("", [101]),
        ]
        results = self._simulate_streaming_with_token_ids(parser, mock_request, deltas)

        name = self._collect_function_name(results)
        assert name == "get_weather"

        args_text = self._collect_arguments(results)
        assert args_text
        parsed = json.loads(args_text)
        assert parsed == {"city": "Tokyo"}

    def test_deferred_tool_call_multiline_params(self, parser, mock_request):
        """<tool_call> stripped + multi-line params — full bug scenario."""
        deltas = [
            (
                "\n<function=Bash>\n<parameter=command>\n",
                [100, 1, 2, 3, 4, 5],
            ),
            (
                "find /workspace -name '*.py' | head -20\n</parameter>\n",
                [6, 7, 8, 9, 10],
            ),
            (
                "<parameter=description>\nFind Python files\n</parameter>\n",
                [11, 12, 13, 14, 15],
            ),
            ("</function>\n", [16, 17]),
            ("", [101]),
        ]
        results = self._simulate_streaming_with_token_ids(parser, mock_request, deltas)

        name = self._collect_function_name(results)
        assert name == "Bash"

        args_text = self._collect_arguments(results)
        assert args_text
        parsed = json.loads(args_text)
        assert "find /workspace" in parsed["command"]
        assert "Find Python files" in parsed["description"]

    def test_holdback_with_stripped_tool_end_two_calls(self, parser, mock_request):
        """Two parallel tool calls where </tool_call> is stripped and
        detokenizer hold-back text arrives in the same delta.

        Reproduces the bug where the second tool call's parameters are
        silently dropped because </tool_call> fires before </function>
        is lexed."""
        deltas = [
            ("\n<function=Bash>\n", [100, 1, 2, 3]),
            ("<parameter=command>\necho hi\n</parameter>\n", [4, 5, 6, 7]),
            ("</function>\n", [8, 9, 101, 10]),
            ("\n<function=Read>\n", [100, 11, 12, 13]),
            (
                "<parameter=file_path>\n/workspace/main.py\n</parameter>\n",
                [14, 15, 16, 17],
            ),
            ("</function>\n", [18, 19]),
            ("", [101]),
        ]
        results = self._simulate_streaming_with_token_ids(parser, mock_request, deltas)

        names = []
        for delta, _ in results:
            if delta and delta.tool_calls:
                for tc in delta.tool_calls:
                    if tc.function and tc.function.name:
                        names.append(tc.function.name)

        assert "Bash" in names
        assert "Read" in names
