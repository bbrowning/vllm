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

    def test_multiline_two_tool_calls(self, parser, mock_request):
        """Two tool calls with multi-line parameter values (bug report)."""
        text = (
            "<tool_call>\n"
            "<function=Bash>\n"
            "<parameter=command>\n"
            "find /workspace -name '*.py' | head -20\n"
            "</parameter>\n"
            "<parameter=description>\n"
            "Find Python files\n"
            "</parameter>\n"
            "</function>\n"
            "</tool_call>"
            "<tool_call>\n"
            "<function=Read>\n"
            "<parameter=file_path>\n"
            "/workspace/main.py\n"
            "</parameter>\n"
            "</function>\n"
            "</tool_call>"
        )
        result = parser.extract_tool_calls(text, mock_request)

        assert result.tools_called is True
        assert len(result.tool_calls) == 2
        assert result.tool_calls[0].function.name == "Bash"
        assert result.tool_calls[1].function.name == "Read"
        args0 = json.loads(result.tool_calls[0].function.arguments)
        assert "find /workspace" in args0["command"]
        assert "Find Python files" in args0["description"]
        args1 = json.loads(result.tool_calls[1].function.arguments)
        assert "/workspace/main.py" in args1["file_path"]


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
    """Tests simulating skip_special_tokens=True stripping <tool_call>.

    When the tokenizer has <tool_call>/<tool_call> as special tokens and
    skip_special_tokens=True, delta_text will NOT contain the token text
    but delta_token_ids WILL contain the token ID. The TokenIDScanner must
    defer the terminal and replay post-terminal text correctly.
    """

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
        return GrammarToolParser(
            special_tokenizer,
            grammar_config=qwen3xml_config(),
        )

    def _simulate_streaming_with_token_ids(
        self,
        parser: GrammarToolParser,
        mock_request,
        deltas: list[tuple[str, list[int]]],
    ) -> list[tuple[Any, str]]:
        """Each delta is (delta_text, delta_token_ids)."""
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
        """<tool_call> token ID present but text stripped — function tag
        in the same delta must be parsed correctly after deferral."""
        deltas = [
            # Delta 1: <tool_call> token ID (100) but text stripped,
            # plus function tag text in same delta
            ("\n<function=get_weather>\n", [100, 1, 2, 3, 4]),
            ("<parameter=city>Tokyo</parameter>\n", [5, 6, 7, 8]),
            ("</function>\n", [9, 10]),
            # </tool_call> token ID (101) stripped
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
            # <tool_call> token (100) stripped, function + first param
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
            ("", [101]),  # </tool_call> stripped
        ]
        results = self._simulate_streaming_with_token_ids(parser, mock_request, deltas)

        name = self._collect_function_name(results)
        assert name == "Bash"

        args_text = self._collect_arguments(results)
        assert args_text
        parsed = json.loads(args_text)
        assert "find /workspace" in parsed["command"]
        assert "Find Python files" in parsed["description"]

    def test_deferred_tool_call_only_whitespace(self, parser, mock_request):
        """<tool_call> stripped with only whitespace in same delta."""
        deltas = [
            ("\n", [100, 1]),  # <tool_call> stripped, just newline
            ("<function=refresh>\n", [2, 3, 4]),
            ("</function>\n", [5, 6]),
            ("", [101]),
        ]
        results = self._simulate_streaming_with_token_ids(parser, mock_request, deltas)

        name = self._collect_function_name(results)
        assert name == "refresh"

    def test_holdback_with_stripped_tool_end_two_calls(self, parser, mock_request):
        """Two parallel tool calls where </tool_call> is stripped and
        detokenizer hold-back text arrives in the same delta.

        Reproduces the bug where the second tool call's parameters are
        silently dropped because </tool_call> fires before </function>
        is lexed."""
        deltas = [
            # First tool call: <tool_call> stripped
            ("\n<function=Bash>\n", [100, 1, 2, 3]),
            ("<parameter=command>\necho hi\n</parameter>\n", [4, 5, 6, 7]),
            # </function> text arrives, then </tool_call> stripped with
            # hold-back "\n" flushed in the same delta
            ("</function>\n", [8, 9, 101, 10]),
            # Second tool call: <tool_call> stripped
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

    def test_holdback_stripped_end_then_start_same_delta(self, parser, mock_request):
        """</tool_call> and <tool_call> both stripped in a single delta
        with hold-back text — worst case for parallel tool calls."""
        deltas = [
            # First tool call
            ("\n<function=Bash>\n", [100, 1, 2, 3]),
            ("<parameter=cmd>\nls\n</parameter>\n</function>", [4, 5, 6, 7]),
            # Both </tool_call> and <tool_call> stripped, newlines flushed
            ("\n\n", [101, 8, 100, 9]),
            # Second tool call continues
            ("<function=Read>\n", [10, 11, 12]),
            (
                "<parameter=file_path>\n/tmp/f.py\n</parameter>\n",
                [13, 14, 15, 16],
            ),
            ("</function>\n", [17, 18]),
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


class TestArgConverter:
    """Direct tests for the qwen3xml arg_converter with multi-line values."""

    def test_multiline_param_values(self):
        from vllm.grammar_parser.grammars.qwen3xml import (
            _qwen3xml_arg_converter,
        )

        raw = (
            "<parameter=command>\n"
            "ls -la /tmp\n"
            "</parameter>\n"
            "<parameter=description>\n"
            "List files\n"
            "</parameter>\n"
        )
        result = json.loads(_qwen3xml_arg_converter(raw, partial=False))
        assert result["command"] == "ls -la /tmp"
        assert result["description"] == "List files"

    def test_two_multiline_params(self):
        from vllm.grammar_parser.grammars.qwen3xml import (
            _qwen3xml_arg_converter,
        )

        raw = (
            "<parameter=a>\nfoo\nbar\n</parameter>\n"
            "<parameter=b>\nbaz\nqux\n</parameter>\n"
        )
        result = json.loads(_qwen3xml_arg_converter(raw, partial=False))
        assert result["a"] == "foo\nbar"
        assert result["b"] == "baz\nqux"

    def test_partial_multiline(self):
        from vllm.grammar_parser.grammars.qwen3xml import (
            _qwen3xml_arg_converter,
        )

        raw = "<parameter=command>\nls -la</parameter>\n<parameter=desc>\npartial value"
        result = json.loads(_qwen3xml_arg_converter(raw, partial=True))
        assert result["command"] == "ls -la"
        assert result["desc"] == "\npartial value"
