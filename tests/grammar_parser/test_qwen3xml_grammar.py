# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the grammar-based Qwen3 XML tool call parser.

These mirror the test cases from test_qwen3xml_tool_parser.py to validate
that the grammar-driven parser produces identical results.
"""

import json
from unittest.mock import MagicMock

import pytest

from tests.grammar_parser.streaming_helpers import (
    collect_function_name,
    collect_tool_arguments,
    simulate_tool_streaming,
    simulate_tool_streaming_with_ids,
)
from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
)
from vllm.grammar_parser.grammars.qwen3xml import (
    TOOL_CALL_END,
    TOOL_CALL_START,
    qwen3xml_config,
)
from vllm.grammar_parser.unified_parser import GrammarParser


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
    return GrammarParser(
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
    def test_basic_streaming(self, parser, mock_request):
        chunks = [
            "<tool_call>\n",
            "<function=get_weather>\n",
            "<parameter=city>Tokyo",
            "</parameter>\n",
            "</function>\n",
            "</tool_call>",
        ]

        results = simulate_tool_streaming(parser, mock_request, chunks)

        name = collect_function_name(results)
        assert name == "get_weather"

        args_text = collect_tool_arguments(results)
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

        results = simulate_tool_streaming(parser, mock_request, chunks)

        name = collect_function_name(results)
        assert name == "get_weather"

        args_text = collect_tool_arguments(results)
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

        results = simulate_tool_streaming(parser, mock_request, chunks)

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

        results = simulate_tool_streaming(parser, mock_request, chunks)

        name = collect_function_name(results)
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

        results = simulate_tool_streaming(parser, mock_request, chunks)

        name = collect_function_name(results)
        assert name == "test"

        args_text = collect_tool_arguments(results)
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

        results = simulate_tool_streaming(parser, mock_request, chunks)
        args_text = collect_tool_arguments(results)
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

        results = simulate_tool_streaming(parser, mock_request, chunks)

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

        results = simulate_tool_streaming(parser, mock_request, chunks)

        args_text = collect_tool_arguments(results)
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
        results = simulate_tool_streaming(parser, mock_request, chunks)

        name = collect_function_name(results)
        assert name == "Bash"

        args_text = collect_tool_arguments(results)
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
        results = simulate_tool_streaming(parser, mock_request, chunks)

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
        return GrammarParser(
            special_tokenizer,
            grammar_config=qwen3xml_config(),
        )

    def test_deferred_tool_call_with_function_in_same_delta(self, parser, mock_request):
        """<tool_call> token ID present but text stripped — function tag
        in the same delta must be parsed correctly after deferral."""
        deltas = [
            ("\n<function=get_weather>\n", [100, 1, 2, 3, 4]),
            ("<parameter=city>Tokyo</parameter>\n", [5, 6, 7, 8]),
            ("</function>\n", [9, 10]),
            ("", [101]),
        ]
        results = simulate_tool_streaming_with_ids(parser, mock_request, deltas)

        name = collect_function_name(results)
        assert name == "get_weather"

        args_text = collect_tool_arguments(results)
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
        results = simulate_tool_streaming_with_ids(parser, mock_request, deltas)

        name = collect_function_name(results)
        assert name == "Bash"

        args_text = collect_tool_arguments(results)
        assert args_text
        parsed = json.loads(args_text)
        assert "find /workspace" in parsed["command"]
        assert "Find Python files" in parsed["description"]

    def test_deferred_tool_call_only_whitespace(self, parser, mock_request):
        """<tool_call> stripped with only whitespace in same delta."""
        deltas = [
            ("\n", [100, 1]),
            ("<function=refresh>\n", [2, 3, 4]),
            ("</function>\n", [5, 6]),
            ("", [101]),
        ]
        results = simulate_tool_streaming_with_ids(parser, mock_request, deltas)

        name = collect_function_name(results)
        assert name == "refresh"

    def test_holdback_with_stripped_tool_end_two_calls(self, parser, mock_request):
        """Two parallel tool calls where </tool_call> is stripped and
        detokenizer hold-back text arrives in the same delta."""
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
        results = simulate_tool_streaming_with_ids(parser, mock_request, deltas)

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
            ("\n<function=Bash>\n", [100, 1, 2, 3]),
            ("<parameter=cmd>\nls\n</parameter>\n</function>", [4, 5, 6, 7]),
            ("\n\n", [101, 8, 100, 9]),
            ("<function=Read>\n", [10, 11, 12]),
            (
                "<parameter=file_path>\n/tmp/f.py\n</parameter>\n",
                [13, 14, 15, 16],
            ),
            ("</function>\n", [17, 18]),
            ("", [101]),
        ]
        results = simulate_tool_streaming_with_ids(parser, mock_request, deltas)

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
