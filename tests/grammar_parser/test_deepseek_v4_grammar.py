# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the DeepSeek V4 grammar parser (DSML tool calls + reasoning)."""

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
from vllm.grammar_parser.parsers.deepseek_v4 import (
    DSML_INVOKE_END,
    DSML_INVOKE_NAME_END,
    DSML_INVOKE_PREFIX,
    DSML_THINK_END,
    DSML_THINK_START,
    DSML_TOOL_CALLS_END,
    DSML_TOOL_CALLS_START,
    _dsml_arg_converter,
    deepseek_v4_config,
)
from vllm.grammar_parser.unified_parser import GrammarParser

_THINK_START_ID = 50
_THINK_END_ID = 51

# DSML convenience strings (｜ is U+FF5C FULLWIDTH VERTICAL LINE)
_PARAM_OPEN = '｜DSML｜parameter name="{name}" string="{is_str}">'
_PARAM_CLOSE = "</｜DSML｜parameter>"


def _param(name: str, is_str: str, value: str) -> str:
    """Return a complete DSML parameter tag."""
    return f"<{_PARAM_OPEN.format(name=name, is_str=is_str)}{value}{_PARAM_CLOSE}"


def _invoke_block(func_name: str, *params: str) -> str:
    """Return a complete <｜DSML｜invoke> block."""
    body = "\n".join(params)
    return (
        f"{DSML_INVOKE_PREFIX}{func_name}{DSML_INVOKE_NAME_END}\n"
        f"{body}\n"
        f"{DSML_INVOKE_END}"
    )


def _tool_section(*invokes: str) -> str:
    """Wrap invoke blocks in a <｜DSML｜tool_calls> section."""
    inner = "\n".join(invokes)
    return DSML_TOOL_CALLS_START + "\n" + inner + "\n" + DSML_TOOL_CALLS_END


@pytest.fixture
def mock_tokenizer():
    tok = MagicMock()
    tok.encode.return_value = [1, 2, 3]
    tok.get_vocab.return_value = {
        DSML_THINK_START: _THINK_START_ID,
        DSML_THINK_END: _THINK_END_ID,
    }
    tok.decode.side_effect = lambda ids: "".join(
        chr(i) if i < 128 else f"<{i}>" for i in ids
    )
    return tok


@pytest.fixture
def parser(mock_tokenizer):
    return GrammarParser(mock_tokenizer, grammar_config=deepseek_v4_config())


@pytest.fixture
def mock_request():
    req = MagicMock(spec=ChatCompletionRequest)
    req.tools = []
    req.tool_choice = "auto"
    return req


# ═══════════════════════════════════════════════════════════════════════════════
# Arg converter unit tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestArgConverter:
    def _raw(self, *params: tuple[str, str, str]) -> str:
        """Build raw args body from (name, is_str, value) tuples."""
        lines = [_param(n, s, v) for n, s, v in params]
        return "\n" + "\n".join(lines) + "\n"

    def test_string_param(self):
        raw = self._raw(("city", "true", "杭州"))
        result = json.loads(_dsml_arg_converter(raw, partial=False))
        assert result == {"city": "杭州"}

    def test_string_with_spaces_and_quotes(self):
        raw = self._raw(("msg", "true", 'He said "hello world"'))
        result = json.loads(_dsml_arg_converter(raw, partial=False))
        assert result["msg"] == 'He said "hello world"'

    def test_integer_param(self):
        raw = self._raw(("count", "false", "42"))
        result = json.loads(_dsml_arg_converter(raw, partial=False))
        assert result["count"] == 42
        assert isinstance(result["count"], int)

    def test_float_param(self):
        raw = self._raw(("ratio", "false", "3.14"))
        result = json.loads(_dsml_arg_converter(raw, partial=False))
        assert abs(result["ratio"] - 3.14) < 1e-9

    def test_bool_param(self):
        raw = self._raw(("flag", "false", "true"))
        result = json.loads(_dsml_arg_converter(raw, partial=False))
        assert result["flag"] is True

    def test_array_param(self):
        raw = self._raw(("items", "false", '["a", "b", "c"]'))
        result = json.loads(_dsml_arg_converter(raw, partial=False))
        assert result["items"] == ["a", "b", "c"]

    def test_object_param(self):
        raw = self._raw(("opts", "false", '{"key": "val"}'))
        result = json.loads(_dsml_arg_converter(raw, partial=False))
        assert result["opts"] == {"key": "val"}

    def test_mixed_types(self):
        raw = self._raw(
            ("location", "true", "Tokyo"),
            ("limit", "false", "10"),
            ("active", "false", "false"),
        )
        result = json.loads(_dsml_arg_converter(raw, partial=False))
        assert result == {"location": "Tokyo", "limit": 10, "active": False}

    def test_empty_args(self):
        result = json.loads(_dsml_arg_converter("", partial=False))
        assert result == {}

    def test_whitespace_only(self):
        result = json.loads(_dsml_arg_converter("\n  \n", partial=False))
        assert result == {}

    def test_invalid_json_fallback(self):
        raw = self._raw(("data", "false", "[broken"))
        result = json.loads(_dsml_arg_converter(raw, partial=False))
        assert result["data"] == "[broken"

    def test_chinese_chars_preserved_in_json(self):
        raw = self._raw(("query", "true", "你好世界"))
        raw_json = _dsml_arg_converter(raw, partial=False)
        assert "你好世界" in raw_json
        result = json.loads(raw_json)
        assert result["query"] == "你好世界"

    def test_partial_complete_plus_in_progress(self):
        raw = self._raw(("city", "true", "Tokyo"))
        raw += f"<{_PARAM_OPEN.format(name='unit', is_str='true')}celsi"
        result = json.loads(_dsml_arg_converter(raw, partial=True))
        assert result["city"] == "Tokyo"
        assert result["unit"] == "celsi"

    def test_partial_no_in_progress(self):
        raw = self._raw(("city", "true", "Tokyo"))
        result = json.loads(_dsml_arg_converter(raw, partial=True))
        assert result == {"city": "Tokyo"}

    def test_null_string_false(self):
        raw = self._raw(("val", "false", "null"))
        result = json.loads(_dsml_arg_converter(raw, partial=False))
        assert result["val"] is None

    def test_string_true_not_json_parsed(self):
        # string="true" means keep the value as-is even if it looks like JSON
        raw = self._raw(("n", "true", "42"))
        result = json.loads(_dsml_arg_converter(raw, partial=False))
        assert result["n"] == "42"
        assert isinstance(result["n"], str)


# ═══════════════════════════════════════════════════════════════════════════════
# Non-streaming tool call extraction
# ═══════════════════════════════════════════════════════════════════════════════


class TestNonStreamingToolCalls:
    def test_no_tool_calls(self, parser, mock_request):
        result = parser.extract_tool_calls("Hello, how can I help?", mock_request)
        assert result.tools_called is False
        assert result.tool_calls == []
        assert result.content == "Hello, how can I help?"

    def test_single_invoke_string_param(self, parser, mock_request):
        text = _tool_section(
            _invoke_block("get_weather", _param("city", "true", "Tokyo"))
        )
        result = parser.extract_tool_calls(text, mock_request)
        assert result.tools_called is True
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].function.name == "get_weather"
        args = json.loads(result.tool_calls[0].function.arguments)
        assert args == {"city": "Tokyo"}

    def test_single_invoke_numeric_param(self, parser, mock_request):
        text = _tool_section(
            _invoke_block("set_limit", _param("limit", "false", "100"))
        )
        result = parser.extract_tool_calls(text, mock_request)
        assert result.tools_called is True
        args = json.loads(result.tool_calls[0].function.arguments)
        assert args["limit"] == 100
        assert isinstance(args["limit"], int)

    def test_parallel_invokes(self, parser, mock_request):
        text = _tool_section(
            _invoke_block("get_weather", _param("city", "true", "Paris")),
            _invoke_block("get_time", _param("tz", "true", "Europe/Paris")),
        )
        result = parser.extract_tool_calls(text, mock_request)
        assert result.tools_called is True
        assert len(result.tool_calls) == 2
        assert result.tool_calls[0].function.name == "get_weather"
        assert result.tool_calls[1].function.name == "get_time"
        args0 = json.loads(result.tool_calls[0].function.arguments)
        assert args0 == {"city": "Paris"}
        args1 = json.loads(result.tool_calls[1].function.arguments)
        assert args1 == {"tz": "Europe/Paris"}

    def test_mixed_type_params(self, parser, mock_request):
        text = _tool_section(
            _invoke_block(
                "search",
                _param("query", "true", "python vllm"),
                _param("limit", "false", "5"),
                _param("exact", "false", "false"),
                _param("tags", "false", '["ai", "ml"]'),
            )
        )
        result = parser.extract_tool_calls(text, mock_request)
        assert result.tools_called is True
        args = json.loads(result.tool_calls[0].function.arguments)
        assert args["query"] == "python vllm"
        assert args["limit"] == 5
        assert args["exact"] is False
        assert args["tags"] == ["ai", "ml"]

    def test_empty_invoke(self, parser, mock_request):
        text = _tool_section(_invoke_block("refresh"))
        result = parser.extract_tool_calls(text, mock_request)
        assert result.tools_called is True
        assert result.tool_calls[0].function.name == "refresh"
        args = json.loads(result.tool_calls[0].function.arguments)
        assert args == {}

    def test_content_before_tool_calls(self, parser, mock_request):
        inv = _invoke_block("get_weather", _param("city", "true", "London"))
        text = "Let me look that up for you.\n" + _tool_section(inv)
        result = parser.extract_tool_calls(text, mock_request)
        assert result.tools_called is True
        assert result.content is not None
        assert "Let me look that up" in result.content

    def test_chinese_param_value(self, parser, mock_request):
        text = _tool_section(
            _invoke_block("get_weather", _param("city", "true", "杭州"))
        )
        result = parser.extract_tool_calls(text, mock_request)
        assert result.tools_called is True
        args = json.loads(result.tool_calls[0].function.arguments)
        assert args["city"] == "杭州"

    def test_string_true_keeps_raw_value(self, parser, mock_request):
        text = _tool_section(_invoke_block("func", _param("n", "true", "42")))
        result = parser.extract_tool_calls(text, mock_request)
        args = json.loads(result.tool_calls[0].function.arguments)
        assert args["n"] == "42"
        assert isinstance(args["n"], str)


# ═══════════════════════════════════════════════════════════════════════════════
# Non-streaming reasoning extraction
# ═══════════════════════════════════════════════════════════════════════════════


class TestNonStreamingReasoning:
    def test_reasoning_and_content(self, parser, mock_request):
        text = "<think>\nLet me think about this.\n</think>\nHere is the answer."
        reasoning, content = parser.extract_reasoning(text, mock_request)
        assert reasoning is not None
        assert "Let me think" in reasoning
        assert "<think>" not in reasoning
        assert "</think>" not in reasoning
        assert content is not None
        assert "Here is the answer" in content

    def test_chat_mode_no_think_tag(self, parser, mock_request):
        text = "The answer is 42."
        reasoning, content = parser.extract_reasoning(text, mock_request)
        assert reasoning is None
        assert content == text

    def test_reasoning_then_tool_call(self, parser, mock_request):
        tool_text = _tool_section(
            _invoke_block("get_weather", _param("city", "true", "Berlin"))
        )
        text = "<think>\nI should check the weather.\n</think>\n" + tool_text
        reasoning, content = parser.extract_reasoning(text, mock_request)
        assert reasoning is not None
        assert "check the weather" in reasoning

        result = parser.extract_tool_calls(text, mock_request)
        assert result.tools_called is True
        assert result.tool_calls[0].function.name == "get_weather"

    def test_no_think_tags_in_reasoning_output(self, parser, mock_request):
        text = "<think>deep thought</think>answer"
        reasoning, _ = parser.extract_reasoning(text, mock_request)
        assert reasoning is not None
        assert "<think>" not in reasoning
        assert "</think>" not in reasoning

    def test_bare_think_end_suppresses_thinking(self, parser, mock_request):
        # Model emits </think> without a prior <think> to suppress reasoning.
        # The tag should be silently absorbed, not leaked into content.
        text = "</think>Here is the direct answer."
        reasoning, content = parser.extract_reasoning(text, mock_request)
        assert reasoning is None
        assert content is not None
        assert "</think>" not in content
        assert "Here is the direct answer" in content


# ═══════════════════════════════════════════════════════════════════════════════
# Streaming tool calls
# ═══════════════════════════════════════════════════════════════════════════════


class TestStreamingToolCalls:
    def test_basic_single_invoke(self, parser, mock_request):
        chunks = [
            DSML_TOOL_CALLS_START + "\n",
            DSML_INVOKE_PREFIX + "get_weather" + DSML_INVOKE_NAME_END + "\n",
            _param("city", "true", "Tokyo") + "\n",
            DSML_INVOKE_END + "\n",
            DSML_TOOL_CALLS_END,
        ]
        results = simulate_tool_streaming(parser, mock_request, chunks)
        name = collect_function_name(results)
        assert name == "get_weather"
        args_text = collect_tool_arguments(results)
        assert args_text
        args = json.loads(args_text)
        assert args == {"city": "Tokyo"}

    def test_streaming_parallel_invokes(self, parser, mock_request):
        chunks = [
            DSML_TOOL_CALLS_START + "\n",
            DSML_INVOKE_PREFIX + "get_weather" + DSML_INVOKE_NAME_END + "\n",
            _param("city", "true", "Paris") + "\n",
            DSML_INVOKE_END + "\n",
            DSML_INVOKE_PREFIX + "get_time" + DSML_INVOKE_NAME_END + "\n",
            _param("tz", "true", "Europe/Paris") + "\n",
            DSML_INVOKE_END + "\n",
            DSML_TOOL_CALLS_END,
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

    def test_streaming_text_before_tool_section(self, parser, mock_request):
        chunks = [
            "Let me check ",
            "the weather. ",
            DSML_TOOL_CALLS_START + "\n",
            DSML_INVOKE_PREFIX + "get_weather" + DSML_INVOKE_NAME_END + "\n",
            _param("city", "true", "Tokyo") + "\n",
            DSML_INVOKE_END + "\n",
            DSML_TOOL_CALLS_END,
        ]
        results = simulate_tool_streaming(parser, mock_request, chunks)
        content_parts = []
        for delta, _ in results:
            if delta and delta.content:
                content_parts.append(delta.content)
        assert "Let me check" in "".join(content_parts)

    def test_streaming_value_split_across_chunks(self, parser, mock_request):
        chunks = [
            DSML_TOOL_CALLS_START + "\n",
            DSML_INVOKE_PREFIX + "search" + DSML_INVOKE_NAME_END + "\n",
            '<｜DSML｜parameter name="query" string="true">hello ',
            "world",
            "</｜DSML｜parameter>\n",
            DSML_INVOKE_END + "\n",
            DSML_TOOL_CALLS_END,
        ]
        results = simulate_tool_streaming(parser, mock_request, chunks)
        args_text = collect_tool_arguments(results)
        assert args_text
        args = json.loads(args_text)
        assert args["query"] == "hello world"

    def test_streaming_chinese_value(self, parser, mock_request):
        chunks = [
            DSML_TOOL_CALLS_START + "\n",
            DSML_INVOKE_PREFIX + "get_weather" + DSML_INVOKE_NAME_END + "\n",
            _param("city", "true", "杭州") + "\n",
            DSML_INVOKE_END + "\n",
            DSML_TOOL_CALLS_END,
        ]
        results = simulate_tool_streaming(parser, mock_request, chunks)
        args_text = collect_tool_arguments(results)
        assert args_text
        args = json.loads(args_text)
        assert args["city"] == "杭州"

    def test_streaming_numeric_param(self, parser, mock_request):
        chunks = [
            DSML_TOOL_CALLS_START + "\n",
            DSML_INVOKE_PREFIX + "set_limit" + DSML_INVOKE_NAME_END + "\n",
            _param("n", "false", "100") + "\n",
            DSML_INVOKE_END + "\n",
            DSML_TOOL_CALLS_END,
        ]
        results = simulate_tool_streaming(parser, mock_request, chunks)
        args_text = collect_tool_arguments(results)
        assert args_text
        args = json.loads(args_text)
        assert args["n"] == 100

    def test_streaming_empty_invoke(self, parser, mock_request):
        chunks = [
            DSML_TOOL_CALLS_START + "\n",
            DSML_INVOKE_PREFIX + "refresh" + DSML_INVOKE_NAME_END + "\n",
            DSML_INVOKE_END + "\n",
            DSML_TOOL_CALLS_END,
        ]
        results = simulate_tool_streaming(parser, mock_request, chunks)
        name = collect_function_name(results)
        assert name == "refresh"


# ═══════════════════════════════════════════════════════════════════════════════
# Streaming reasoning
# ═══════════════════════════════════════════════════════════════════════════════


class TestStreamingReasoning:
    def test_reasoning_then_content(self, parser):
        chunks = [
            "<think>\n",
            "Let me consider this carefully.\n",
            "</think>\n",
            "Here is the result.",
        ]
        reasoning, content = simulate_reasoning_streaming(parser, chunks)
        assert "Let me consider" in reasoning
        assert "Here is the result" in content

    def test_chat_mode_no_think(self, parser):
        chunks = ["The answer is ", "42."]
        reasoning, content = simulate_reasoning_streaming(parser, chunks)
        assert reasoning == ""
        assert "42" in content

    def test_reasoning_split_across_chunks(self, parser):
        chunks = [
            "<thi",
            "nk>\n",
            "Step one: gather info.\n",
            "Step two: analyze.\n",
            "</think>\n",
            "Done.",
        ]
        reasoning, content = simulate_reasoning_streaming(parser, chunks)
        assert "gather info" in reasoning
        assert "analyze" in reasoning
        assert "Done" in content

    def test_reasoning_then_tool_call_streaming(self, parser, mock_request):
        chunks = [
            "<think>\nI should look this up.\n</think>\n",
            DSML_TOOL_CALLS_START + "\n",
            DSML_INVOKE_PREFIX + "get_weather" + DSML_INVOKE_NAME_END + "\n",
            _param("city", "true", "Tokyo") + "\n",
            DSML_INVOKE_END + "\n",
            DSML_TOOL_CALLS_END,
        ]
        results = simulate_tool_streaming(parser, mock_request, chunks)
        name = collect_function_name(results)
        assert name == "get_weather"
