# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the adapter layer (GrammarToolParser, GrammarReasoningParser)."""

from unittest.mock import MagicMock

from vllm.grammar_parser.adapter import (
    GrammarReasoningParser,
    GrammarToolParser,
)
from vllm.grammar_parser.events import EventType
from vllm.grammar_parser.grammar_config import (
    GrammarConfig,
    ParserState,
    Transition,
)


def _hermes_config() -> GrammarConfig:
    return GrammarConfig(
        name="hermes_test",
        terminals={
            "TOOL_START": "<tool_call>",
            "TOOL_END": "</tool_call>",
        },
        transitions={
            (ParserState.CONTENT, "TOOL_START"): Transition(
                ParserState.TOOL_ARGS,
                [EventType.TOOL_CALL_START],
            ),
            (ParserState.TOOL_ARGS, "TOOL_END"): Transition(
                ParserState.CONTENT,
                [EventType.TOOL_CALL_END],
            ),
        },
    )


def _think_config() -> GrammarConfig:
    return GrammarConfig(
        name="think_test",
        terminals={
            "THINK_START": "<think>",
            "THINK_END": "</think>",
        },
        token_id_terminals={
            "THINK_START": "<think>",
            "THINK_END": "</think>",
        },
        transitions={
            (ParserState.CONTENT, "THINK_START"): Transition(
                ParserState.REASONING,
                [EventType.REASONING_START],
            ),
            (ParserState.REASONING, "THINK_END"): Transition(
                ParserState.CONTENT,
                [EventType.REASONING_END],
            ),
        },
    )


def _mock_tokenizer(vocab=None):
    tok = MagicMock()
    tok.get_vocab.return_value = vocab or {}
    tok.decode.side_effect = lambda ids: "".join(
        chr(i) if i < 128 else f"<{i}>" for i in ids
    )
    return tok


def _mock_request():
    req = MagicMock()
    req.tools = []
    req.tool_choice = "auto"
    return req


class TestGrammarToolParser:
    def test_non_streaming_single_tool(self):
        parser = GrammarToolParser(
            _mock_tokenizer(),
            grammar_config=_hermes_config(),
        )
        text = (
            '<tool_call>{"name": "get_weather",'
            ' "arguments": {"city": "NYC"}}</tool_call>'
        )
        result = parser.extract_tool_calls(text, _mock_request())

        assert result.tools_called
        assert len(result.tool_calls) == 1
        tc = result.tool_calls[0]
        assert tc.function.name == "get_weather"
        assert '"city"' in tc.function.arguments
        assert '"NYC"' in tc.function.arguments

    def test_non_streaming_with_content(self):
        parser = GrammarToolParser(
            _mock_tokenizer(),
            grammar_config=_hermes_config(),
        )
        text = 'Sure!<tool_call>{"name": "add", "arguments": {"a": 1}}</tool_call>'
        result = parser.extract_tool_calls(text, _mock_request())

        assert result.tools_called
        assert result.content == "Sure!"
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].function.name == "add"

    def test_non_streaming_multiple_tools(self):
        parser = GrammarToolParser(
            _mock_tokenizer(),
            grammar_config=_hermes_config(),
        )
        text = (
            '<tool_call>{"name": "a", "arguments": {}}</tool_call>'
            '<tool_call>{"name": "b", "arguments": {"x": 1}}</tool_call>'
        )
        result = parser.extract_tool_calls(text, _mock_request())

        assert result.tools_called
        assert len(result.tool_calls) == 2
        assert result.tool_calls[0].function.name == "a"
        assert result.tool_calls[1].function.name == "b"

    def test_streaming_produces_deltas(self):
        parser = GrammarToolParser(
            _mock_tokenizer(),
            grammar_config=_hermes_config(),
        )
        text = '<tool_call>{"name": "f", "arguments": {"k": "v"}}</tool_call>'

        all_deltas = []
        for i, ch in enumerate(text):
            delta = parser.extract_tool_calls_streaming(
                previous_text=text[:i],
                current_text=text[: i + 1],
                delta_text=ch,
                previous_token_ids=[],
                current_token_ids=[],
                delta_token_ids=[],
                request=_mock_request(),
            )
            if delta is not None:
                all_deltas.append(delta)

        name_deltas = [
            d
            for d in all_deltas
            if d.tool_calls
            and d.tool_calls[0].function
            and d.tool_calls[0].function.name is not None
        ]
        assert len(name_deltas) >= 1

        arg_text = "".join(
            d.tool_calls[0].function.arguments
            for d in all_deltas
            if d.tool_calls
            and d.tool_calls[0].function
            and d.tool_calls[0].function.arguments is not None
        )
        assert len(arg_text) > 0

    def test_no_tools_returns_empty(self):
        parser = GrammarToolParser(
            _mock_tokenizer(),
            grammar_config=_hermes_config(),
        )
        result = parser.extract_tool_calls("Just text.", _mock_request())
        assert not result.tools_called
        assert len(result.tool_calls) == 0
        assert result.content == "Just text."


class TestGrammarReasoningParser:
    def test_non_streaming(self):
        parser = GrammarReasoningParser(
            _mock_tokenizer(),
            grammar_config=_think_config(),
        )
        reasoning, content = parser.extract_reasoning(
            "<think>Let me think</think>The answer",
            _mock_request(),
        )
        assert reasoning is not None
        assert "Let me think" in reasoning
        assert content is not None
        assert "The answer" in content

    def test_streaming(self):
        parser = GrammarReasoningParser(
            _mock_tokenizer(),
            grammar_config=_think_config(),
        )
        text = "<think>hmm</think>ok"
        all_deltas = []
        for i, ch in enumerate(text):
            delta = parser.extract_reasoning_streaming(
                previous_text=text[:i],
                current_text=text[: i + 1],
                delta_text=ch,
                previous_token_ids=[],
                current_token_ids=[],
                delta_token_ids=[],
            )
            if delta is not None:
                all_deltas.append(delta)

        reasoning = "".join(d.reasoning for d in all_deltas if d.reasoning)
        content = "".join(d.content for d in all_deltas if d.content)
        assert "hmm" in reasoning
        assert "ok" in content

    def test_is_reasoning_end(self):
        tok = _mock_tokenizer(vocab={"<think>": 100, "</think>": 101})
        parser = GrammarReasoningParser(
            tok,
            grammar_config=_think_config(),
        )
        assert parser.is_reasoning_end([100, 50, 51, 101]) is True
        assert parser.is_reasoning_end([100, 50, 51]) is False
        assert parser.is_reasoning_end([50, 51]) is False

    def test_extract_content_ids(self):
        tok = _mock_tokenizer(vocab={"<think>": 100, "</think>": 101})
        parser = GrammarReasoningParser(
            tok,
            grammar_config=_think_config(),
        )
        assert parser.extract_content_ids([100, 50, 51, 101, 60, 70]) == [60, 70]
        assert parser.extract_content_ids([50, 51, 60]) == [50, 51, 60]
