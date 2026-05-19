# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the streaming parser engine core pipeline."""

from unittest.mock import MagicMock

from vllm.grammar_parser.events import EventType, SemanticEvent
from vllm.grammar_parser.grammar_config import (
    GrammarConfig,
    ParserState,
    Transition,
)
from vllm.grammar_parser.parser_engine import StreamingParserEngine


def _hermes_config() -> GrammarConfig:
    """Simple Hermes-style config: <tool_call>JSON</tool_call>."""
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
    """Simple think-tag reasoning config: <think>...</think>."""
    return GrammarConfig(
        name="think_test",
        terminals={
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


class TestNonStreaming:
    def test_plain_text(self):
        engine = StreamingParserEngine(_hermes_config(), tokenizer=None)
        events = engine.parse_complete("Hello, world!")
        assert len(events) == 1
        assert events[0].type == EventType.TEXT_CHUNK
        assert events[0].value == "Hello, world!"

    def test_single_tool_call(self):
        engine = StreamingParserEngine(_hermes_config(), tokenizer=None)
        text = (
            '<tool_call>{"name": "get_weather",'
            ' "arguments": {"city": "SF"}}'
            "</tool_call>"
        )
        events = engine.parse_complete(text)

        types = [e.type for e in events]
        assert EventType.TOOL_CALL_START in types
        assert EventType.TOOL_CALL_END in types
        assert EventType.ARG_VALUE_CHUNK in types

        arg_text = "".join(
            e.value for e in events if e.type == EventType.ARG_VALUE_CHUNK
        )
        assert '"name": "get_weather"' in arg_text
        assert '"city": "SF"' in arg_text

    def test_text_then_tool_call(self):
        engine = StreamingParserEngine(_hermes_config(), tokenizer=None)
        text = 'Sure!<tool_call>{"name": "add"}</tool_call>'
        events = engine.parse_complete(text)

        types = [e.type for e in events]
        assert types[0] == EventType.TEXT_CHUNK
        assert events[0].value == "Sure!"
        assert EventType.TOOL_CALL_START in types
        assert EventType.TOOL_CALL_END in types

    def test_multiple_tool_calls(self):
        engine = StreamingParserEngine(_hermes_config(), tokenizer=None)
        text = (
            '<tool_call>{"name": "a"}</tool_call><tool_call>{"name": "b"}</tool_call>'
        )
        events = engine.parse_complete(text)

        starts = [e for e in events if e.type == EventType.TOOL_CALL_START]
        ends = [e for e in events if e.type == EventType.TOOL_CALL_END]
        assert len(starts) == 2
        assert len(ends) == 2
        assert starts[0].tool_index == 0
        assert starts[1].tool_index == 1

    def test_reasoning(self):
        engine = StreamingParserEngine(_think_config(), tokenizer=None)
        text = "<think>Let me think...</think>The answer is 42."
        events = engine.parse_complete(text)

        types = [e.type for e in events]
        assert types[0] == EventType.REASONING_START
        assert EventType.REASONING_CHUNK in types
        assert EventType.REASONING_END in types
        assert EventType.TEXT_CHUNK in types

        reasoning = "".join(
            e.value for e in events if e.type == EventType.REASONING_CHUNK
        )
        assert "Let me think..." in reasoning

        content = "".join(e.value for e in events if e.type == EventType.TEXT_CHUNK)
        assert "The answer is 42." in content


class TestStreaming:
    @staticmethod
    def _feed_chars(
        engine: StreamingParserEngine,
        text: str,
    ) -> list[SemanticEvent]:
        """Feed text one character at a time."""
        all_events = []
        for ch in text:
            all_events.extend(engine.feed(ch, []))
        all_events.extend(engine.finish())
        return all_events

    @staticmethod
    def _feed_chunks(
        engine: StreamingParserEngine,
        text: str,
        chunk_size: int,
    ) -> list[SemanticEvent]:
        """Feed text in fixed-size chunks."""
        all_events = []
        for i in range(0, len(text), chunk_size):
            chunk = text[i : i + chunk_size]
            all_events.extend(engine.feed(chunk, []))
        all_events.extend(engine.finish())
        return all_events

    def test_char_by_char_tool_call(self):
        engine = StreamingParserEngine(_hermes_config(), tokenizer=None)
        text = '<tool_call>{"name": "add", "arguments": {"a": 1}}</tool_call>'
        events = self._feed_chars(engine, text)

        types = [e.type for e in events]
        assert EventType.TOOL_CALL_START in types
        assert EventType.TOOL_CALL_END in types
        assert EventType.ARG_VALUE_CHUNK in types

        arg_text = "".join(
            e.value for e in events if e.type == EventType.ARG_VALUE_CHUNK
        )
        assert '"name": "add"' in arg_text

    def test_chunk_sizes_produce_same_content(self):
        """Different chunk sizes must produce identical concatenated content."""
        text = '<tool_call>{"name": "get", "arguments": {"x": "hello"}}</tool_call>'

        results = {}
        for chunk_size in [1, 2, 3, 5, 7, len(text)]:
            engine = StreamingParserEngine(_hermes_config(), tokenizer=None)
            events = self._feed_chunks(engine, text, chunk_size)
            arg_text = "".join(
                e.value for e in events if e.type == EventType.ARG_VALUE_CHUNK
            )
            results[chunk_size] = arg_text

        values = list(results.values())
        for v in values[1:]:
            assert v == values[0], f"Mismatch: {results}"

    def test_prefix_buffering_prevents_premature_emit(self):
        """Text like '<tool_' should be buffered, not emitted as content."""
        engine = StreamingParserEngine(_hermes_config(), tokenizer=None)

        events1 = engine.feed("<tool_", [])
        content_events = [e for e in events1 if e.type == EventType.TEXT_CHUNK]
        assert len(content_events) == 0, "Should buffer partial tag"

        events2 = engine.feed("call>", [])
        starts = [e for e in events2 if e.type == EventType.TOOL_CALL_START]
        assert len(starts) == 1

    def test_prefix_buffering_flush_on_mismatch(self):
        """Text like '<tool_box' should eventually flush as content."""
        engine = StreamingParserEngine(_hermes_config(), tokenizer=None)

        events1 = engine.feed("<tool_", [])
        assert len([e for e in events1 if e.type == EventType.TEXT_CHUNK]) == 0

        events2 = engine.feed("box>rest", [])
        events2.extend(engine.finish())
        content = "".join(e.value for e in events2 if e.type == EventType.TEXT_CHUNK)
        assert content == "<tool_box>rest"

    def test_reasoning_streaming(self):
        engine = StreamingParserEngine(_think_config(), tokenizer=None)
        events = self._feed_chars(engine, "<think>hmm</think>answer")

        reasoning = "".join(
            e.value for e in events if e.type == EventType.REASONING_CHUNK
        )
        content = "".join(e.value for e in events if e.type == EventType.TEXT_CHUNK)
        assert "hmm" in reasoning
        assert "answer" in content

    def test_text_between_tool_calls(self):
        engine = StreamingParserEngine(_hermes_config(), tokenizer=None)
        text = (
            'Hi<tool_call>{"name":"a"}</tool_call>'
            'mid<tool_call>{"name":"b"}</tool_call>end'
        )
        events = self._feed_chunks(engine, text, 3)

        texts = "".join(e.value for e in events if e.type == EventType.TEXT_CHUNK)
        assert "Hi" in texts
        assert "mid" in texts
        assert "end" in texts

        starts = [e for e in events if e.type == EventType.TOOL_CALL_START]
        assert len(starts) == 2

    def test_json_args_no_premature_close_brace(self):
        """Closing braces of the top-level JSON shouldn't be streamed
        until confirmed by the end tag."""
        engine = StreamingParserEngine(_hermes_config(), tokenizer=None)

        engine.feed("<tool_call>", [])
        events = engine.feed('{"name": "f"}', [])

        arg_text = "".join(
            e.value for e in events if e.type == EventType.ARG_VALUE_CHUNK
        )
        assert "}" not in arg_text, "Top-level } should be held back"

        events2 = engine.feed("</tool_call>", [])
        arg_text2 = "".join(
            e.value for e in events2 if e.type == EventType.ARG_VALUE_CHUNK
        )
        assert "}" in arg_text2, "} should flush on end tag"


_START_ID = 50
_END_ID = 51


def _make_think_tokenizer():
    tok = MagicMock()
    tok.encode.return_value = [1, 2, 3]
    tok.get_vocab.return_value = {"<think>": _START_ID, "</think>": _END_ID}
    tok.decode.side_effect = lambda ids: {
        _START_ID: "<think>",
        _END_ID: "</think>",
    }.get(ids[0], f"tok{ids[0]}")
    return tok


class TestLexerBufferFlush:
    """Lexer buffer must be flushed before PreLexedTerminal transitions."""

    def test_buffered_prefix_emitted_in_current_state(self):
        """Text buffered by the lexer (e.g. '<') must be emitted as
        REASONING_CHUNK before THINK_END transitions to CONTENT."""
        engine = StreamingParserEngine(_think_config(), _make_think_tokenizer())

        events = engine.feed("<think>", [_START_ID])
        assert any(e.type == EventType.REASONING_START for e in events)

        events = engine.feed("reasoning text<", [])
        reasoning_text = "".join(
            e.value for e in events if e.type == EventType.REASONING_CHUNK
        )
        assert "reasoning text" in reasoning_text

        events = engine.feed("</think>", [_END_ID])
        event_types = [e.type for e in events]
        if EventType.REASONING_CHUNK in event_types:
            rc_idx = event_types.index(EventType.REASONING_CHUNK)
            re_idx = event_types.index(EventType.REASONING_END)
            assert rc_idx < re_idx, (
                "'<' must be emitted as REASONING_CHUNK before REASONING_END"
            )
            flushed = events[rc_idx].value
            assert "<" in flushed

    def test_empty_buffer_no_extra_events(self):
        """When the lexer buffer is empty, flushing is a no-op."""
        engine = StreamingParserEngine(_think_config(), _make_think_tokenizer())

        engine.feed("<think>", [_START_ID])
        engine.feed("clean text", [])

        events = engine.feed("</think>", [_END_ID])
        assert any(e.type == EventType.REASONING_END for e in events)
        chunk_events = [e for e in events if e.type == EventType.REASONING_CHUNK]
        assert all(e.value for e in chunk_events)
