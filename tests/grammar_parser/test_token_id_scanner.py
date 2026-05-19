# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for TokenIDScanner, focusing on hold-back text recovery.

Uses gemma4_config for all end-to-end engine tests, covering
reasoning channels, tool calls, and combined flows."""

from unittest.mock import MagicMock

import pytest

from vllm.grammar_parser.events import EventType
from vllm.grammar_parser.token_id_scanner import (
    PreLexedTerminal,
    TextChunk,
    TokenIDScanner,
)

CHANNEL_START = "<|channel>"
CHANNEL_END = "<channel|>"
CHANNEL_START_ID = 100
CHANNEL_END_ID = 101
REGULAR_TOKEN_ID = 200


@pytest.fixture
def tokenizer():
    tok = MagicMock()
    tok.get_vocab.return_value = {
        CHANNEL_START: CHANNEL_START_ID,
        CHANNEL_END: CHANNEL_END_ID,
    }
    tok.decode.side_effect = lambda ids: {
        CHANNEL_START_ID: CHANNEL_START,
        CHANNEL_END_ID: CHANNEL_END,
        REGULAR_TOKEN_ID: "regular",
    }.get(ids[0], f"<unk:{ids[0]}>")
    return tok


@pytest.fixture
def scanner(tokenizer):
    return TokenIDScanner(
        token_id_to_terminal={
            CHANNEL_START_ID: "THINK_START",
            CHANNEL_END_ID: "THINK_END",
        },
        tokenizer=tokenizer,
    )


class TestHoldbackTextRecovery:
    def test_holdback_text_with_special_token_text_absent(self, scanner):
        """delta_text has hold-back text but the special token's text is
        NOT in delta_text (stripped by skip_special_tokens).  Text and
        terminal are both emitted immediately."""
        result = scanner.scan(
            delta_text="processed is appropriate.",
            delta_token_ids=[CHANNEL_END_ID],
        )

        assert len(result) == 2
        assert isinstance(result[0], TextChunk)
        assert result[0].text == "processed is appropriate."
        assert isinstance(result[1], PreLexedTerminal)
        assert result[1].terminal == "THINK_END"

        # Second scan: terminal text arrives (detokenizer flushes).
        # Since the terminal already fired, this is just content text.
        result2 = scanner.scan(
            delta_text="<channel|>Understood.",
            delta_token_ids=[20, 21],
        )
        texts = [r.text for r in result2 if isinstance(r, TextChunk)]
        combined = "".join(texts)
        assert "<channel|>" in combined or "Understood." in combined

    def test_holdback_text_with_special_token_text_present(self, scanner):
        """delta_text includes hold-back text AND the special token text."""
        result = scanner.scan(
            delta_text="holdback text<channel|>",
            delta_token_ids=[CHANNEL_END_ID],
        )

        assert len(result) == 2
        assert isinstance(result[0], TextChunk)
        assert result[0].text == "holdback text"
        assert isinstance(result[1], PreLexedTerminal)
        assert result[1].terminal == "THINK_END"

    def test_no_holdback_text(self, scanner):
        """delta_text is exactly the special token text — no hold-back."""
        result = scanner.scan(
            delta_text="<channel|>",
            delta_token_ids=[CHANNEL_END_ID],
        )

        assert len(result) == 1
        assert isinstance(result[0], PreLexedTerminal)
        assert result[0].terminal == "THINK_END"

    def test_empty_delta_text(self, scanner):
        """delta_text is empty — no text to lose, terminal emits now."""
        result = scanner.scan(
            delta_text="",
            delta_token_ids=[CHANNEL_END_ID],
        )

        assert len(result) == 1
        assert isinstance(result[0], PreLexedTerminal)
        assert result[0].terminal == "THINK_END"

    def test_empty_delta_text_drops_individual_decode_text(self, tokenizer):
        """delta_text="" with multiple tokens including special: only
        PreLexedTerminals are emitted — individually-decoded TextChunks
        are dropped since the detokenizer hasn't confirmed them yet."""
        tool_start_id = 400
        tok_a = 201
        tok_b = 202
        tokenizer.decode.side_effect = lambda ids: {
            tool_start_id: "<|tool_call>",
            tok_a: "call:",
            tok_b: "get_weather",
        }.get(ids[0], "?")

        scanner = TokenIDScanner(
            token_id_to_terminal={tool_start_id: "TOOL_START"},
            tokenizer=tokenizer,
        )

        result = scanner.scan(
            delta_text="",
            delta_token_ids=[tool_start_id, tok_a, tok_b],
        )

        assert len(result) == 1
        assert isinstance(result[0], PreLexedTerminal)
        assert result[0].terminal == "TOOL_START"

    def test_holdback_before_start_tag(self, scanner):
        """Hold-back text before a reasoning start tag."""
        result = scanner.scan(
            delta_text="prefix text<|channel>",
            delta_token_ids=[CHANNEL_START_ID],
        )

        assert len(result) == 2
        assert isinstance(result[0], TextChunk)
        assert result[0].text == "prefix text"
        assert isinstance(result[1], PreLexedTerminal)
        assert result[1].terminal == "THINK_START"

    def test_multi_token_batch_special_in_middle(self, scanner, tokenizer):
        """Stream-interval > 1: batch has regular tokens + special token.
        delta_text differs from individual decodes (context-dependent)."""
        tok_a = 201
        tok_b = 202
        tokenizer.decode.side_effect = lambda ids: {
            tok_a: "wordA",
            tok_b: "wordB",
            CHANNEL_END_ID: CHANNEL_END,
        }.get(ids[0], "?")

        scanner_multi = TokenIDScanner(
            token_id_to_terminal={CHANNEL_END_ID: "THINK_END"},
            tokenizer=tokenizer,
        )

        result = scanner_multi.scan(
            delta_text="holdback wordA<channel|> wordB",
            delta_token_ids=[tok_a, CHANNEL_END_ID, tok_b],
        )

        texts = [r.text for r in result if isinstance(r, TextChunk)]
        terminals = [r.terminal for r in result if isinstance(r, PreLexedTerminal)]
        assert "THINK_END" in terminals
        assert "holdback wordA" in "".join(texts)

    def test_multi_token_batch_special_token_text_absent(self, scanner, tokenizer):
        """Stream-interval > 1: batch has regular + special token, but
        delta_text doesn't contain the special token text at all
        (held back by detokenizer along with trailing regular tokens).
        Terminal is deferred, text is preserved."""
        tok_a = 201
        tok_b = 202
        tokenizer.decode.side_effect = lambda ids: {
            tok_a: "alpha",
            tok_b: "beta",
            CHANNEL_END_ID: CHANNEL_END,
        }.get(ids[0], "?")

        scanner_multi = TokenIDScanner(
            token_id_to_terminal={CHANNEL_END_ID: "THINK_END"},
            tokenizer=tokenizer,
        )

        result = scanner_multi.scan(
            delta_text="holdback alpha",
            delta_token_ids=[tok_a, CHANNEL_END_ID, tok_b],
        )

        # Terminal fires immediately with holdback text emitted first.
        pre_lexed = [r for r in result if isinstance(r, PreLexedTerminal)]
        assert len(pre_lexed) == 1
        assert pre_lexed[0].terminal == "THINK_END"
        text_chunks = [r for r in result if isinstance(r, TextChunk)]
        combined = "".join(t.text for t in text_chunks)
        assert "holdback" in combined

        # Next delta: terminal text arrives (detokenizer flushes).
        # Since the terminal already fired, this is just content text.
        result2 = scanner_multi.scan(
            delta_text="<channel|> more text",
            delta_token_ids=[300],
        )
        text_chunks2 = [r for r in result2 if isinstance(r, TextChunk)]
        combined2 = "".join(t.text for t in text_chunks2)
        assert "more text" in combined2

    def test_holdback_with_content_after_special_token(self, tokenizer):
        """delta_text has hold-back + special token + content after,
        with corresponding token IDs for all parts."""
        tok_content = 210
        tokenizer.decode.side_effect = lambda ids: {
            CHANNEL_END_ID: CHANNEL_END,
            tok_content: "content start",
        }.get(ids[0], "?")

        scanner = TokenIDScanner(
            token_id_to_terminal={CHANNEL_END_ID: "THINK_END"},
            tokenizer=tokenizer,
        )

        result = scanner.scan(
            delta_text="reasoning end.<channel|>content start",
            delta_token_ids=[CHANNEL_END_ID, tok_content],
        )

        pre_lexed = [r for r in result if isinstance(r, PreLexedTerminal)]
        assert len(pre_lexed) == 1
        assert pre_lexed[0].terminal == "THINK_END"

        text_chunks = [r for r in result if isinstance(r, TextChunk)]
        combined = "".join(t.text for t in text_chunks)
        assert "reasoning end." in combined


class TestTerminalWithAbsentText:
    """Tests for terminal emission when the special token text is
    absent from delta_text (stripped by skip_special_tokens=True).
    With token_id_text_in_delta=False (default), terminals fire
    immediately instead of being deferred."""

    def test_terminal_fires_immediately_text_absent(self, scanner):
        """Terminal fires immediately even when its text is absent.
        Holdback text is emitted first as TextChunk."""
        r1 = scanner.scan(
            delta_text="reasoning tail.",
            delta_token_ids=[CHANNEL_END_ID],
        )
        assert len(r1) == 2
        assert isinstance(r1[0], TextChunk)
        assert r1[0].text == "reasoning tail."
        assert isinstance(r1[1], PreLexedTerminal)
        assert r1[1].terminal == "THINK_END"

        # Next delta: terminal text arrives (detokenizer flushes).
        # Since the terminal already fired, this is just content text.
        r2 = scanner.scan(
            delta_text="<channel|>Content here.",
            delta_token_ids=[20],
        )
        texts = [r.text for r in r2 if isinstance(r, TextChunk)]
        combined = "".join(texts)
        assert "Content here." in combined

    def test_terminal_fires_immediately_no_flush_needed(self, scanner):
        """Terminal fires immediately; flush_pending returns nothing."""
        r1 = scanner.scan(
            delta_text="some holdback text",
            delta_token_ids=[CHANNEL_END_ID],
        )
        assert len(r1) == 2
        assert isinstance(r1[0], TextChunk)
        assert r1[0].text == "some holdback text"
        assert isinstance(r1[1], PreLexedTerminal)
        assert r1[1].terminal == "THINK_END"

        flushed = scanner.flush_pending()
        assert len(flushed) == 0

    def test_terminal_fires_immediately_with_holdback_and_later_content(self, scanner):
        """Holdback text + stripped terminal fires immediately.
        Next delta's content is processed normally."""
        r1 = scanner.scan(
            delta_text="the server is responding.",
            delta_token_ids=[CHANNEL_END_ID],
        )
        assert len(r1) == 2
        assert isinstance(r1[0], TextChunk)
        assert r1[0].text == "the server is responding."
        assert isinstance(r1[1], PreLexedTerminal)
        assert r1[1].terminal == "THINK_END"

        r2 = scanner.scan(
            delta_text="<channel|>Understood, I'll help.",
            delta_token_ids=[30, 31],
        )
        texts = [r.text for r in r2 if isinstance(r, TextChunk)]
        combined = "".join(texts)
        assert "Understood, I'll help." in combined

    def test_no_deferred_when_text_present(self, scanner):
        """No deferral when the terminal's text IS in delta_text."""
        r1 = scanner.scan(
            delta_text="tail<channel|>",
            delta_token_ids=[CHANNEL_END_ID],
        )
        terminals = [r for r in r1 if isinstance(r, PreLexedTerminal)]
        assert len(terminals) == 1

        assert len(scanner._deferred_terminals) == 0

    def test_immediate_terminal_does_not_swallow_current_tokens(self, tokenizer):
        """When a terminal fires immediately (text absent), the next
        delta's token IDs must still be scanned for new special tokens."""
        tool_call_id = 500
        tokenizer.decode.side_effect = lambda ids: {
            CHANNEL_END_ID: CHANNEL_END,
            tool_call_id: "<|tool_call>",
            201: "content ",
        }.get(ids[0], f"tok{ids[0]}")

        scanner = TokenIDScanner(
            token_id_to_terminal={
                CHANNEL_END_ID: "THINK_END",
                tool_call_id: "TOOL_START",
            },
            tokenizer=tokenizer,
        )

        # Delta 1: reasoning holdback + channel end (text absent).
        # Terminal fires immediately.
        r1 = scanner.scan(
            delta_text="reasoning tail",
            delta_token_ids=[CHANNEL_END_ID],
        )
        assert len(r1) == 2
        assert isinstance(r1[0], TextChunk)
        assert r1[0].text == "reasoning tail"
        assert isinstance(r1[1], PreLexedTerminal)
        assert r1[1].terminal == "THINK_END"

        # Delta 2: channel end text (already consumed) + content + tool call.
        r2 = scanner.scan(
            delta_text="<channel|>content <|tool_call>",
            delta_token_ids=[201, tool_call_id],
        )

        terminals = [r for r in r2 if isinstance(r, PreLexedTerminal)]
        terminal_names = [t.terminal for t in terminals]
        assert "TOOL_START" in terminal_names


class TestDropTokens:
    def test_drop_token_with_holdback(self, tokenizer):
        """Drop tokens stripped from delta_text, hold-back text preserved.
        Terminal fires immediately with text emitted first."""
        drop_id = 300
        tokenizer.decode.side_effect = lambda ids: {
            CHANNEL_END_ID: CHANNEL_END,
            drop_id: "<eos>",
        }.get(ids[0], "?")

        scanner = TokenIDScanner(
            token_id_to_terminal={CHANNEL_END_ID: "THINK_END"},
            tokenizer=tokenizer,
            drop_token_ids={drop_id},
        )

        result = scanner.scan(
            delta_text="holdback<eos>",
            delta_token_ids=[drop_id, CHANNEL_END_ID],
        )

        # Terminal fires immediately; holdback text emitted first.
        assert len(result) == 2
        assert isinstance(result[0], TextChunk)
        assert "holdback" in result[0].text
        assert "<eos>" not in result[0].text
        assert isinstance(result[1], PreLexedTerminal)
        assert result[1].terminal == "THINK_END"

        # Nothing left to flush.
        flushed = scanner.flush_pending()
        assert len(flushed) == 0


class TestEndToEndReasoningHoldback:
    """End-to-end tests through the full parser engine simulating
    stream-interval > 1 and detokenizer hold-back, using
    gemma4_config."""

    def test_reasoning_content_not_truncated(self):
        from vllm.grammar_parser.grammars.gemma4 import (
            gemma4_config,
        )
        from vllm.grammar_parser.parser_engine import StreamingParserEngine

        config = gemma4_config()
        tok = MagicMock()
        vocab = {
            CHANNEL_START: CHANNEL_START_ID,
            CHANNEL_END: CHANNEL_END_ID,
        }
        tok.get_vocab.return_value = vocab
        tok.decode.side_effect = lambda ids: {
            CHANNEL_START_ID: CHANNEL_START,
            CHANNEL_END_ID: CHANNEL_END,
        }.get(ids[0], f"tok{ids[0]}")

        engine = StreamingParserEngine(config, tok)
        all_events = []

        # Delta 1: channel start token (text includes start tag)
        all_events.extend(engine.feed(CHANNEL_START, [CHANNEL_START_ID]))

        # Delta 2: reasoning text (normal content, no special tokens)
        all_events.extend(
            engine.feed(
                "thought\nThe request was received and ",
                [10, 11, 12, 13, 14],
            )
        )

        # Delta 3: MORE reasoning text, the detokenizer held some back.
        # Then channel end token arrives in token_ids, but its text
        # is NOT in delta_text (held back by detokenizer).
        # delta_text = previously held-back reasoning text only.
        all_events.extend(
            engine.feed(
                "processed is appropriate.",
                [CHANNEL_END_ID],
            )
        )

        # Delta 4: detokenizer flushes held-back channel end text
        # plus new content tokens.
        all_events.extend(
            engine.feed(
                "<channel|>Understood.",
                [20, 21],
            )
        )

        all_events.extend(engine.finish())

        reasoning_text = "".join(
            e.value for e in all_events if e.type == EventType.REASONING_CHUNK
        )
        content_text = "".join(
            e.value for e in all_events if e.type == EventType.TEXT_CHUNK
        )

        assert "processed is appropriate." in reasoning_text
        assert "Understood." in content_text

    def test_backtick_content_not_truncated(self):
        """Reproduces the hostname backtick truncation case."""
        from vllm.grammar_parser.grammars.gemma4 import (
            gemma4_config,
        )
        from vllm.grammar_parser.parser_engine import StreamingParserEngine

        config = gemma4_config()
        tok = MagicMock()
        vocab = {
            CHANNEL_START: CHANNEL_START_ID,
            CHANNEL_END: CHANNEL_END_ID,
        }
        tok.get_vocab.return_value = vocab
        tok.decode.side_effect = lambda ids: {
            CHANNEL_START_ID: CHANNEL_START,
            CHANNEL_END_ID: CHANNEL_END,
        }.get(ids[0], f"tok{ids[0]}")

        engine = StreamingParserEngine(config, tok)
        all_events = []

        all_events.extend(engine.feed(CHANNEL_START, [CHANNEL_START_ID]))
        all_events.extend(
            engine.feed(
                "thought\n1/10 completed. Next: ",
                [10, 11, 12, 13],
            )
        )

        # Hold-back text includes backtick content; channel end text
        # absent from delta_text.
        all_events.extend(
            engine.feed(
                "`hostname`.\n",
                [CHANNEL_END_ID],
            )
        )

        # Next delta flushes channel end + tool call start
        all_events.extend(
            engine.feed(
                "<channel|>tool output",
                [20, 21],
            )
        )

        all_events.extend(engine.finish())

        reasoning_text = "".join(
            e.value for e in all_events if e.type == EventType.REASONING_CHUNK
        )

        assert "`hostname`." in reasoning_text


# ---------------------------------------------------------------------------
# Token IDs for multi-token boundary tests
# ---------------------------------------------------------------------------
_CHANNEL_START_TAG = "<|channel>"
_CHANNEL_END_TAG = "<channel|>"
_TOOL_START_TAG = "<|tool_call>"
_TOOL_END_TAG = "<tool_call|>"
_QUOTE_TAG = '<|"|>'

_CHANNEL_START_TID = 100
_CHANNEL_END_TID = 101
_TOOL_START_TID = 102
_TOOL_END_TID = 103
_QUOTE_TID = 104
_TOK = list(range(200, 215))


def _gemma4_vocab() -> dict[str, int]:
    """Vocab mapping for Gemma4 special tokens."""
    return {
        _CHANNEL_START_TAG: _CHANNEL_START_TID,
        _CHANNEL_END_TAG: _CHANNEL_END_TID,
        _TOOL_START_TAG: _TOOL_START_TID,
        _TOOL_END_TAG: _TOOL_END_TID,
        _QUOTE_TAG: _QUOTE_TID,
    }


def _make_gemma4_tokenizer(
    extra_decode: dict[int, str] | None = None,
) -> MagicMock:
    """Mock tokenizer for Gemma4 with configurable regular-token text."""
    special = {
        _CHANNEL_START_TID: _CHANNEL_START_TAG,
        _CHANNEL_END_TID: _CHANNEL_END_TAG,
        _TOOL_START_TID: _TOOL_START_TAG,
        _TOOL_END_TID: _TOOL_END_TAG,
        _QUOTE_TID: _QUOTE_TAG,
    }
    decode_map = {**special, **(extra_decode or {})}

    tok = MagicMock()
    tok.get_vocab.return_value = _gemma4_vocab()
    tok.decode.side_effect = lambda ids: decode_map.get(ids[0], f"tok{ids[0]}")
    return tok


def _collect_events(engine, deltas):
    """Feed all (delta_text, delta_token_ids) pairs and return events."""
    from vllm.grammar_parser.events import SemanticEvent

    all_events: list[SemanticEvent] = []
    for delta_text, delta_token_ids in deltas:
        all_events.extend(engine.feed(delta_text, delta_token_ids))
    all_events.extend(engine.finish())
    return all_events


def _reasoning_text(events) -> str:
    return "".join(e.value for e in events if e.type == EventType.REASONING_CHUNK)


def _content_text(events) -> str:
    return "".join(e.value for e in events if e.type == EventType.TEXT_CHUNK)


def _arg_text(events) -> str:
    return "".join(e.value for e in events if e.type == EventType.ARG_VALUE_CHUNK)


def _has_event(events, event_type) -> bool:
    return any(e.type == event_type for e in events)


class TestMultiTokenBoundaryPreservation:
    """End-to-end tests verifying no text is lost at state boundaries
    when multiple tokens arrive per delta with detokenizer holdback.

    Uses gemma4_config which covers both reasoning and tool calls
    in a single engine."""

    # -- Unique edge cases from channel-only tests -------------------------

    def test_empty_delta_text_at_channel_end_unified(self):
        """delta_text="" when CHANNEL_END arrives; text comes later.

        When delta_text is empty the PreLexedTerminal fires immediately.
        The tag text then appears in the *next* delta's delta_text and
        may be echoed by the lexer — that is accepted.  The invariant
        we enforce is that no reasoning or content text is *lost*."""
        from vllm.grammar_parser.grammars.gemma4 import (
            gemma4_config,
        )
        from vllm.grammar_parser.parser_engine import StreamingParserEngine

        tok = _make_gemma4_tokenizer()
        engine = StreamingParserEngine(gemma4_config(), tok)

        events = _collect_events(
            engine,
            [
                # CHANNEL_START with empty delta_text (detokenizer hasn't
                # flushed yet).
                ("", [_CHANNEL_START_TID]),
                # Detokenizer flushes start tag text + reasoning.
                ("<|channel>thought\nSome reasoning.", [_TOK[0], _TOK[1]]),
                # CHANNEL_END with empty delta_text.
                ("", [_CHANNEL_END_TID]),
                # Detokenizer flushes end tag text + content.
                ("<channel|>Final answer.", [_TOK[2], _TOK[3]]),
            ],
        )

        reasoning = _reasoning_text(events)
        content = _content_text(events)
        assert "Some reasoning." in reasoning
        assert "Final answer." in content
        assert _has_event(events, EventType.REASONING_START)
        assert _has_event(events, EventType.REASONING_END)

    def test_deferred_channel_end_flushed_at_finish_unified(self):
        """Deferred CHANNEL_END flushed at end-of-stream via finish()."""
        from vllm.grammar_parser.grammars.gemma4 import (
            gemma4_config,
        )
        from vllm.grammar_parser.parser_engine import StreamingParserEngine

        tok = _make_gemma4_tokenizer()
        engine = StreamingParserEngine(gemma4_config(), tok)

        events = _collect_events(
            engine,
            [
                (_CHANNEL_START_TAG, [_CHANNEL_START_TID]),
                ("thought\nReasoning text.", [_TOK[0]]),
                # Holdback + deferred — no more deltas after this.
                (" Final thought.", [_CHANNEL_END_TID]),
            ],
        )

        reasoning = _reasoning_text(events)
        assert "Reasoning text. Final thought." in reasoning
        assert _has_event(events, EventType.REASONING_END)

    # -- Cross-engine: reasoning → tool call in single unified engine -----

    def test_reasoning_to_tool_call_handoff_unified(self):
        """Full reasoning → content → tool call through a single engine.

        Verifies the unified config handles the complete flow without
        needing separate reasoning and tool-call engines."""
        from vllm.grammar_parser.grammars.gemma4 import (
            gemma4_config,
        )
        from vllm.grammar_parser.parser_engine import StreamingParserEngine

        tok = _make_gemma4_tokenizer()
        engine = StreamingParserEngine(gemma4_config(), tok)

        events = _collect_events(
            engine,
            [
                # Reasoning section
                (_CHANNEL_START_TAG, [_CHANNEL_START_TID]),
                ("thought\nI need to check the weather.", [_TOK[0], _TOK[1], _TOK[2]]),
                (_CHANNEL_END_TAG, [_CHANNEL_END_TID]),
                # Content + tool call
                ("Let me call a tool.", [_TOK[3], _TOK[4]]),
                (_TOOL_START_TAG, [_TOOL_START_TID]),
                ("call:get_weather{city:", [_TOK[5], _TOK[6]]),
                ('<|"|>SF<|"|>}', [_QUOTE_TID, _TOK[7], _QUOTE_TID, _TOK[8]]),
                (_TOOL_END_TAG, [_TOOL_END_TID]),
            ],
        )

        reasoning = _reasoning_text(events)
        content = _content_text(events)

        assert "I need to check the weather." in reasoning
        assert "Let me call a tool." in content
        assert _has_event(events, EventType.REASONING_START)
        assert _has_event(events, EventType.REASONING_END)
        assert _has_event(events, EventType.TOOL_CALL_START)
        assert _has_event(events, EventType.TOOL_CALL_END)
        assert "SF" in _arg_text(events)

    def test_multiple_tool_calls_rapid_transitions_unified(self):
        """Two back-to-back tool calls in the unified config.

        Verifies tool_index tracking and text integrity — the key behavior
        lost when test_multiple_tool_calls_rapid_transitions was removed."""
        from vllm.grammar_parser.grammars.gemma4 import (
            gemma4_config,
        )
        from vllm.grammar_parser.parser_engine import StreamingParserEngine

        tok = _make_gemma4_tokenizer()
        engine = StreamingParserEngine(gemma4_config(), tok)

        events = _collect_events(
            engine,
            [
                # Tool call 0
                (_TOOL_START_TAG, [_TOOL_START_TID]),
                ("call:get_weather{city:", [_TOK[0], _TOK[1]]),
                ('<|"|>NYC<|"|>}', [_QUOTE_TID, _TOK[2], _QUOTE_TID, _TOK[3]]),
                (_TOOL_END_TAG, [_TOOL_END_TID]),
                # Tool call 1 immediately after
                (_TOOL_START_TAG, [_TOOL_START_TID]),
                ("call:get_time{tz:", [_TOK[4], _TOK[5]]),
                ('<|"|>EST<|"|>}', [_QUOTE_TID, _TOK[6], _QUOTE_TID, _TOK[7]]),
                (_TOOL_END_TAG, [_TOOL_END_TID]),
            ],
        )

        starts = [e for e in events if e.type == EventType.TOOL_CALL_START]
        ends = [e for e in events if e.type == EventType.TOOL_CALL_END]
        assert len(starts) == 2
        assert len(ends) == 2
        assert starts[0].tool_index == 0
        assert starts[1].tool_index == 1

        names = "".join(e.value for e in events if e.type == EventType.TOOL_NAME)
        assert "get_weather" in names
        assert "get_time" in names

    def test_deferred_channel_end_before_tool_call_unified(self):
        """CHANNEL_END deferred (text held back), then tool call follows.

        Covers the case where reasoning ends with holdback at <channel|>
        and a tool call fires in the same unified engine afterward — the
        compound scenario from the deleted
        test_reasoning_to_tool_call_with_deferred_channel_end."""
        from vllm.grammar_parser.grammars.gemma4 import (
            gemma4_config,
        )
        from vllm.grammar_parser.parser_engine import StreamingParserEngine

        tok = _make_gemma4_tokenizer()
        engine = StreamingParserEngine(gemma4_config(), tok)

        events = _collect_events(
            engine,
            [
                (_CHANNEL_START_TAG, [_CHANNEL_START_TID]),
                ("thought\nNeed to call a tool.", [_TOK[0], _TOK[1]]),
                # Holdback: reasoning tail in delta_text, CHANNEL_END text absent.
                (" Let me proceed.", [_CHANNEL_END_TID]),
                # Deferred CHANNEL_END resolves.
                (_CHANNEL_END_TAG, [_TOK[2]]),
                # Tool call follows
                (_TOOL_START_TAG, [_TOOL_START_TID]),
                ("call:get_weather{city:", [_TOK[3], _TOK[4]]),
                ('<|"|>Tokyo<|"|>}', [_QUOTE_TID, _TOK[5], _QUOTE_TID, _TOK[6]]),
                (_TOOL_END_TAG, [_TOOL_END_TID]),
            ],
        )

        reasoning = _reasoning_text(events)
        assert "Need to call a tool. Let me proceed." in reasoning
        assert _has_event(events, EventType.REASONING_END)
        assert _has_event(events, EventType.TOOL_CALL_START)
        assert _has_event(events, EventType.TOOL_CALL_END)
        assert "Tokyo" in _arg_text(events)


class TestStreamInterval10:
    """Tests that model ``--stream-interval 10`` behavior.

    With stream_interval=10 the output processor holds tokens until 10
    have accumulated, then emits them all at once.  ``delta_token_ids``
    contains ~10 token IDs and ``delta_text`` is a substring of the
    detokenizer's accumulated output — it includes hold-back text from
    *previous* batches and may or may not include special-token text.

    The critical difference from interval=1: a special token can land
    in the *middle* of a 10-token batch, meaning tokens before it belong
    to one parser state and tokens after belong to another, all arriving
    in a single ``feed()`` call."""

    def test_channel_end_mid_batch_text_present(self):
        """<channel|> lands at position 4 of a 10-token batch.

        delta_text includes all text: holdback from previous batch +
        reasoning text + <channel|> text + content text.  All in one
        feed() call with 10 token IDs."""
        from vllm.grammar_parser.grammars.gemma4 import (
            gemma4_config,
        )
        from vllm.grammar_parser.parser_engine import StreamingParserEngine

        tok = _make_gemma4_tokenizer({_TOK[i]: f"word{i} " for i in range(15)})
        engine = StreamingParserEngine(gemma4_config(), tok)

        events: list = []
        # Batch 1: channel start + first reasoning tokens (10 tokens)
        events.extend(
            engine.feed(
                "<|channel>thought\nword0 word1 word2 word3 word4 "
                "word5 word6 word7 word8 ",
                [
                    _CHANNEL_START_TID,
                    _TOK[0],
                    _TOK[1],
                    _TOK[2],
                    _TOK[3],
                    _TOK[4],
                    _TOK[5],
                    _TOK[6],
                    _TOK[7],
                    _TOK[8],
                ],
            )
        )

        # Batch 2: 10 tokens, <channel|> at position 4.
        # delta_text includes holdback from previous batch ("word9 ")
        # + reasoning tokens + <channel|> + content tokens.
        events.extend(
            engine.feed(
                "word9 word10 word11 <channel|>word12 word13 word14 word0 word1 word2 ",
                [
                    _TOK[9],
                    _TOK[10],
                    _TOK[11],
                    _CHANNEL_END_TID,
                    _TOK[12],
                    _TOK[13],
                    _TOK[14],
                    _TOK[0],
                    _TOK[1],
                    _TOK[2],
                ],
            )
        )

        events.extend(engine.finish())

        reasoning = _reasoning_text(events)
        content = _content_text(events)

        # Reasoning must include all text up to <channel|>.
        for w in ("word9", "word10", "word11"):
            assert w in reasoning, f"{w!r} missing from reasoning"

        # Content must include all text after <channel|>.
        for w in ("word12", "word13", "word14"):
            assert w in content, f"{w!r} missing from content"

        assert _has_event(events, EventType.REASONING_END)

    def test_channel_end_mid_batch_text_absent(self):
        """<channel|> at position 4 of 10-token batch, but its text is
        NOT in delta_text — detokenizer held it back.

        This is the core stream_interval>1 failure mode: the terminal
        is deferred, and tokens after it in the same batch have their
        individually-decoded text dropped (unreliable without
        delta_text confirmation)."""
        from vllm.grammar_parser.grammars.gemma4 import (
            gemma4_config,
        )
        from vllm.grammar_parser.parser_engine import StreamingParserEngine

        tok = _make_gemma4_tokenizer({_TOK[i]: f"word{i} " for i in range(15)})
        engine = StreamingParserEngine(gemma4_config(), tok)

        events: list = []
        # Batch 1: channel start + reasoning (10 tokens)
        events.extend(
            engine.feed(
                "<|channel>thought\nword0 word1 word2 word3 word4 "
                "word5 word6 word7 word8 ",
                [
                    _CHANNEL_START_TID,
                    _TOK[0],
                    _TOK[1],
                    _TOK[2],
                    _TOK[3],
                    _TOK[4],
                    _TOK[5],
                    _TOK[6],
                    _TOK[7],
                    _TOK[8],
                ],
            )
        )

        # Batch 2: 10 tokens, <channel|> at position 4.
        # delta_text has holdback + reasoning text but NOT the <channel|>
        # text or anything after — detokenizer held those back.
        events.extend(
            engine.feed(
                "word9 word10 word11 ",
                [
                    _TOK[9],
                    _TOK[10],
                    _TOK[11],
                    _CHANNEL_END_TID,
                    _TOK[12],
                    _TOK[13],
                    _TOK[14],
                    _TOK[0],
                    _TOK[1],
                    _TOK[2],
                ],
            )
        )

        # Batch 3: detokenizer flushes held-back text.
        events.extend(
            engine.feed(
                "<channel|>word12 word13 word14 word0 word1 word2 ",
                [_TOK[3], _TOK[4], _TOK[5]],
            )
        )

        events.extend(engine.finish())

        reasoning = _reasoning_text(events)
        content = _content_text(events)

        # All reasoning text preserved, including holdback "word9..word11".
        for w in ("word9", "word10", "word11"):
            assert w in reasoning, f"{w!r} missing from reasoning"

        # Content after <channel|> preserved.
        for w in ("word12", "word13", "word14"):
            assert w in content, f"{w!r} missing from content"

        assert _has_event(events, EventType.REASONING_END)

    def test_tool_end_mid_batch_text_absent_unified(self):
        """<tool_call|> at position 5 of 10-token batch, text absent.

        Same pattern as channel_end but for tool calls — verifies
        arg text isn't lost at tool-call end with large batches."""
        from vllm.grammar_parser.grammars.gemma4 import (
            gemma4_config,
        )
        from vllm.grammar_parser.parser_engine import StreamingParserEngine

        tok = _make_gemma4_tokenizer({_TOK[i]: f"w{i}" for i in range(15)})
        engine = StreamingParserEngine(gemma4_config(), tok)

        events: list = []
        # Batch 1: channel start (to enter unified flow) + tool start
        events.extend(
            engine.feed(
                _CHANNEL_START_TAG,
                [_CHANNEL_START_TID],
            )
        )
        events.extend(
            engine.feed(
                "thought\nNeed a tool.",
                [_TOK[0], _TOK[1]],
            )
        )
        # Reasoning → tool call directly (unified config feature)
        events.extend(
            engine.feed(
                _TOOL_START_TAG,
                [_TOOL_START_TID],
            )
        )
        events.extend(
            engine.feed(
                "call:get_weather{city:",
                [_TOK[2], _TOK[3], _TOK[4]],
            )
        )

        # Batch 2: 10 tokens, quote + value tokens + quote + close brace
        # + TOOL_END (text absent) + content tokens.
        # delta_text only has the arg text, not <tool_call|> or after.
        events.extend(
            engine.feed(
                '<|"|>San Francisco<|"|>}',
                [
                    _QUOTE_TID,
                    _TOK[5],
                    _TOK[6],
                    _QUOTE_TID,
                    _TOK[7],
                    _TOOL_END_TID,
                    _TOK[8],
                    _TOK[9],
                    _TOK[10],
                    _TOK[11],
                ],
            )
        )

        # Batch 3: detokenizer flushes <tool_call|> text + content.
        events.extend(
            engine.feed(
                "<tool_call|>w8w9w10w11w12",
                [_TOK[12], _TOK[13]],
            )
        )

        events.extend(engine.finish())

        assert _has_event(events, EventType.TOOL_CALL_END)
        assert "San Francisco" in _arg_text(events)

    def test_channel_end_and_tool_start_same_batch_unified(self):
        """Both <channel|> AND <|tool_call> in a single 10-token batch,
        handled by the unified config in one engine."""
        from vllm.grammar_parser.grammars.gemma4 import (
            gemma4_config,
        )
        from vllm.grammar_parser.parser_engine import StreamingParserEngine

        tok = _make_gemma4_tokenizer({_TOK[i]: f"w{i} " for i in range(15)})
        engine = StreamingParserEngine(gemma4_config(), tok)

        events: list = []

        # Batch 1: reasoning start + content (10 tokens)
        events.extend(
            engine.feed(
                "<|channel>thought\nw0 w1 w2 w3 w4 w5 w6 w7 w8 ",
                [
                    _CHANNEL_START_TID,
                    _TOK[0],
                    _TOK[1],
                    _TOK[2],
                    _TOK[3],
                    _TOK[4],
                    _TOK[5],
                    _TOK[6],
                    _TOK[7],
                    _TOK[8],
                ],
            )
        )

        # Batch 2: 10 tokens with <channel|> at pos 2, <|tool_call> at pos 4.
        # Unified engine handles both the reasoning end and tool call start.
        events.extend(
            engine.feed(
                "w9 w10 <channel|>w11 <|tool_call>",
                [
                    _TOK[9],
                    _TOK[10],
                    _CHANNEL_END_TID,
                    _TOK[11],
                    _TOOL_START_TID,
                    _TOK[12],
                    _TOK[13],
                    _TOK[14],
                    _TOK[0],
                    _TOK[1],
                ],
            )
        )
        events.extend(engine.finish())

        reasoning = _reasoning_text(events)

        assert "w9" in reasoning
        assert "w10" in reasoning
        assert _has_event(events, EventType.REASONING_END)
        assert _has_event(events, EventType.TOOL_CALL_START)

    def test_large_batch_holdback_spans_two_batches(self):
        """Realistic stream_interval=10: reasoning text accumulates
        across two 10-token batches, with <channel|> in the second
        batch and holdback from the first.

        This is the most realistic production scenario: the detokenizer
        has been accumulating text across multiple tokens, holds some
        back at the batch boundary, and the special token arrives in
        the next batch with the held-back text in delta_text."""
        from vllm.grammar_parser.grammars.gemma4 import (
            gemma4_config,
        )
        from vllm.grammar_parser.parser_engine import StreamingParserEngine

        tok = _make_gemma4_tokenizer({_TOK[i]: f"w{i} " for i in range(15)})
        engine = StreamingParserEngine(gemma4_config(), tok)

        events: list = []

        # Batch 1 (10 tokens): channel start + reasoning
        events.extend(
            engine.feed(
                "<|channel>thought\nThe user asked about machine learning "
                "and I need to think about the best approach to",
                [
                    _CHANNEL_START_TID,
                    _TOK[0],
                    _TOK[1],
                    _TOK[2],
                    _TOK[3],
                    _TOK[4],
                    _TOK[5],
                    _TOK[6],
                    _TOK[7],
                    _TOK[8],
                ],
            )
        )

        # Batch 2 (10 tokens): holdback from batch 1 (" explain")
        # + more reasoning + <channel|> (text absent from delta_text)
        # + content tokens (text also absent).
        events.extend(
            engine.feed(
                " explain this complex topic. Let me organize my thoughts.",
                [
                    _TOK[9],
                    _TOK[10],
                    _TOK[11],
                    _TOK[12],
                    _TOK[13],
                    _TOK[14],
                    _CHANNEL_END_TID,
                    _TOK[0],
                    _TOK[1],
                    _TOK[2],
                ],
            )
        )

        # Batch 3 (10 tokens): detokenizer flushes <channel|> text
        # + content from batch 2 + new content.
        events.extend(
            engine.feed(
                "<channel|>w0 w1 w2 Here is what I recommend: start with "
                "the fundamentals and build up from there.",
                [
                    _TOK[3],
                    _TOK[4],
                    _TOK[5],
                    _TOK[6],
                    _TOK[7],
                    _TOK[8],
                    _TOK[9],
                    _TOK[10],
                    _TOK[11],
                    _TOK[12],
                ],
            )
        )

        events.extend(engine.finish())

        reasoning = _reasoning_text(events)
        content = _content_text(events)

        # The held-back reasoning text must be preserved.
        assert "organize my thoughts." in reasoning
        assert "explain" in reasoning

        # Content after <channel|> must be present.
        assert "recommend" in content

        assert _has_event(events, EventType.REASONING_START)
        assert _has_event(events, EventType.REASONING_END)
