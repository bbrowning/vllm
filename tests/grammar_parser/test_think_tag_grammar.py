# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the parameterized think-tag reasoning grammar config.

Validates that a single parameterized grammar config correctly handles
the <think>/</think> pattern used by 12+ models, including variant
tag formats.
"""

from unittest.mock import MagicMock

import pytest

from vllm.grammar_parser.adapter import GrammarReasoningParser
from vllm.grammar_parser.events import EventType
from vllm.grammar_parser.grammars.think_tag import think_tag_config
from vllm.grammar_parser.parser_engine import StreamingParserEngine
from vllm.grammar_parser.token_id_scanner import (
    PreLexedTerminal,
    TextChunk,
    TokenIDScanner,
)

# Token IDs used in tests
_START_ID = 50
_END_ID = 51
_DROP_ID = 52
_TEXT_ID = 100


def _make_tokenizer(start_tag: str, end_tag: str):
    tokenizer = MagicMock()
    tokenizer.encode.return_value = [1, 2, 3]
    vocab = {start_tag: _START_ID, end_tag: _END_ID}
    tokenizer.get_vocab.return_value = vocab
    tokenizer.decode.side_effect = lambda ids: "".join(
        chr(i) if i < 128 else f"<{i}>" for i in ids
    )
    return tokenizer


def _make_tokenizer_with_text_map(
    start_tag: str,
    end_tag: str,
    token_text_map: dict[int, str],
):
    """Tokenizer mock where decode([tid]) returns a specific string."""
    tokenizer = MagicMock()
    tokenizer.encode.return_value = [1, 2, 3]
    vocab = {start_tag: _START_ID, end_tag: _END_ID}
    tokenizer.get_vocab.return_value = vocab

    def _decode(ids):
        parts = []
        for tid in ids:
            if tid in token_text_map:
                parts.append(token_text_map[tid])
            elif tid < 128:
                parts.append(chr(tid))
            else:
                parts.append(f"<{tid}>")
        return "".join(parts)

    tokenizer.decode.side_effect = _decode
    return tokenizer


def _stream_and_collect(
    parser: GrammarReasoningParser,
    chunks: list[str],
) -> tuple[str, str]:
    """Feed chunks through streaming and collect reasoning/content."""
    reasoning_parts: list[str] = []
    content_parts: list[str] = []
    prev_text = ""
    prev_ids: list[int] = []
    for chunk in chunks:
        cur_text = prev_text + chunk
        cur_ids = prev_ids + [0]
        delta = parser.extract_reasoning_streaming(
            previous_text=prev_text,
            current_text=cur_text,
            delta_text=chunk,
            previous_token_ids=tuple(prev_ids),
            current_token_ids=tuple(cur_ids),
            delta_token_ids=(0,),
        )
        if delta:
            if delta.reasoning:
                reasoning_parts.append(delta.reasoning)
            if delta.content:
                content_parts.append(delta.content)
        prev_text = cur_text
        prev_ids = cur_ids
    return "".join(reasoning_parts), "".join(content_parts)


class TestDefaultThinkTags:
    """Tests with default <think>/</think> tags (DeepSeekR1, Qwen3, etc.)."""

    @pytest.fixture
    def parser(self):
        config = think_tag_config()
        tokenizer = _make_tokenizer("<think>", "</think>")
        return GrammarReasoningParser(tokenizer, grammar_config=config)

    def test_reasoning_then_content(self, parser):
        text = "<think>Let me analyze this.</think>The answer is 42."
        reasoning, content = parser.extract_reasoning(text, None)
        assert reasoning == "Let me analyze this."
        assert content == "The answer is 42."

    def test_reasoning_only(self, parser):
        text = "<think>Still thinking...</think>"
        reasoning, content = parser.extract_reasoning(text, None)
        assert reasoning == "Still thinking..."
        assert content is None

    def test_content_only(self, parser):
        text = "Hello, no reasoning here."
        reasoning, content = parser.extract_reasoning(text, None)
        assert reasoning is None
        assert content == "Hello, no reasoning here."

    def test_multiline_reasoning(self, parser):
        text = (
            "<think>Step 1: parse the input.\n"
            "Step 2: compute the result.\n"
            "Step 3: format output.</think>"
            "The result is 7."
        )
        reasoning, content = parser.extract_reasoning(text, None)
        assert "Step 1" in reasoning
        assert "Step 3" in reasoning
        assert content == "The result is 7."

    def test_is_reasoning_end(self, parser):
        end_id = parser._reasoning_end_token_id
        start_id = parser._reasoning_start_token_id
        assert end_id is not None

        assert parser.is_reasoning_end([start_id, 1, 2, end_id])
        assert not parser.is_reasoning_end([start_id, 1, 2])
        assert not parser.is_reasoning_end([end_id, start_id, 1])

    def test_extract_content_ids(self, parser):
        end_id = parser._reasoning_end_token_id
        result = parser.extract_content_ids([10, 20, end_id, 30, 40])
        assert result == [30, 40]

    def test_streaming_reasoning_then_content(self, parser):
        reasoning, content = _stream_and_collect(
            parser, ["<think>", "thinking", " hard", "</think>", "done"]
        )
        assert reasoning == "thinking hard"
        assert content == "done"


class TestSeedOSSTags:
    """Tests with <seed:think>/</seed:think> tags."""

    @pytest.fixture
    def parser(self):
        config = think_tag_config(
            start_tag="<seed:think>",
            end_tag="</seed:think>",
            name="seedoss",
        )
        tokenizer = _make_tokenizer("<seed:think>", "</seed:think>")
        return GrammarReasoningParser(tokenizer, grammar_config=config)

    def test_reasoning_extraction(self, parser):
        text = "<seed:think>analyzing...</seed:think>Result: yes."
        reasoning, content = parser.extract_reasoning(text, None)
        assert reasoning == "analyzing..."
        assert content == "Result: yes."

    def test_streaming(self, parser):
        reasoning, content = _stream_and_collect(
            parser, ["<seed:think>", "step 1", "</seed:think>", "answer"]
        )
        assert reasoning == "step 1"
        assert content == "answer"


class TestMistralTags:
    """Tests with [THINK]/[/THINK] tags."""

    @pytest.fixture
    def parser(self):
        config = think_tag_config(
            start_tag="[THINK]",
            end_tag="[/THINK]",
            name="mistral",
        )
        tokenizer = _make_tokenizer("[THINK]", "[/THINK]")
        return GrammarReasoningParser(tokenizer, grammar_config=config)

    def test_reasoning_extraction(self, parser):
        text = "[THINK]Let me reason.[/THINK]The answer."
        reasoning, content = parser.extract_reasoning(text, None)
        assert reasoning == "Let me reason."
        assert content == "The answer."

    def test_streaming(self, parser):
        reasoning, content = _stream_and_collect(
            parser, ["[THINK]", "reasoning", "[/THINK]", "content"]
        )
        assert reasoning == "reasoning"
        assert content == "content"


class TestEmptyReasoning:
    """Edge case: empty reasoning block."""

    @pytest.fixture
    def parser(self):
        config = think_tag_config()
        tokenizer = _make_tokenizer("<think>", "</think>")
        return GrammarReasoningParser(tokenizer, grammar_config=config)

    def test_empty_think_block(self, parser):
        text = "<think></think>The answer."
        reasoning, content = parser.extract_reasoning(text, None)
        assert content == "The answer."

    def test_content_before_think(self, parser):
        """Some models emit content before the think block."""
        text = "Hmm, <think>let me think</think>OK got it."
        reasoning, content = parser.extract_reasoning(text, None)
        assert reasoning == "let me think"
        assert "Hmm, " in content
        assert "OK got it." in content


class TestDetokenizerHoldback:
    """TokenIDScanner must not lose text from detokenizer hold-back.

    The detokenizer may flush previously held-back text in delta_text
    that has no token ID in delta_token_ids.  The scanner must emit
    this text as a TextChunk.
    """

    def test_holdback_text_before_special_token(self):
        """Held-back text preceding a special token must be preserved."""
        token_map = {_END_ID: "</think>"}
        tokenizer = _make_tokenizer_with_text_map("<think>", "</think>", token_map)
        scanner = TokenIDScanner(
            {_END_ID: "THINK_END"},
            tokenizer,
        )
        # delta_text has "'ll" from hold-back + "</think>" from new token
        # delta_token_ids has only the end token
        items = scanner.scan("'ll</think>", [_END_ID])
        assert len(items) == 2
        assert isinstance(items[0], TextChunk)
        assert items[0].text == "'ll"
        assert isinstance(items[1], PreLexedTerminal)
        assert items[1].terminal == "THINK_END"

    def test_holdback_text_before_text_and_special(self):
        """Held-back text + regular text + special token."""
        token_map = {_TEXT_ID: " world", _END_ID: "</think>"}
        tokenizer = _make_tokenizer_with_text_map("<think>", "</think>", token_map)
        scanner = TokenIDScanner(
            {_END_ID: "THINK_END"},
            tokenizer,
        )
        items = scanner.scan("llo world</think>", [_TEXT_ID, _END_ID])
        texts = [item.text for item in items if isinstance(item, TextChunk)]
        assert "llo" in "".join(texts)
        assert " world" in "".join(texts)

    def test_holdback_with_all_drop_tokens(self):
        """When all tokens are dropped, delta_text is entirely hold-back."""
        token_map = {_DROP_ID: "<drop>"}
        tokenizer = _make_tokenizer_with_text_map("<think>", "</think>", token_map)
        scanner = TokenIDScanner(
            {},
            tokenizer,
            drop_token_ids={_DROP_ID},
        )
        items = scanner.scan("remaining text", [_DROP_ID])
        assert len(items) == 1
        assert isinstance(items[0], TextChunk)
        assert items[0].text == "remaining text"

    def test_no_holdback_no_change(self):
        """When delta_text matches reconstructed text, no extra TextChunk."""
        token_map = {_END_ID: "</think>"}
        tokenizer = _make_tokenizer_with_text_map("<think>", "</think>", token_map)
        scanner = TokenIDScanner(
            {_END_ID: "THINK_END"},
            tokenizer,
        )
        items = scanner.scan("</think>", [_END_ID])
        assert len(items) == 1
        assert isinstance(items[0], PreLexedTerminal)

    def test_holdback_streaming_reasoning_preserved(self):
        """End-to-end: hold-back text at reasoning end is emitted as
        reasoning, not lost."""
        start_tag = "<think>"
        end_tag = "</think>"
        token_map = {
            _START_ID: start_tag,
            _END_ID: end_tag,
            _TEXT_ID: "I'",
        }
        tokenizer = _make_tokenizer_with_text_map(start_tag, end_tag, token_map)
        config = think_tag_config(start_tag=start_tag, end_tag=end_tag)
        parser = GrammarReasoningParser(tokenizer, grammar_config=config)

        reasoning_parts: list[str] = []
        content_parts: list[str] = []

        # Step 1: <think> token
        d = parser.extract_reasoning_streaming(
            "", start_tag, start_tag, (), (_START_ID,), (_START_ID,)
        )
        if d and d.reasoning:
            reasoning_parts.append(d.reasoning)

        # Step 2: text token "I'" (detokenizer held back "'ll")
        d = parser.extract_reasoning_streaming(
            start_tag,
            start_tag + "I'",
            "I'",
            (_START_ID,),
            (_START_ID, _TEXT_ID),
            (_TEXT_ID,),
        )
        if d and d.reasoning:
            reasoning_parts.append(d.reasoning)

        # Step 3: </think> token, but delta_text = "ll</think>"
        # because detokenizer flushed the held-back "ll"
        d = parser.extract_reasoning_streaming(
            start_tag + "I'",
            start_tag + "I'll" + end_tag,
            "ll" + end_tag,
            (_START_ID, _TEXT_ID),
            (_START_ID, _TEXT_ID, _END_ID),
            (_END_ID,),
        )
        if d and d.reasoning:
            reasoning_parts.append(d.reasoning)
        if d and d.content:
            content_parts.append(d.content)

        # Step 4: content "done"
        d = parser.extract_reasoning_streaming(
            start_tag + "I'll" + end_tag,
            start_tag + "I'll" + end_tag + "done",
            "done",
            (_START_ID, _TEXT_ID, _END_ID),
            (_START_ID, _TEXT_ID, _END_ID, 200),
            (200,),
        )
        if d and d.content:
            content_parts.append(d.content)

        assert "".join(reasoning_parts) == "I'll"
        assert "".join(content_parts) == "done"


class TestLexerBufferFlush:
    """Lexer buffer must be flushed before PreLexedTerminal transitions."""

    def test_buffered_prefix_emitted_in_current_state(self):
        """Text buffered by the lexer (e.g. '<') must be emitted as
        REASONING_CHUNK before THINK_END transitions to CONTENT."""
        start_tag = "<think>"
        end_tag = "</think>"
        token_map = {
            _START_ID: start_tag,
            _END_ID: end_tag,
        }
        tokenizer = _make_tokenizer_with_text_map(start_tag, end_tag, token_map)
        config = think_tag_config(start_tag=start_tag, end_tag=end_tag)
        engine = StreamingParserEngine(config, tokenizer)

        # Start reasoning
        events = engine.feed(start_tag, [_START_ID])
        assert any(e.type == EventType.REASONING_START for e in events)

        # Feed text ending with '<' — the lexer buffers '<' because
        # it is a prefix of both "<think>" and "</think>".
        events = engine.feed("reasoning text<", [])
        reasoning_text = "".join(
            e.value for e in events if e.type == EventType.REASONING_CHUNK
        )
        assert "reasoning text" in reasoning_text

        # End token arrives via token ID — lexer buffer should flush
        # the '<' as REASONING_CHUNK before the state transition.
        events = engine.feed(end_tag, [_END_ID])
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
        start_tag = "<think>"
        end_tag = "</think>"
        token_map = {_START_ID: start_tag, _END_ID: end_tag}
        tokenizer = _make_tokenizer_with_text_map(start_tag, end_tag, token_map)
        config = think_tag_config(start_tag=start_tag, end_tag=end_tag)
        engine = StreamingParserEngine(config, tokenizer)

        engine.feed(start_tag, [_START_ID])
        engine.feed("clean text", [])

        events = engine.feed(end_tag, [_END_ID])
        assert any(e.type == EventType.REASONING_END for e in events)
        # No stale REASONING_CHUNK from empty buffer
        chunk_events = [e for e in events if e.type == EventType.REASONING_CHUNK]
        assert all(e.value for e in chunk_events)
