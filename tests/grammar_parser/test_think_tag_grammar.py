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
from vllm.grammar_parser.grammars.think_tag import think_tag_config


def _make_tokenizer(start_tag: str, end_tag: str):
    tokenizer = MagicMock()
    tokenizer.encode.return_value = [1, 2, 3]
    vocab = {start_tag: 50, end_tag: 51}
    tokenizer.get_vocab.return_value = vocab
    tokenizer.decode.side_effect = lambda ids: "".join(
        chr(i) if i < 128 else f"<{i}>" for i in ids
    )
    return tokenizer


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
        chunks = ["<think>", "thinking", " hard", "</think>", "done"]
        reasoning_parts = []
        content_parts = []

        prev_text = ""
        prev_ids: list[int] = []

        for chunk in chunks:
            cur_text = prev_text + chunk
            delta_ids = [0]
            cur_ids = prev_ids + delta_ids
            delta = parser.extract_reasoning_streaming(
                previous_text=prev_text,
                current_text=cur_text,
                delta_text=chunk,
                previous_token_ids=tuple(prev_ids),
                current_token_ids=tuple(cur_ids),
                delta_token_ids=tuple(delta_ids),
            )
            if delta:
                if delta.reasoning:
                    reasoning_parts.append(delta.reasoning)
                if delta.content:
                    content_parts.append(delta.content)
            prev_text = cur_text
            prev_ids = list(cur_ids)

        assert "".join(reasoning_parts) == "thinking hard"
        assert "".join(content_parts) == "done"


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
        chunks = [
            "<seed:think>",
            "step 1",
            "</seed:think>",
            "answer",
        ]
        reasoning_parts = []
        content_parts = []
        prev_text = ""
        prev_ids: list[int] = []

        for chunk in chunks:
            cur_text = prev_text + chunk
            delta_ids = [0]
            cur_ids = prev_ids + delta_ids
            delta = parser.extract_reasoning_streaming(
                previous_text=prev_text,
                current_text=cur_text,
                delta_text=chunk,
                previous_token_ids=tuple(prev_ids),
                current_token_ids=tuple(cur_ids),
                delta_token_ids=tuple(delta_ids),
            )
            if delta:
                if delta.reasoning:
                    reasoning_parts.append(delta.reasoning)
                if delta.content:
                    content_parts.append(delta.content)
            prev_text = cur_text
            prev_ids = list(cur_ids)

        assert "".join(reasoning_parts) == "step 1"
        assert "".join(content_parts) == "answer"


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
        chunks = ["[THINK]", "reasoning", "[/THINK]", "content"]
        reasoning_parts = []
        content_parts = []
        prev_text = ""
        prev_ids: list[int] = []

        for chunk in chunks:
            cur_text = prev_text + chunk
            delta_ids = [0]
            cur_ids = prev_ids + delta_ids
            delta = parser.extract_reasoning_streaming(
                previous_text=prev_text,
                current_text=cur_text,
                delta_text=chunk,
                previous_token_ids=tuple(prev_ids),
                current_token_ids=tuple(cur_ids),
                delta_token_ids=tuple(delta_ids),
            )
            if delta:
                if delta.reasoning:
                    reasoning_parts.append(delta.reasoning)
                if delta.content:
                    content_parts.append(delta.content)
            prev_text = cur_text
            prev_ids = list(cur_ids)

        assert "".join(reasoning_parts) == "reasoning"
        assert "".join(content_parts) == "content"


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
