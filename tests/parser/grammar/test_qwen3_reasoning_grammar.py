# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the grammar-based Qwen3 reasoning parser.

Validates that ``GrammarQwen3ReasoningParser`` correctly handles
``<think>``/``</think>`` reasoning with Qwen3-specific extensions:
- ``<tool_call>`` as implicit reasoning end (grammar terminal + token ID)
- Stripping ``<think>`` from generated output (old template compat)
- No terminal text (``</think>``, ``<tool_call>``) leaks into output
"""

from unittest.mock import MagicMock

import pytest

from tests.parser.grammar.streaming_helpers import simulate_reasoning_streaming
from vllm.parser.grammar.parsers.qwen3 import Qwen3GrammarParser

_THINK_START_ID = 50
_THINK_END_ID = 51
_TOOL_CALL_ID = 60
_TOOL_CALL_END_ID = 61
_TEXT_ID = 100


_SPECIAL_TOKEN_TEXT = {
    _THINK_START_ID: "<think>",
    _THINK_END_ID: "</think>",
    _TOOL_CALL_ID: "<tool_call>",
    _TOOL_CALL_END_ID: "</tool_call>",
}


def _make_tokenizer():
    tokenizer = MagicMock()
    tokenizer.encode.return_value = [1, 2, 3]
    tokenizer.get_vocab.return_value = {
        "<think>": _THINK_START_ID,
        "</think>": _THINK_END_ID,
        "<tool_call>": _TOOL_CALL_ID,
        "</tool_call>": _TOOL_CALL_END_ID,
    }
    tokenizer.decode.side_effect = lambda ids: "".join(
        _SPECIAL_TOKEN_TEXT.get(i, chr(i) if i < 128 else f"<{i}>") for i in ids
    )
    return tokenizer


@pytest.fixture
def parser():
    return Qwen3GrammarParser(_make_tokenizer())


class TestNonStreaming:
    def test_reasoning_then_content(self, parser):
        text = "<think>Let me analyze.</think>The answer is 42."
        reasoning, content = parser.extract_reasoning(text, None)
        assert reasoning == "Let me analyze."
        assert content == "The answer is 42."

    def test_no_start_token_in_output(self, parser):
        """Qwen3.5+ style: <think> in prompt, only </think> in output."""
        text = "Let me think about this.</think>The answer is 42."
        reasoning, content = parser.extract_reasoning(text, None)
        assert reasoning == "Let me think about this."
        assert content == "The answer is 42."

    def test_reasoning_only(self, parser):
        text = "<think>Still thinking...</think>"
        reasoning, content = parser.extract_reasoning(text, None)
        assert reasoning == "Still thinking..."
        assert content is None

    def test_no_end_tag_all_reasoning(self, parser):
        """No </think> means truncated output — everything is reasoning."""
        text = "Hello, no reasoning here."
        reasoning, content = parser.extract_reasoning(text, None)
        assert reasoning == "Hello, no reasoning here."
        assert content is None

    def test_multiline_reasoning(self, parser):
        text = (
            "<think>Step 1: parse.\nStep 2: compute.\nStep 3: output.</think>Result: 7."
        )
        reasoning, content = parser.extract_reasoning(text, None)
        assert "Step 1" in reasoning
        assert "Step 3" in reasoning
        assert content == "Result: 7."

    def test_tool_call_implicit_end(self, parser):
        """<tool_call> without </think> acts as implicit reasoning end."""
        text = (
            "<think>I need to read the file.\n\n"
            "<tool_call>\n<function=bash>\n"
            "<parameter=cmd>ls</parameter>\n"
            "</function>\n</tool_call>"
        )
        reasoning, content = parser.extract_reasoning(text, None)
        assert reasoning == "I need to read the file.\n\n"
        assert "</think>" not in reasoning
        assert "<tool_call>" not in reasoning

    def test_tool_call_implicit_end_no_think(self, parser):
        """<tool_call> as implicit end, no <think> in output."""
        text = (
            "I need to read the file.\n\n"
            "<tool_call>\n<function=bash>\n"
            "<parameter=cmd>ls</parameter>\n"
            "</function>\n</tool_call>"
        )
        reasoning, content = parser.extract_reasoning(text, None)
        assert reasoning == "I need to read the file.\n\n"
        assert "<tool_call>" not in reasoning

    def test_live_scenario_think_end_before_tool_call(self, parser):
        """Real model output: </think> immediately before <tool_call>.

        Regression test for the bug where </think> and <parameter=...>
        leaked into reasoning content.
        """
        text = (
            "The user wants to see what files are in the current directory"
            " and their contents. Let me start by listing the directory."
            "</think><tool_call><function=read>"
            "<parameter=filePath>/Users/test/demo</parameter>"
            "</function></tool_call>"
        )
        reasoning, content = parser.extract_reasoning(text, None)
        expected_reasoning = (
            "The user wants to see what files are in the current directory"
            " and their contents. Let me start by listing the directory."
        )
        assert reasoning == expected_reasoning
        assert "</think>" not in reasoning
        assert "<tool_call>" not in reasoning
        assert "<parameter=" not in reasoning

    def test_no_terminal_text_in_reasoning(self, parser):
        """Terminal text must never appear in reasoning output."""
        text = "Reasoning here.</think>Content here."
        reasoning, content = parser.extract_reasoning(text, None)
        assert "</think>" not in (reasoning or "")
        assert "<think>" not in (reasoning or "")

    def test_no_terminal_text_in_content(self, parser):
        """Terminal text must never appear in content output."""
        text = "Reasoning here.</think>Content here."
        reasoning, content = parser.extract_reasoning(text, None)
        assert "</think>" not in (content or "")
        assert "<think>" not in (content or "")


class TestIsReasoningEnd:
    def test_think_end_token(self, parser):
        assert parser.is_reasoning_end([_THINK_START_ID, 1, _THINK_END_ID])

    def test_no_end_token(self, parser):
        assert not parser.is_reasoning_end([_THINK_START_ID, 1, 2])

    def test_start_after_end_means_not_ended(self, parser):
        assert not parser.is_reasoning_end([_THINK_END_ID, _THINK_START_ID, 1])

    def test_tool_call_as_implicit_end(self, parser):
        """Unpaired <tool_call> is implicit reasoning end."""
        assert parser.is_reasoning_end([_THINK_START_ID, 1, _TOOL_CALL_ID])

    def test_paired_tool_call_not_end(self, parser):
        """Paired <tool_call>...</tool_call> (from template) is NOT end."""
        assert not parser.is_reasoning_end(
            [_THINK_START_ID, 1, _TOOL_CALL_ID, 2, _TOOL_CALL_END_ID]
        )

    def test_tool_call_after_think_end(self, parser):
        """<tool_call> after </think> — already ended."""
        assert parser.is_reasoning_end(
            [_THINK_START_ID, 1, _THINK_END_ID, _TOOL_CALL_ID]
        )

    def test_empty_ids(self, parser):
        assert not parser.is_reasoning_end([])


class TestStreaming:
    def test_basic_streaming(self, parser):
        reasoning, content = simulate_reasoning_streaming(
            parser,
            ["<think>", "thinking", " hard", "</think>", "done"],
            [
                (_THINK_START_ID,),
                (1,),
                (2,),
                (_THINK_END_ID,),
                (3,),
            ],
        )
        assert reasoning == "thinking hard"
        assert content == "done"

    def test_streaming_no_start_token(self, parser):
        """Qwen3.5 style: no <think> in output, just reasoning then </think>."""
        reasoning, content = simulate_reasoning_streaming(
            parser,
            ["reasoning ", "text", "</think>", "content"],
            [
                (1,),
                (2,),
                (_THINK_END_ID,),
                (3,),
            ],
        )
        assert reasoning == "reasoning text"
        assert content == "content"

    def test_streaming_start_token_stripped(self, parser):
        """<think> in output (old template) should be stripped."""
        reasoning, content = simulate_reasoning_streaming(
            parser,
            ["<think>reasoning", "</think>", "content"],
            [
                (_THINK_START_ID, 1),
                (_THINK_END_ID,),
                (2,),
            ],
        )
        assert reasoning == "reasoning"
        assert content == "content"

    def test_streaming_tool_call_implicit_end(self, parser):
        """<tool_call> ends reasoning implicitly during streaming."""
        reasoning, content = simulate_reasoning_streaming(
            parser,
            ["I need to check.", "<tool_call>", "\n<function=test>"],
            [
                (1,),
                (_TOOL_CALL_ID,),
                (2,),
            ],
        )
        assert reasoning == "I need to check."
        assert "<tool_call>" not in reasoning
        assert "</think>" not in reasoning
        assert content is not None

    def test_streaming_content_after_think_end(self, parser):
        """Content deltas after </think> are routed as content."""
        reasoning, content = simulate_reasoning_streaming(
            parser,
            ["reasoning", "</think>", "content1", " content2"],
            [
                (1,),
                (_THINK_END_ID,),
                (2,),
                (3,),
            ],
        )
        assert reasoning == "reasoning"
        assert content == "content1 content2"

    def test_streaming_content_after_tool_call(self, parser):
        """Content deltas after <tool_call> are routed as content."""
        reasoning, content = simulate_reasoning_streaming(
            parser,
            ["thinking", "<tool_call>", "<function=f>"],
            [
                (1,),
                (_TOOL_CALL_ID,),
                (2,),
            ],
        )
        assert reasoning == "thinking"
        assert "<tool_call>" not in reasoning
        assert content is not None

    def test_streaming_end_grouped_with_content(self, parser):
        """</think> grouped with following content in one delta."""
        reasoning, content = simulate_reasoning_streaming(
            parser,
            ["reasoning", "</think>the answer"],
            [
                (1,),
                (_THINK_END_ID, 2),
            ],
        )
        assert reasoning == "reasoning"
        assert content == "the answer"

    def test_streaming_think_and_end_in_one_delta(self, parser):
        """<think> and </think> in the same delta."""
        reasoning, content = simulate_reasoning_streaming(
            parser,
            ["<think>reasoning</think>"],
            [
                (_THINK_START_ID, 1, _THINK_END_ID),
            ],
        )
        assert reasoning == "reasoning"
        assert content == ""

    def test_streaming_pure_content_no_think(self, parser):
        """No think tokens at all — everything is reasoning (truncated)."""
        reasoning, content = simulate_reasoning_streaming(
            parser,
            ["hello ", "world"],
            [
                (1,),
                (2,),
            ],
        )
        assert reasoning == "hello world"
        assert content == ""

    def test_streaming_think_end_and_tool_call_same_delta(self, parser):
        """</think> and <tool_call> in the same delta — no leakage.

        Regression test: the old override split at <tool_call> without
        stripping </think>, causing </think> to leak into reasoning.
        """
        reasoning, content = simulate_reasoning_streaming(
            parser,
            [
                "Let me list the directory.",
                "</think><tool_call>",
                "<function=read>",
                "<parameter=filePath>/tmp</parameter>",
            ],
            [
                (1,),
                (_THINK_END_ID, _TOOL_CALL_ID),
                (2,),
                (3,),
            ],
        )
        assert reasoning == "Let me list the directory."
        assert "</think>" not in reasoning
        assert "<tool_call>" not in reasoning
        assert "<parameter=" not in reasoning
        assert content is not None

    def test_streaming_no_terminal_text_leaks(self, parser):
        """Terminal text must never appear in reasoning or content."""
        reasoning, content = simulate_reasoning_streaming(
            parser,
            ["reasoning", "</think>", "content"],
            [
                (1,),
                (_THINK_END_ID,),
                (2,),
            ],
        )
        assert "</think>" not in reasoning
        assert "</think>" not in content
        assert "<think>" not in reasoning
