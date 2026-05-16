# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the grammar-based Qwen3 reasoning parser.

Validates that ``GrammarQwen3ReasoningParser`` correctly handles
``<think>``/``</think>`` reasoning with Qwen3-specific extensions:
- ``<tool_call>`` as implicit reasoning end
- Stripping ``<think>`` from generated output (old template compat)
"""

from unittest.mock import MagicMock

import pytest

from vllm.grammar_parser.registered_parsers import GrammarQwen3ReasoningParser

_THINK_START_ID = 50
_THINK_END_ID = 51
_TOOL_CALL_ID = 60
_TOOL_CALL_END_ID = 61
_TEXT_ID = 100


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
        chr(i) if i < 128 else f"<{i}>" for i in ids
    )
    return tokenizer


@pytest.fixture
def parser():
    return GrammarQwen3ReasoningParser(_make_tokenizer())


def _stream_and_collect(
    parser: GrammarQwen3ReasoningParser,
    chunks: list[str],
    delta_token_ids_per_chunk: list[tuple[int, ...]] | None = None,
) -> tuple[str, str]:
    """Feed chunks through streaming and collect reasoning/content."""
    reasoning_parts: list[str] = []
    content_parts: list[str] = []
    prev_text = ""
    prev_ids: list[int] = []
    for i, chunk in enumerate(chunks):
        cur_text = prev_text + chunk
        if delta_token_ids_per_chunk is not None:
            d_ids = delta_token_ids_per_chunk[i]
        else:
            d_ids = (0,)
        cur_ids = prev_ids + list(d_ids)
        delta = parser.extract_reasoning_streaming(
            previous_text=prev_text,
            current_text=cur_text,
            delta_text=chunk,
            previous_token_ids=tuple(prev_ids),
            current_token_ids=tuple(cur_ids),
            delta_token_ids=d_ids,
        )
        if delta:
            if delta.reasoning:
                reasoning_parts.append(delta.reasoning)
            if delta.content:
                content_parts.append(delta.content)
        prev_text = cur_text
        prev_ids = list(cur_ids)
    return "".join(reasoning_parts), "".join(content_parts)


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
        assert content is not None
        assert "<tool_call>" in content

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
        assert content is not None
        assert "<tool_call>" in content


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
        reasoning, content = _stream_and_collect(
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
        reasoning, content = _stream_and_collect(
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
        reasoning, content = _stream_and_collect(
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
        reasoning, content = _stream_and_collect(
            parser,
            ["I need to check.", "<tool_call>", "\n<function=test>"],
            [
                (1,),
                (_TOOL_CALL_ID,),
                (2,),
            ],
        )
        assert reasoning == "I need to check."
        assert "<tool_call>" in content

    def test_streaming_content_after_think_end(self, parser):
        """Content deltas after </think> are routed as content."""
        reasoning, content = _stream_and_collect(
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
        reasoning, content = _stream_and_collect(
            parser,
            ["thinking", "<tool_call>", "<function=f>"],
            [
                (1,),
                (_TOOL_CALL_ID,),
                (2,),
            ],
        )
        assert reasoning == "thinking"
        assert "<tool_call>" in content
        assert "<function=f>" in content

    def test_streaming_end_grouped_with_content(self, parser):
        """</think> grouped with following content in one delta."""
        reasoning, content = _stream_and_collect(
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
        reasoning, content = _stream_and_collect(
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
        reasoning, content = _stream_and_collect(
            parser,
            ["hello ", "world"],
            [
                (1,),
                (2,),
            ],
        )
        assert reasoning == "hello world"
        assert content == ""
