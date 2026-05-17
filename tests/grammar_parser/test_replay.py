# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Data-driven replay tests for unified grammar parsers.

Loads token sequences from JSONL files and replays them at different
chunk sizes to verify chunk-size invariance: the same token sequence
must produce identical output regardless of how tokens are batched.
"""

from __future__ import annotations

import pytest

from tests.grammar_parser.replay_harness import (
    assert_parse_output,
    collect_output,
    load_samples,
    make_mock_tokenizer,
    replay_streaming,
)
from vllm.grammar_parser.unified_parsers import (
    Gemma4GrammarParser,
    Qwen3GrammarParser,
)

CHUNK_SIZES = [1, 2, 3, 5, 10, 20, None]

_gemma4_samples = load_samples("gemma4")
_qwen3_samples = load_samples("qwen3")


@pytest.mark.parametrize("chunk_size", CHUNK_SIZES, ids=lambda c: f"chunk{c}")
@pytest.mark.parametrize("sample", _gemma4_samples, ids=lambda s: s.id)
class TestGemma4Replay:
    """Replay Gemma4 token sequences at different chunk sizes."""

    def test_parse_output(self, sample, chunk_size):
        tokenizer = make_mock_tokenizer(sample)
        parser = Gemma4GrammarParser(tokenizer)
        deltas = replay_streaming(parser, sample.tokens, chunk_size=chunk_size)
        output = collect_output(deltas)
        assert_parse_output(output, sample)

    def test_no_terminal_leakage(self, sample, chunk_size):
        """Terminal text must never appear in reasoning or content."""
        tokenizer = make_mock_tokenizer(sample)
        parser = Gemma4GrammarParser(tokenizer)
        deltas = replay_streaming(parser, sample.tokens, chunk_size=chunk_size)
        output = collect_output(deltas)

        for terminal in ["<|channel>", "<channel|>", "<|tool_call>", "<tool_call|>"]:
            assert terminal not in output.reasoning, (
                f"{terminal!r} leaked into reasoning"
            )
            assert terminal not in output.content, f"{terminal!r} leaked into content"


HOLDBACK_CONFIGS = [6, 12, 24]


@pytest.mark.parametrize("holdback", HOLDBACK_CONFIGS, ids=lambda h: f"holdback{h}")
@pytest.mark.parametrize("chunk_size", [5, 10], ids=lambda c: f"chunk{c}")
@pytest.mark.parametrize("sample", _gemma4_samples, ids=lambda s: s.id)
class TestGemma4ReplayWithHoldback:
    """Replay with simulated detokenizer holdback."""

    def test_parse_output_with_holdback(self, sample, chunk_size, holdback):
        tokenizer = make_mock_tokenizer(sample)
        parser = Gemma4GrammarParser(tokenizer)
        deltas = replay_streaming(
            parser,
            sample.tokens,
            chunk_size=chunk_size,
            holdback_chars=holdback,
        )
        output = collect_output(deltas)
        assert_parse_output(output, sample)


@pytest.mark.parametrize("chunk_size", CHUNK_SIZES, ids=lambda c: f"chunk{c}")
@pytest.mark.parametrize("sample", _qwen3_samples, ids=lambda s: s.id)
class TestQwen3Replay:
    """Replay Qwen3 token sequences at different chunk sizes."""

    def test_parse_output(self, sample, chunk_size):
        tokenizer = make_mock_tokenizer(sample)
        parser = Qwen3GrammarParser(tokenizer)
        deltas = replay_streaming(parser, sample.tokens, chunk_size=chunk_size)
        output = collect_output(deltas)
        assert_parse_output(output, sample)

    def test_no_terminal_leakage(self, sample, chunk_size):
        """Terminal text must never appear in reasoning or content."""
        tokenizer = make_mock_tokenizer(sample)
        parser = Qwen3GrammarParser(tokenizer)
        deltas = replay_streaming(parser, sample.tokens, chunk_size=chunk_size)
        output = collect_output(deltas)

        for terminal in [
            "<think>",
            "</think>",
            "<tool_call>",
            "</tool_call>",
            "<function=",
            "</function>",
        ]:
            assert terminal not in output.reasoning, (
                f"{terminal!r} leaked into reasoning"
            )
            assert terminal not in output.content, f"{terminal!r} leaked into content"


@pytest.mark.parametrize("holdback", HOLDBACK_CONFIGS, ids=lambda h: f"holdback{h}")
@pytest.mark.parametrize("chunk_size", [5, 10], ids=lambda c: f"chunk{c}")
@pytest.mark.parametrize("sample", _qwen3_samples, ids=lambda s: s.id)
class TestQwen3ReplayWithHoldback:
    """Replay Qwen3 with simulated detokenizer holdback."""

    def test_parse_output_with_holdback(self, sample, chunk_size, holdback):
        tokenizer = make_mock_tokenizer(sample)
        parser = Qwen3GrammarParser(tokenizer)
        deltas = replay_streaming(
            parser,
            sample.tokens,
            chunk_size=chunk_size,
            holdback_chars=holdback,
        )
        output = collect_output(deltas)
        assert_parse_output(output, sample)
