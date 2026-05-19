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
from vllm.grammar_parser.parsers import (
    Gemma4GrammarParser,
    Qwen3GrammarParser,
)

CHUNK_SIZES = [1, 2, 3, 5, 10, 20, None]

_gemma4_samples = load_samples("gemma4")
_qwen3_samples = load_samples("qwen3")

_GEMMA4_TERMINALS = ["<|channel>", "<channel|>", "<|tool_call>", "<tool_call|>"]


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

        for terminal in _GEMMA4_TERMINALS:
            assert terminal not in output.reasoning, (
                f"{terminal!r} leaked into reasoning"
            )
            assert terminal not in output.content, f"{terminal!r} leaked into content"

    def test_no_thought_prefix_leakage(self, sample, chunk_size):
        """The ``thought\\n`` prefix must never appear in content."""
        tokenizer = make_mock_tokenizer(sample)
        parser = Gemma4GrammarParser(tokenizer)
        deltas = replay_streaming(parser, sample.tokens, chunk_size=chunk_size)
        output = collect_output(deltas)

        assert "thought\n" not in output.content, (
            "'thought\\n' prefix leaked into content"
        )
        assert not output.reasoning.startswith("thought\n"), (
            "reasoning starts with unstripped 'thought\\n' prefix"
        )


HOLDBACK_CONFIGS = [6, 12, 24]


@pytest.mark.parametrize("holdback", HOLDBACK_CONFIGS, ids=lambda h: f"holdback{h}")
@pytest.mark.parametrize("chunk_size", [3, 5, 10], ids=lambda c: f"chunk{c}")
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

    def test_no_terminal_leakage_with_holdback(self, sample, chunk_size, holdback):
        """Terminal text must never appear in content or reasoning under holdback."""
        tokenizer = make_mock_tokenizer(sample)
        parser = Gemma4GrammarParser(tokenizer)
        deltas = replay_streaming(
            parser,
            sample.tokens,
            chunk_size=chunk_size,
            holdback_chars=holdback,
        )
        output = collect_output(deltas)
        for terminal in _GEMMA4_TERMINALS:
            assert terminal not in output.content, (
                f"{terminal!r} leaked into content "
                f"(chunk_size={chunk_size}, holdback={holdback})"
            )
            assert terminal not in output.reasoning, (
                f"{terminal!r} leaked into reasoning "
                f"(chunk_size={chunk_size}, holdback={holdback})"
            )


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


class TestGemma4AdjustRequest:
    """Verify Gemma4GrammarParser.adjust_request sets skip_special_tokens."""

    def test_adjust_request_disables_skip_special_tokens(self):
        from vllm.entrypoints.openai.chat_completion.protocol import (
            ChatCompletionRequest,
        )

        sample = _gemma4_samples[0]
        tokenizer = make_mock_tokenizer(sample)
        parser = Gemma4GrammarParser(tokenizer)
        request = ChatCompletionRequest(
            model="test-model",
            messages=[{"role": "user", "content": "test"}],
        )
        assert request.skip_special_tokens is True
        adjusted = parser.adjust_request(request)
        assert adjusted.skip_special_tokens is False

    def test_thought_prefix_leaks_without_adjust_request(self):
        """Without adjust_request (skip_special_tokens=True), thought\\n
        leaks into content. This documents the failure mode that the
        server-side fix (calling adjust_request) prevents.

        Uses chunk_size=3 to match the original stream-interval=3 capture.
        """
        samples = [s for s in _gemma4_samples if "thought-prefix-leak" in s.id]
        if not samples:
            pytest.skip("sample 006 not found")
        sample = samples[0]
        special_ids = set(sample.vocab.values())
        tokenizer = make_mock_tokenizer(sample)
        parser = Gemma4GrammarParser(tokenizer)
        deltas = replay_streaming(
            parser,
            sample.tokens,
            chunk_size=3,
            special_token_ids=special_ids,
        )
        output = collect_output(deltas)
        assert "thought" in output.content, (
            "Expected 'thought' to leak into content without adjust_request, "
            "but parser handled it correctly — update this test"
        )
