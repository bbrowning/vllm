# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Data-driven replay tests for unified grammar parsers.

Loads token sequences from JSONL files and replays them at different
chunk sizes to verify chunk-size invariance: the same token sequence
must produce identical output regardless of how tokens are batched.
"""

from __future__ import annotations

import dataclasses

import pytest

from tests.grammar_parser.replay_harness import (
    assert_parse_output,
    collect_output,
    load_samples,
    make_mock_tokenizer,
    replay_streaming,
)
from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
)
from vllm.grammar_parser.parsers import (
    Gemma4GrammarParser,
    NemotronV3GrammarParser,
    Qwen3GrammarParser,
)

CHUNK_SIZES = [1, 2, 3, 5, 10, 20, None]

_gemma4_samples = load_samples("gemma4")
_nemotron_v3_samples = load_samples("nemotron_v3")
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


class TestGrammarParserAdjustRequest:
    """Verify GrammarParser and its adapters set skip_special_tokens=False."""

    def test_adjust_request_disables_skip_special_tokens(self):
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

    @pytest.mark.parametrize(
        "adapter_base",
        [
            pytest.param("GrammarReasoningAdapter", id="reasoning"),
            pytest.param("GrammarToolAdapter", id="tool"),
        ],
    )
    def test_adapter_delegates_adjust_request(self, adapter_base):
        """Adapters must delegate adjust_request so that
        skip_special_tokens=False reaches the detokenizer even when the
        serving code only calls the adapter (not the unified parser)."""
        import vllm.grammar_parser.adapters as adapters_mod

        base_cls = getattr(adapters_mod, adapter_base)
        sample = _nemotron_v3_samples[0]
        tokenizer = make_mock_tokenizer(sample)

        adapter_cls = type(
            f"Test{adapter_base}",
            (base_cls,),
            {"_grammar_cls": NemotronV3GrammarParser},
        )
        adapter = adapter_cls(tokenizer)
        request = ChatCompletionRequest(
            model="test-model",
            messages=[{"role": "user", "content": "test"}],
        )
        assert request.skip_special_tokens is True
        adjusted = adapter.adjust_request(request)
        assert adjusted.skip_special_tokens is False


_NEMOTRON_V3_TERMINALS = [
    "<think>",
    "</think>",
    "<tool_call>",
    "</tool_call>",
]

NEMOTRON_CHUNK_SIZES = [1, 2, 3, 5, 10, 19, 20, None]


@pytest.mark.parametrize("chunk_size", NEMOTRON_CHUNK_SIZES, ids=lambda c: f"chunk{c}")
@pytest.mark.parametrize("sample", _nemotron_v3_samples, ids=lambda s: s.id)
class TestNemotronV3Replay:
    """Replay Nemotron V3 token sequences at different chunk sizes."""

    def test_parse_output(self, sample, chunk_size):
        tokenizer = make_mock_tokenizer(sample)
        parser = NemotronV3GrammarParser(tokenizer)
        deltas = replay_streaming(parser, sample.tokens, chunk_size=chunk_size)
        output = collect_output(deltas)
        assert_parse_output(output, sample)

    def test_no_terminal_leakage(self, sample, chunk_size):
        """Terminal text must never appear in reasoning or content."""
        tokenizer = make_mock_tokenizer(sample)
        parser = NemotronV3GrammarParser(tokenizer)
        deltas = replay_streaming(parser, sample.tokens, chunk_size=chunk_size)
        output = collect_output(deltas)

        for terminal in _NEMOTRON_V3_TERMINALS:
            assert terminal not in output.reasoning, (
                f"{terminal!r} leaked into reasoning"
            )
            assert terminal not in output.content, f"{terminal!r} leaked into content"


class TestNemotronV3StreamInterval19:
    """Replay captured sample at stream_interval=19 with aligned text/tokens.

    These tests verify the parser works when ``skip_special_tokens=False``
    is properly set (text and token IDs are aligned).
    """

    _target_id = "nemotron-v3-live-capture-git-diff-003"

    def _get_sample(self):
        for s in _nemotron_v3_samples:
            if s.id == self._target_id:
                return s
        pytest.skip(f"sample {self._target_id!r} not found")

    @pytest.mark.parametrize(
        "chunk_size", [19, 20, 10, 5, 1], ids=lambda c: f"chunk{c}"
    )
    @pytest.mark.parametrize("finished", [False, True], ids=["no_finish", "finish"])
    def test_tool_call_args_parsed(self, chunk_size, finished):
        """Tool call must have non-empty command argument."""
        sample = self._get_sample()
        tokenizer = make_mock_tokenizer(sample)
        parser = NemotronV3GrammarParser(tokenizer)
        deltas = replay_streaming(
            parser,
            sample.tokens,
            chunk_size=chunk_size,
            finished_on_last=finished,
        )
        output = collect_output(deltas)
        assert_parse_output(output, sample)


class TestNemotronV3DeferralFinish:
    """Test that parse_delta(finished=True) resolves deferred scanner state.

    Simulates the production failure where delta_text is missing the
    </tool_call> text but delta_token_ids has the token, causing the
    scanner to defer it. Without finish(), the deferred state is lost
    and tool call arguments are empty.
    """

    @pytest.mark.parametrize("sample", _nemotron_v3_samples, ids=lambda s: s.id)
    def test_misaligned_last_delta_with_finish(self, sample):
        """Tool args must be parsed even when last delta has text/token mismatch."""
        if not sample.expected_tool_calls:
            pytest.skip("no tool calls in sample")

        tokenizer = make_mock_tokenizer(sample)
        parser = NemotronV3GrammarParser(tokenizer)

        request = ChatCompletionRequest(
            model="test-model",
            messages=[{"role": "user", "content": "test"}],
        )

        all_ids = [tid for tid, _ in sample.tokens]
        all_texts = [text for _, text in sample.tokens]

        tool_end_id = sample.vocab.get("</tool_call>")
        split_idx = None
        for i in range(len(all_ids) - 1, -1, -1):
            if all_ids[i] == tool_end_id:
                split_idx = i
                break

        if split_idx is None:
            pytest.skip("no </tool_call> token found")

        first_ids = all_ids[:split_idx]
        first_text = "".join(all_texts[:split_idx])

        last_ids = all_ids[split_idx:]
        last_text_missing = "".join(all_texts[split_idx:]).replace("</tool_call>", "")

        result1 = parser.parse_delta(
            first_text, first_ids, request, prompt_token_ids=[]
        )
        result2 = parser.parse_delta(
            last_text_missing, last_ids, request, finished=True
        )

        output = collect_output([result1, result2])

        tool_calls_only = dataclasses.replace(
            sample, expected_reasoning=None, expected_content=None
        )
        assert_parse_output(output, tool_calls_only)
