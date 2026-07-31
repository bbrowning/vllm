# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for ``is_reasoning_end_streaming`` across all parser-engine-based
reasoning parsers.

Validates that every parser's streaming reasoning-end detection:
- Returns True when delta contains a trigger token
- Returns False when delta contains only non-trigger tokens
- Agrees with ``is_reasoning_end`` at the transition point (parity)
- Handles edge cases: empty delta, multi-token delta (spec decode),
  thinking-disabled mode
"""

import pytest

from tests.parser.engine.conftest import make_mock_tokenizer
from vllm.parser.deepseek_v4 import DeepSeekV4Parser
from vllm.parser.deepseek_v32 import DeepSeekV32Parser
from vllm.parser.engine.registered_adapters import (
    DeepSeekV32ParserReasoningAdapter,
    Gemma4ParserReasoningAdapter,
    InklingParserReasoningAdapter,
    KimiK2ParserReasoningAdapter,
    Qwen3ParserReasoningAdapter,
)
from vllm.parser.gemma4 import Gemma4Parser
from vllm.parser.glm47_moe import Glm47MoeParser
from vllm.parser.inkling import InklingParser
from vllm.parser.kimi_k2 import KimiK2Parser
from vllm.parser.minimax_m2 import MinimaxM2Parser
from vllm.parser.nemotron_v3 import NemotronV3Parser
from vllm.parser.qwen3 import Qwen3Parser
from vllm.parser.seed_oss import SeedOssParser

# Shared non-trigger token IDs
_TEXT_A = 100
_TEXT_B = 101
_TEXT_C = 102


_DELTA_SIZES = [1, 5, 50, 200]


def _assert_sequence_replay(parser, start_token, end_token, fill_count=20):
    """Step through a think->reasoning->end sequence, verify detection."""
    sequence = [start_token] + [_TEXT_A] * fill_count + [end_token]
    for i in range(1, len(sequence)):
        all_tokens = sequence[: i + 1]
        delta = [sequence[i]]
        result = parser.is_reasoning_end_streaming(all_tokens, delta)
        if sequence[i] == end_token:
            assert result, f"Should detect end at position {i}"
        else:
            assert not result, f"False positive at position {i}"


def _assert_parity(parser, start_token, end_token, fill_count=10):
    """Streaming and full-scan must agree at every position."""
    sequence = [start_token] + [_TEXT_A] * fill_count + [end_token]
    for i in range(1, len(sequence)):
        all_tokens = sequence[: i + 1]
        delta = [sequence[i]]
        streaming = parser.is_reasoning_end_streaming(all_tokens, delta)
        full_scan = parser.is_reasoning_end(all_tokens)
        assert streaming == full_scan, (
            f"Parity mismatch at pos {i}: streaming={streaming}, full_scan={full_scan}"
        )


def _assert_variable_delta_trigger(parser, start_token, end_token):
    """Trigger detected regardless of delta size (spec decode batches)."""
    for n in _DELTA_SIZES:
        filler = [_TEXT_A, _TEXT_B, _TEXT_C] * ((n - 1) // 3 + 1)
        delta = filler[: n - 1] + [end_token]
        all_tokens = [start_token] + [_TEXT_A] * 10 + delta
        assert parser.is_reasoning_end_streaming(all_tokens, delta), (
            f"Missed trigger with delta_size={len(delta)}"
        )


def _assert_variable_delta_no_trigger(parser, start_token):
    """No false positive regardless of delta size."""
    for n in _DELTA_SIZES:
        filler = [_TEXT_A, _TEXT_B, _TEXT_C] * ((n) // 3 + 1)
        delta = filler[:n]
        all_tokens = [start_token] + [_TEXT_A] * 10 + delta
        assert not parser.is_reasoning_end_streaming(all_tokens, delta), (
            f"False positive with delta_size={len(delta)}"
        )


def _assert_reentry_returns_false(parser, start_token, end_token):
    """Multi-token delta with end then start must return False (re-entry)."""
    delta = [end_token, start_token, _TEXT_A]
    all_tokens = [start_token] + [_TEXT_A] * 5 + delta
    streaming = parser.is_reasoning_end_streaming(all_tokens, delta)
    full_scan = parser.is_reasoning_end(all_tokens)
    assert not streaming, "Should not report end when reasoning re-opens in delta"
    assert streaming == full_scan, (
        f"Re-entry parity: streaming={streaming}, full_scan={full_scan}"
    )


def _assert_trigger_reentry_returns_false(parser, start_token, trigger_token):
    """A trigger followed by start token in the same delta must return False."""
    delta = [trigger_token, start_token, _TEXT_A]
    all_tokens = [start_token] + [_TEXT_A] * 5 + delta
    assert not parser.is_reasoning_end_streaming(all_tokens, delta)


def _assert_parity_multi_token(parser, start_token, end_token):
    """Streaming and full-scan must agree for multi-token deltas."""
    delta = [_TEXT_A, end_token]
    all_tokens = [start_token] + [_TEXT_A] * 5 + delta
    streaming = parser.is_reasoning_end_streaming(all_tokens, delta)
    full_scan = parser.is_reasoning_end(all_tokens)
    assert streaming == full_scan, (
        f"Multi-token parity: streaming={streaming}, full_scan={full_scan}"
    )

    _assert_reentry_returns_false(parser, start_token, end_token)


# ── Qwen3 ────────────────────────────────────────────────────────────

_QWEN3_THINK_START = 50
_QWEN3_THINK_END = 51
_QWEN3_TOOL_CALL = 60
_QWEN3_TOOL_CALL_END = 61

_QWEN3_VOCAB = {
    "<think>": _QWEN3_THINK_START,
    "</think>": _QWEN3_THINK_END,
    "<tool_call>": _QWEN3_TOOL_CALL,
    "</tool_call>": _QWEN3_TOOL_CALL_END,
}


@pytest.fixture
def qwen3_parser():
    return Qwen3Parser(make_mock_tokenizer(_QWEN3_VOCAB))


class TestQwen3Streaming:
    def test_think_end_returns_true(self, qwen3_parser):
        ids = [_QWEN3_THINK_START, _TEXT_A, _QWEN3_THINK_END]
        assert qwen3_parser.is_reasoning_end_streaming(ids, [_QWEN3_THINK_END])

    def test_tool_call_in_delta_triggers_end(self, qwen3_parser):
        ids = [_QWEN3_THINK_START, _TEXT_A, _QWEN3_TOOL_CALL]
        assert qwen3_parser.is_reasoning_end_streaming(ids, [_QWEN3_TOOL_CALL])

    def test_non_trigger_returns_false(self, qwen3_parser):
        ids = [_QWEN3_THINK_START, _TEXT_A]
        assert not qwen3_parser.is_reasoning_end_streaming(ids, [_TEXT_A])

    def test_empty_delta_returns_false(self, qwen3_parser):
        ids = [_QWEN3_THINK_START, _TEXT_A]
        assert not qwen3_parser.is_reasoning_end_streaming(ids, [])

    def test_variable_delta_trigger(self, qwen3_parser):
        _assert_variable_delta_trigger(
            qwen3_parser, _QWEN3_THINK_START, _QWEN3_THINK_END
        )

    def test_variable_delta_tool_call_trigger(self, qwen3_parser):
        for n in _DELTA_SIZES:
            filler = [_TEXT_A, _TEXT_B, _TEXT_C] * ((n - 1) // 3 + 1)
            delta = filler[: n - 1] + [_QWEN3_TOOL_CALL]
            all_tokens = [_QWEN3_THINK_START] + [_TEXT_A] * 10 + delta
            assert qwen3_parser.is_reasoning_end_streaming(all_tokens, delta), (
                f"Missed tool_call trigger with delta_size={len(delta)}"
            )

    def test_variable_delta_no_trigger(self, qwen3_parser):
        _assert_variable_delta_no_trigger(qwen3_parser, _QWEN3_THINK_START)

    def test_realistic_sequence_replay(self, qwen3_parser):
        _assert_sequence_replay(qwen3_parser, _QWEN3_THINK_START, _QWEN3_THINK_END, 50)

    def test_parity_with_is_reasoning_end(self, qwen3_parser):
        _assert_parity(qwen3_parser, _QWEN3_THINK_START, _QWEN3_THINK_END)

    def test_parity_multi_token_delta(self, qwen3_parser):
        _assert_parity_multi_token(qwen3_parser, _QWEN3_THINK_START, _QWEN3_THINK_END)

    def test_tool_call_then_reenter_thinking(self, qwen3_parser):
        _assert_trigger_reentry_returns_false(
            qwen3_parser, _QWEN3_THINK_START, _QWEN3_TOOL_CALL
        )

    def test_closed_tool_call_returns_false(self, qwen3_parser):
        delta = [_QWEN3_TOOL_CALL, _TEXT_A, _QWEN3_TOOL_CALL_END, _TEXT_A]
        all_tokens = [_QWEN3_THINK_START] + [_TEXT_A] * 5 + delta
        streaming = qwen3_parser.is_reasoning_end_streaming(all_tokens, delta)
        full_scan = qwen3_parser.is_reasoning_end(all_tokens)
        assert streaming == full_scan
        assert not streaming


# ── NemotronV3 (extends Qwen3) ──────────────────────────────────────


@pytest.fixture
def nemotron_parser():
    return NemotronV3Parser(make_mock_tokenizer(_QWEN3_VOCAB))


class TestNemotronV3Streaming:
    def test_think_end_returns_true(self, nemotron_parser):
        ids = [_QWEN3_THINK_START, _TEXT_A, _QWEN3_THINK_END]
        assert nemotron_parser.is_reasoning_end_streaming(ids, [_QWEN3_THINK_END])

    def test_tool_call_in_delta_triggers_end(self, nemotron_parser):
        ids = [_QWEN3_THINK_START, _TEXT_A, _QWEN3_TOOL_CALL]
        assert nemotron_parser.is_reasoning_end_streaming(ids, [_QWEN3_TOOL_CALL])

    def test_non_trigger_returns_false(self, nemotron_parser):
        ids = [_QWEN3_THINK_START, _TEXT_A]
        assert not nemotron_parser.is_reasoning_end_streaming(ids, [_TEXT_A])

    def test_empty_delta_returns_false(self, nemotron_parser):
        ids = [_QWEN3_THINK_START, _TEXT_A]
        assert not nemotron_parser.is_reasoning_end_streaming(ids, [])

    def test_variable_delta_trigger(self, nemotron_parser):
        _assert_variable_delta_trigger(
            nemotron_parser, _QWEN3_THINK_START, _QWEN3_THINK_END
        )

    def test_variable_delta_no_trigger(self, nemotron_parser):
        _assert_variable_delta_no_trigger(nemotron_parser, _QWEN3_THINK_START)


# ── SeedOss (extends Qwen3) ─────────────────────────────────────────

_SEED_THINK_START = 70
_SEED_THINK_END = 71
_SEED_TOOL_CALL = 72
_SEED_TOOL_CALL_END = 73

_SEED_VOCAB = {
    "<seed:think>": _SEED_THINK_START,
    "</seed:think>": _SEED_THINK_END,
    "<seed:tool_call>": _SEED_TOOL_CALL,
    "</seed:tool_call>": _SEED_TOOL_CALL_END,
}


@pytest.fixture
def seed_parser():
    return SeedOssParser(make_mock_tokenizer(_SEED_VOCAB))


class TestSeedOssStreaming:
    def test_think_end_returns_true(self, seed_parser):
        ids = [_SEED_THINK_START, _TEXT_A, _SEED_THINK_END]
        assert seed_parser.is_reasoning_end_streaming(ids, [_SEED_THINK_END])

    def test_tool_call_in_delta_triggers_end(self, seed_parser):
        ids = [_SEED_THINK_START, _TEXT_A, _SEED_TOOL_CALL]
        assert seed_parser.is_reasoning_end_streaming(ids, [_SEED_TOOL_CALL])

    def test_non_trigger_returns_false(self, seed_parser):
        ids = [_SEED_THINK_START, _TEXT_A]
        assert not seed_parser.is_reasoning_end_streaming(ids, [_TEXT_A])

    def test_empty_delta_returns_false(self, seed_parser):
        ids = [_SEED_THINK_START, _TEXT_A]
        assert not seed_parser.is_reasoning_end_streaming(ids, [])

    def test_variable_delta_trigger(self, seed_parser):
        _assert_variable_delta_trigger(seed_parser, _SEED_THINK_START, _SEED_THINK_END)

    def test_variable_delta_no_trigger(self, seed_parser):
        _assert_variable_delta_no_trigger(seed_parser, _SEED_THINK_START)

    def test_realistic_sequence_replay(self, seed_parser):
        _assert_sequence_replay(seed_parser, _SEED_THINK_START, _SEED_THINK_END)


# ── Gemma4 ───────────────────────────────────────────────────────────

_GEMMA4_CHANNEL_START = 80
_GEMMA4_CHANNEL_END = 81
_GEMMA4_TOOL_CALL = 82
_GEMMA4_TURN = 83
_GEMMA4_TOOL_RESPONSE = 84

_GEMMA4_VOCAB = {
    "<|channel>": _GEMMA4_CHANNEL_START,
    "<channel|>": _GEMMA4_CHANNEL_END,
    "<|tool_call>": _GEMMA4_TOOL_CALL,
    "<|turn>": _GEMMA4_TURN,
    "<|tool_response>": _GEMMA4_TOOL_RESPONSE,
}


@pytest.fixture
def gemma4_parser():
    return Gemma4Parser(make_mock_tokenizer(_GEMMA4_VOCAB))


@pytest.fixture
def gemma4_parser_thinking_disabled():
    return Gemma4Parser(
        make_mock_tokenizer(_GEMMA4_VOCAB),
        chat_template_kwargs={"enable_thinking": False},
    )


class TestGemma4Streaming:
    def test_channel_end_returns_true(self, gemma4_parser):
        ids = [_GEMMA4_CHANNEL_START, _TEXT_A, _GEMMA4_CHANNEL_END]
        assert gemma4_parser.is_reasoning_end_streaming(ids, [_GEMMA4_CHANNEL_END])

    def test_tool_call_triggers_end(self, gemma4_parser):
        ids = [_GEMMA4_CHANNEL_START, _TEXT_A, _GEMMA4_TOOL_CALL]
        assert gemma4_parser.is_reasoning_end_streaming(ids, [_GEMMA4_TOOL_CALL])

    def test_new_turn_thinking_enabled_no_trigger(self, gemma4_parser):
        ids = [_GEMMA4_CHANNEL_START, _TEXT_A, _GEMMA4_TURN]
        assert not gemma4_parser.is_reasoning_end_streaming(ids, [_GEMMA4_TURN])

    def test_new_turn_thinking_disabled_triggers(self, gemma4_parser_thinking_disabled):
        ids = [_TEXT_A, _GEMMA4_TURN]
        assert gemma4_parser_thinking_disabled.is_reasoning_end_streaming(
            ids, [_GEMMA4_TURN]
        )

    def test_tool_response_thinking_enabled_no_trigger(self, gemma4_parser):
        ids = [_GEMMA4_CHANNEL_START, _TEXT_A, _GEMMA4_TOOL_RESPONSE]
        assert not gemma4_parser.is_reasoning_end_streaming(
            ids, [_GEMMA4_TOOL_RESPONSE]
        )

    def test_tool_response_thinking_disabled_triggers(
        self, gemma4_parser_thinking_disabled
    ):
        ids = [_TEXT_A, _GEMMA4_TOOL_RESPONSE]
        assert gemma4_parser_thinking_disabled.is_reasoning_end_streaming(
            ids, [_GEMMA4_TOOL_RESPONSE]
        )

    def test_non_trigger_returns_false(self, gemma4_parser):
        ids = [_GEMMA4_CHANNEL_START, _TEXT_A]
        assert not gemma4_parser.is_reasoning_end_streaming(ids, [_TEXT_A])

    def test_empty_delta_returns_false(self, gemma4_parser):
        ids = [_GEMMA4_CHANNEL_START, _TEXT_A]
        assert not gemma4_parser.is_reasoning_end_streaming(ids, [])

    def test_variable_delta_trigger(self, gemma4_parser):
        _assert_variable_delta_trigger(
            gemma4_parser, _GEMMA4_CHANNEL_START, _GEMMA4_CHANNEL_END
        )

    def test_variable_delta_no_trigger(self, gemma4_parser):
        _assert_variable_delta_no_trigger(gemma4_parser, _GEMMA4_CHANNEL_START)

    def test_realistic_sequence_replay(self, gemma4_parser):
        _assert_sequence_replay(
            gemma4_parser, _GEMMA4_CHANNEL_START, _GEMMA4_CHANNEL_END, 30
        )

    def test_parity_with_is_reasoning_end(self, gemma4_parser):
        _assert_parity(gemma4_parser, _GEMMA4_CHANNEL_START, _GEMMA4_CHANNEL_END)

    def test_parity_multi_token_delta(self, gemma4_parser):
        _assert_parity_multi_token(
            gemma4_parser, _GEMMA4_CHANNEL_START, _GEMMA4_CHANNEL_END
        )

    def test_tool_call_then_reenter_channel(self, gemma4_parser):
        _assert_trigger_reentry_returns_false(
            gemma4_parser, _GEMMA4_CHANNEL_START, _GEMMA4_TOOL_CALL
        )


# ── KimiK2 ──────────────────────────────────────────────────────────

_KIMI_THINK_START = 90
_KIMI_THINK_END = 91
_KIMI_TOOL_SECTION = 92

_KIMI_VOCAB = {
    "<think>": _KIMI_THINK_START,
    "</think>": _KIMI_THINK_END,
    "<|tool_calls_section_begin|>": _KIMI_TOOL_SECTION,
}


@pytest.fixture
def kimi_parser():
    return KimiK2Parser(make_mock_tokenizer(_KIMI_VOCAB))


@pytest.fixture
def kimi_parser_thinking_disabled():
    return KimiK2Parser(
        make_mock_tokenizer(_KIMI_VOCAB),
        chat_template_kwargs={"enable_thinking": False},
    )


class TestKimiK2Streaming:
    def test_think_end_returns_true(self, kimi_parser):
        ids = [_KIMI_THINK_START, _TEXT_A, _KIMI_THINK_END]
        assert kimi_parser.is_reasoning_end_streaming(ids, [_KIMI_THINK_END])

    def test_tool_section_start_triggers_end(self, kimi_parser):
        ids = [_KIMI_THINK_START, _TEXT_A, _KIMI_TOOL_SECTION]
        assert kimi_parser.is_reasoning_end_streaming(ids, [_KIMI_TOOL_SECTION])

    def test_non_trigger_returns_false(self, kimi_parser):
        ids = [_KIMI_THINK_START, _TEXT_A]
        assert not kimi_parser.is_reasoning_end_streaming(ids, [_TEXT_A])

    def test_empty_delta_returns_false(self, kimi_parser):
        ids = [_KIMI_THINK_START, _TEXT_A]
        assert not kimi_parser.is_reasoning_end_streaming(ids, [])

    def test_thinking_disabled_returns_true(self, kimi_parser_thinking_disabled):
        ids = [_TEXT_A]
        assert kimi_parser_thinking_disabled.is_reasoning_end_streaming(ids, [_TEXT_A])

    def test_thinking_disabled_empty_delta(self, kimi_parser_thinking_disabled):
        assert kimi_parser_thinking_disabled.is_reasoning_end_streaming([], [])

    def test_variable_delta_trigger(self, kimi_parser):
        _assert_variable_delta_trigger(kimi_parser, _KIMI_THINK_START, _KIMI_THINK_END)

    def test_variable_delta_no_trigger(self, kimi_parser):
        _assert_variable_delta_no_trigger(kimi_parser, _KIMI_THINK_START)

    def test_realistic_sequence_replay(self, kimi_parser):
        _assert_sequence_replay(kimi_parser, _KIMI_THINK_START, _KIMI_THINK_END)

    def test_parity_with_is_reasoning_end(self, kimi_parser):
        _assert_parity(kimi_parser, _KIMI_THINK_START, _KIMI_THINK_END)

    def test_parity_multi_token_delta(self, kimi_parser):
        _assert_parity_multi_token(kimi_parser, _KIMI_THINK_START, _KIMI_THINK_END)

    def test_tool_section_then_reenter_thinking(self, kimi_parser):
        _assert_trigger_reentry_returns_false(
            kimi_parser, _KIMI_THINK_START, _KIMI_TOOL_SECTION
        )


# ── Glm47Moe ────────────────────────────────────────────────────────

_GLM_THINK_START = 110
_GLM_THINK_END = 111
_GLM_TOOL_CALL = 112
_GLM_TOOL_CALL_END = 113

_GLM_VOCAB = {
    "<think>": _GLM_THINK_START,
    "</think>": _GLM_THINK_END,
    "<tool_call>": _GLM_TOOL_CALL,
    "</tool_call>": _GLM_TOOL_CALL_END,
}


@pytest.fixture
def glm_parser():
    return Glm47MoeParser(make_mock_tokenizer(_GLM_VOCAB))


@pytest.fixture
def glm_parser_thinking_disabled():
    return Glm47MoeParser(
        make_mock_tokenizer(_GLM_VOCAB),
        chat_template_kwargs={"enable_thinking": False},
    )


class TestGlm47MoeStreaming:
    def test_think_end_returns_true(self, glm_parser):
        ids = [_GLM_THINK_START, _TEXT_A, _GLM_THINK_END]
        assert glm_parser.is_reasoning_end_streaming(ids, [_GLM_THINK_END])

    def test_non_trigger_returns_false(self, glm_parser):
        ids = [_GLM_THINK_START, _TEXT_A]
        assert not glm_parser.is_reasoning_end_streaming(ids, [_TEXT_A])

    def test_empty_delta_returns_false(self, glm_parser):
        ids = [_GLM_THINK_START, _TEXT_A]
        assert not glm_parser.is_reasoning_end_streaming(ids, [])

    def test_thinking_disabled_returns_true(self, glm_parser_thinking_disabled):
        ids = [_TEXT_A]
        assert glm_parser_thinking_disabled.is_reasoning_end_streaming(ids, [_TEXT_A])

    def test_thinking_disabled_empty_delta(self, glm_parser_thinking_disabled):
        assert glm_parser_thinking_disabled.is_reasoning_end_streaming([], [])

    def test_variable_delta_trigger(self, glm_parser):
        _assert_variable_delta_trigger(glm_parser, _GLM_THINK_START, _GLM_THINK_END)

    def test_variable_delta_no_trigger(self, glm_parser):
        _assert_variable_delta_no_trigger(glm_parser, _GLM_THINK_START)

    def test_realistic_sequence_replay(self, glm_parser):
        _assert_sequence_replay(glm_parser, _GLM_THINK_START, _GLM_THINK_END)

    def test_parity_with_is_reasoning_end(self, glm_parser):
        _assert_parity(glm_parser, _GLM_THINK_START, _GLM_THINK_END)

    def test_parity_multi_token_delta(self, glm_parser):
        _assert_parity_multi_token(glm_parser, _GLM_THINK_START, _GLM_THINK_END)


# ── DeepSeekV4 ───────────────────────────────────────────────────────

_DSV4_THINK_START = 120
_DSV4_THINK_END = 121

_DSV4_VOCAB = {
    "<think>": _DSV4_THINK_START,
    "</think>": _DSV4_THINK_END,
}


@pytest.fixture
def dsv4_parser():
    return DeepSeekV4Parser(make_mock_tokenizer(_DSV4_VOCAB))


class TestDeepSeekV4Streaming:
    def test_think_end_returns_true(self, dsv4_parser):
        ids = [_DSV4_THINK_START, _TEXT_A, _DSV4_THINK_END]
        assert dsv4_parser.is_reasoning_end_streaming(ids, [_DSV4_THINK_END])

    def test_non_trigger_returns_false(self, dsv4_parser):
        ids = [_DSV4_THINK_START, _TEXT_A]
        assert not dsv4_parser.is_reasoning_end_streaming(ids, [_TEXT_A])

    def test_empty_delta_returns_false(self, dsv4_parser):
        ids = [_DSV4_THINK_START, _TEXT_A]
        assert not dsv4_parser.is_reasoning_end_streaming(ids, [])

    def test_variable_delta_trigger(self, dsv4_parser):
        _assert_variable_delta_trigger(dsv4_parser, _DSV4_THINK_START, _DSV4_THINK_END)

    def test_variable_delta_no_trigger(self, dsv4_parser):
        _assert_variable_delta_no_trigger(dsv4_parser, _DSV4_THINK_START)

    def test_realistic_sequence_replay(self, dsv4_parser):
        _assert_sequence_replay(dsv4_parser, _DSV4_THINK_START, _DSV4_THINK_END)

    def test_parity_multi_token_delta(self, dsv4_parser):
        _assert_parity_multi_token(dsv4_parser, _DSV4_THINK_START, _DSV4_THINK_END)


# ── MinimaxM2 ────────────────────────────────────────────────────────

_MM2_THINK_START = 130
_MM2_THINK_END = 131

_MM2_VOCAB = {
    "<think>": _MM2_THINK_START,
    "</think>": _MM2_THINK_END,
}


@pytest.fixture
def mm2_parser():
    return MinimaxM2Parser(make_mock_tokenizer(_MM2_VOCAB))


class TestMinimaxM2Streaming:
    def test_think_end_returns_true(self, mm2_parser):
        ids = [_MM2_THINK_START, _TEXT_A, _MM2_THINK_END]
        assert mm2_parser.is_reasoning_end_streaming(ids, [_MM2_THINK_END])

    def test_non_trigger_returns_false(self, mm2_parser):
        ids = [_MM2_THINK_START, _TEXT_A]
        assert not mm2_parser.is_reasoning_end_streaming(ids, [_TEXT_A])

    def test_empty_delta_returns_false(self, mm2_parser):
        ids = [_MM2_THINK_START, _TEXT_A]
        assert not mm2_parser.is_reasoning_end_streaming(ids, [])

    def test_variable_delta_trigger(self, mm2_parser):
        _assert_variable_delta_trigger(mm2_parser, _MM2_THINK_START, _MM2_THINK_END)

    def test_variable_delta_no_trigger(self, mm2_parser):
        _assert_variable_delta_no_trigger(mm2_parser, _MM2_THINK_START)

    def test_realistic_sequence_replay(self, mm2_parser):
        _assert_sequence_replay(mm2_parser, _MM2_THINK_START, _MM2_THINK_END)

    def test_parity_multi_token_delta(self, mm2_parser):
        _assert_parity_multi_token(mm2_parser, _MM2_THINK_START, _MM2_THINK_END)


# ── DeepSeekV32 (no reasoning) ───────────────────────────────────────

_DSV32_VOCAB: dict[str, int] = {}


@pytest.fixture
def dsv32_parser():
    return DeepSeekV32Parser(make_mock_tokenizer(_DSV32_VOCAB))


class TestDeepSeekV32Streaming:
    def test_always_returns_true(self, dsv32_parser):
        assert dsv32_parser.is_reasoning_end_streaming([_TEXT_A], [_TEXT_A])

    def test_empty_ids_returns_true(self, dsv32_parser):
        assert dsv32_parser.is_reasoning_end_streaming([], [])

    def test_long_sequence_returns_true(self, dsv32_parser):
        ids = [_TEXT_A] * 100
        assert dsv32_parser.is_reasoning_end_streaming(ids, [_TEXT_A])


# ── Inkling ─────────────────────────────────────────────────────────
#
# Inkling has no THINK_END token: its is_reasoning_end keys on the current
# block marker, returning True once the last block marker is content_text or
# end_sampling, False while it is content_thinking or message_model. The base
# THINK_END-only trigger set would leave streaming detection permanently False
# for Inkling, so its triggers are seeded from those end markers instead.

_INK_MSG_MODEL = 200001
_INK_TEXT = 200004
_INK_END_SAMPLING = 200006
_INK_THINKING = 200008
_INK_END_MESSAGE = 200010
_INK_TOOL_JSON = 200049

_INK_VOCAB = {
    "<|message_model|>": _INK_MSG_MODEL,
    "<|content_text|>": _INK_TEXT,
    "<|content_model_end_sampling|>": _INK_END_SAMPLING,
    "<|content_thinking|>": _INK_THINKING,
    "<|end_message|>": _INK_END_MESSAGE,
    "<|content_invoke_tool_json|>": _INK_TOOL_JSON,
}


@pytest.fixture
def inkling_parser():
    return InklingParser(make_mock_tokenizer(_INK_VOCAB))


class TestInklingStreaming:
    def test_content_text_triggers_end(self, inkling_parser):
        ids = [_INK_THINKING, _TEXT_A, _INK_TEXT]
        assert inkling_parser.is_reasoning_end_streaming(ids, [_INK_TEXT])

    def test_end_sampling_triggers_end(self, inkling_parser):
        ids = [_INK_THINKING, _TEXT_A, _INK_END_SAMPLING]
        assert inkling_parser.is_reasoning_end_streaming(ids, [_INK_END_SAMPLING])

    def test_active_reasoning_returns_false(self, inkling_parser):
        ids = [_INK_THINKING, _TEXT_A]
        assert not inkling_parser.is_reasoning_end_streaming(ids, [_TEXT_A])

    def test_message_model_matches_full_scan(self, inkling_parser):
        # A message-model header keeps reasoning open; neither streaming nor
        # the full scan should report an end.
        ids = [_INK_TEXT, _INK_MSG_MODEL]
        assert not inkling_parser.is_reasoning_end_streaming(ids, [_INK_MSG_MODEL])
        assert not inkling_parser.is_reasoning_end(ids)

    def test_empty_delta_returns_false(self, inkling_parser):
        ids = [_INK_THINKING, _TEXT_A]
        assert not inkling_parser.is_reasoning_end_streaming(ids, [])

    def test_variable_delta_trigger(self, inkling_parser):
        _assert_variable_delta_trigger(inkling_parser, _INK_THINKING, _INK_TEXT)

    def test_variable_delta_no_trigger(self, inkling_parser):
        _assert_variable_delta_no_trigger(inkling_parser, _INK_THINKING)

    def test_realistic_sequence_replay(self, inkling_parser):
        _assert_sequence_replay(inkling_parser, _INK_THINKING, _INK_TEXT)

    def test_parity_with_is_reasoning_end(self, inkling_parser):
        _assert_parity(inkling_parser, _INK_THINKING, _INK_TEXT)

    def test_parity_multi_token_delta(self, inkling_parser):
        _assert_parity_multi_token(inkling_parser, _INK_THINKING, _INK_TEXT)

    def test_text_then_reenter_thinking(self, inkling_parser):
        _assert_trigger_reentry_returns_false(inkling_parser, _INK_THINKING, _INK_TEXT)


# ── Adapter delegation ──────────────────────────────────────────────


class TestAdapterDelegation:
    """Verify that ParserEngineReasoningAdapter.is_reasoning_end_streaming
    correctly delegates to the engine's method."""

    def test_adapter_think_end(self):
        adapter = Qwen3ParserReasoningAdapter(make_mock_tokenizer(_QWEN3_VOCAB))
        ids = [_QWEN3_THINK_START, _TEXT_A, _QWEN3_THINK_END]
        assert adapter.is_reasoning_end_streaming(ids, [_QWEN3_THINK_END])

    def test_adapter_tool_call(self):
        adapter = Qwen3ParserReasoningAdapter(make_mock_tokenizer(_QWEN3_VOCAB))
        ids = [_QWEN3_THINK_START, _TEXT_A, _QWEN3_TOOL_CALL]
        assert adapter.is_reasoning_end_streaming(ids, [_QWEN3_TOOL_CALL])

    def test_adapter_non_trigger(self):
        adapter = Qwen3ParserReasoningAdapter(make_mock_tokenizer(_QWEN3_VOCAB))
        ids = [_QWEN3_THINK_START, _TEXT_A]
        assert not adapter.is_reasoning_end_streaming(ids, [_TEXT_A])

    def test_adapter_handles_iterator_delta(self):
        """delta_ids may be an iterator (e.g. itertools.islice)."""
        import itertools

        adapter = Qwen3ParserReasoningAdapter(make_mock_tokenizer(_QWEN3_VOCAB))
        ids = [_QWEN3_THINK_START, _TEXT_A, _QWEN3_THINK_END]
        delta_iter = itertools.islice([_QWEN3_THINK_END], 0, None)
        assert adapter.is_reasoning_end_streaming(ids, delta_iter)

    def test_adapter_gemma4(self):
        adapter = Gemma4ParserReasoningAdapter(make_mock_tokenizer(_GEMMA4_VOCAB))
        ids = [_GEMMA4_CHANNEL_START, _TEXT_A, _GEMMA4_TOOL_CALL]
        assert adapter.is_reasoning_end_streaming(ids, [_GEMMA4_TOOL_CALL])

    def test_adapter_kimi_thinking_disabled(self):
        adapter = KimiK2ParserReasoningAdapter(
            make_mock_tokenizer(_KIMI_VOCAB),
            chat_template_kwargs={"enable_thinking": False},
        )
        assert adapter.is_reasoning_end_streaming([_TEXT_A], [_TEXT_A])

    def test_adapter_dsv32_always_true(self):
        adapter = DeepSeekV32ParserReasoningAdapter(make_mock_tokenizer(_DSV32_VOCAB))
        assert adapter.is_reasoning_end_streaming([_TEXT_A], [_TEXT_A])

    def test_adapter_inkling_content_text(self):
        adapter = InklingParserReasoningAdapter(make_mock_tokenizer(_INK_VOCAB))
        ids = [_INK_THINKING, _TEXT_A, _INK_TEXT]
        assert adapter.is_reasoning_end_streaming(ids, [_INK_TEXT])
        assert not adapter.is_reasoning_end_streaming(
            [_INK_THINKING, _TEXT_A], [_TEXT_A]
        )


# ── Active reasoning negative tests ─────────────────────────────────


class TestActiveReasoningNegative:
    """Long sequences of non-trigger tokens must always return False.
    This is the hot path we're optimizing."""

    def test_qwen3_long_reasoning(self, qwen3_parser):
        reasoning_tokens = [_TEXT_A, _TEXT_B, _TEXT_C] * 100
        base = [_QWEN3_THINK_START] + reasoning_tokens
        for i in range(len(reasoning_tokens)):
            all_tokens = base[: i + 2]
            delta = [reasoning_tokens[i]]
            assert not qwen3_parser.is_reasoning_end_streaming(all_tokens, delta)

    def test_gemma4_long_reasoning(self, gemma4_parser):
        reasoning_tokens = [_TEXT_A, _TEXT_B, _TEXT_C] * 100
        base = [_GEMMA4_CHANNEL_START] + reasoning_tokens
        for i in range(len(reasoning_tokens)):
            all_tokens = base[: i + 2]
            delta = [reasoning_tokens[i]]
            assert not gemma4_parser.is_reasoning_end_streaming(all_tokens, delta)

    def test_kimi_long_reasoning(self, kimi_parser):
        reasoning_tokens = [_TEXT_A, _TEXT_B, _TEXT_C] * 100
        base = [_KIMI_THINK_START] + reasoning_tokens
        for i in range(len(reasoning_tokens)):
            all_tokens = base[: i + 2]
            delta = [reasoning_tokens[i]]
            assert not kimi_parser.is_reasoning_end_streaming(all_tokens, delta)
