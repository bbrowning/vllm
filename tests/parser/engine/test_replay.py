# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Data-driven replay tests for unified parser engines.

Loads token sequences from JSONL files and replays them at different
chunk sizes to verify chunk-size invariance: the same token sequence
must produce identical output regardless of how tokens are batched.
"""

from __future__ import annotations

import dataclasses

import pytest

from tests.parser.engine.replay_harness import (
    _test_request,
    assert_no_terminal_leakage,
    assert_parse_output,
    collect_output,
    make_mock_tokenizer,
    replay_streaming,
    replay_with_text_holdback,
)
from tests.parser.engine.trace_builder import build_samples
from vllm.parser.abstract_parser import DelegatingParser, Parser
from vllm.parser.engine.registered_adapters import (
    DeepSeekV4Parser,
    Gemma4Parser,
    NemotronV3Parser,
    NemotronV3ParserReasoningAdapter,
    NemotronV3ParserToolAdapter,
    Qwen3Parser,
    Qwen3XMLParser,
)

_ENGINE_PARSERS: dict[str, type[Parser]] = {
    "deepseek_v4_engine": DeepSeekV4Parser,
    "gemma4_engine": Gemma4Parser,
    "qwen3_engine": Qwen3Parser,
    "qwen3_xml_engine": Qwen3XMLParser,
    "qwen3_coder_engine": Qwen3XMLParser,
    "nemotron_v3_engine": NemotronV3Parser,
}

_gemma4_samples = build_samples("gemma4")
_nemotron_v3_samples = build_samples("nemotron_v3")
_qwen3_samples = build_samples("qwen3")
_deepseek_v4_samples = build_samples("deepseek_v4")

_GEMMA4_TERMINALS = ["<|channel>", "<channel|>", "<|tool_call>", "<tool_call|>"]

HOLDBACK_CONFIGS = [6, 12, 24]


@pytest.mark.parametrize("holdback", HOLDBACK_CONFIGS, ids=lambda h: f"holdback{h}")
@pytest.mark.parametrize("chunk_size", [3, 5, 10], ids=lambda c: f"chunk{c}")
@pytest.mark.parametrize("sample", _gemma4_samples, ids=lambda s: s.id)
class TestGemma4ReplayWithHoldback:
    """Replay with simulated detokenizer holdback."""

    def test_replay(self, sample, chunk_size, holdback):
        tokenizer = make_mock_tokenizer(sample)
        parser = Gemma4Parser(tokenizer)
        deltas = replay_streaming(
            parser,
            sample.tokens,
            chunk_size=chunk_size,
            holdback_chars=holdback,
        )
        output = collect_output(deltas)

        assert_parse_output(output, sample)
        assert_no_terminal_leakage(
            output,
            _GEMMA4_TERMINALS,
            context=f"chunk_size={chunk_size}, holdback={holdback}",
        )


_QWEN3_TERMINALS = [
    "<think>",
    "</think>",
    "<tool_call>",
    "</tool_call>",
    "<function=",
    "</function>",
]


@pytest.mark.parametrize("holdback", HOLDBACK_CONFIGS, ids=lambda h: f"holdback{h}")
@pytest.mark.parametrize("chunk_size", [5, 10], ids=lambda c: f"chunk{c}")
@pytest.mark.parametrize("sample", _qwen3_samples, ids=lambda s: s.id)
class TestQwen3ReplayWithHoldback:
    """Replay Qwen3 with simulated detokenizer holdback."""

    def test_replay(self, sample, chunk_size, holdback):
        tokenizer = make_mock_tokenizer(sample)
        parser = Qwen3Parser(tokenizer)
        deltas = replay_streaming(
            parser,
            sample.tokens,
            chunk_size=chunk_size,
            holdback_chars=holdback,
        )
        output = collect_output(deltas)

        assert_parse_output(output, sample)
        assert_no_terminal_leakage(
            output,
            _QWEN3_TERMINALS,
            context=f"chunk_size={chunk_size}, holdback={holdback}",
        )


TEXT_HOLDBACK_DELAYS = [1, 2, 3]


@pytest.mark.parametrize("delay", TEXT_HOLDBACK_DELAYS, ids=lambda d: f"delay{d}")
@pytest.mark.parametrize("sample", _gemma4_samples, ids=lambda s: s.id)
class TestGemma4TextHoldback:
    """Replay with production-like text/token-ID misalignment.

    In production the detokenizer sends token IDs immediately but holds
    back text by N tokens.  This exercises the TokenIDScanner deferred
    terminal path that aligned-holdback tests do not cover.
    """

    def test_replay(self, sample, delay):
        tokenizer = make_mock_tokenizer(sample)
        parser = Gemma4Parser(tokenizer)
        deltas = replay_with_text_holdback(parser, sample.tokens, text_delay=delay)
        output = collect_output(deltas)

        assert_parse_output(output, sample)
        assert_no_terminal_leakage(
            output,
            _GEMMA4_TERMINALS,
            context=f"text_delay={delay}",
        )


class TestParserEngineAdjustRequest:
    """Verify ParserEngine and its adapters set skip_special_tokens=False."""

    def test_adjust_request_disables_skip_special_tokens(self):
        sample = _gemma4_samples[0]
        tokenizer = make_mock_tokenizer(sample)
        parser = Gemma4Parser(tokenizer)
        request = _test_request()
        assert request.skip_special_tokens is True
        adjusted = parser.adjust_request(request)
        assert adjusted.skip_special_tokens is False

    @pytest.mark.parametrize(
        "adapter_base",
        [
            pytest.param("ParserEngineReasoningAdapter", id="reasoning"),
            pytest.param("ParserEngineToolAdapter", id="tool"),
        ],
    )
    def test_adapter_delegates_adjust_request(self, adapter_base):
        """Adapters must delegate adjust_request so that
        skip_special_tokens=False reaches the detokenizer even when the
        serving code only calls the adapter (not the unified parser)."""
        import vllm.parser.engine.adapters as adapters_mod

        base_cls = getattr(adapters_mod, adapter_base)
        sample = _nemotron_v3_samples[0]
        tokenizer = make_mock_tokenizer(sample)

        adapter_cls = type(
            f"Test{adapter_base}",
            (base_cls,),
            {"_parser_engine_cls": NemotronV3Parser},
        )
        adapter = adapter_cls(tokenizer)
        request = _test_request()
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

    def test_replay(self, sample, chunk_size):
        tokenizer = make_mock_tokenizer(sample)
        parser = NemotronV3Parser(tokenizer)
        deltas = replay_streaming(parser, sample.tokens, chunk_size=chunk_size)
        output = collect_output(deltas)

        assert_parse_output(output, sample)
        assert_no_terminal_leakage(output, _NEMOTRON_V3_TERMINALS)


_DSV4_TERMINALS = [
    "<think>",
    "</think>",
]


@pytest.mark.parametrize("chunk_size", NEMOTRON_CHUNK_SIZES, ids=lambda c: f"chunk{c}")
@pytest.mark.parametrize("sample", _deepseek_v4_samples, ids=lambda s: s.id)
class TestDeepSeekV4Replay:
    """Replay DeepSeek V4 token sequences at different chunk sizes."""

    def test_replay(self, sample, chunk_size):
        tokenizer = make_mock_tokenizer(sample)
        kwargs = {}
        if sample.chat_template_kwargs:
            kwargs["chat_template_kwargs"] = sample.chat_template_kwargs
        parser = DeepSeekV4Parser(tokenizer, sample.tools, **kwargs)
        deltas = replay_streaming(
            parser, sample.tokens, chunk_size=chunk_size, tools=sample.tools
        )
        output = collect_output(deltas)

        assert_parse_output(output, sample)
        assert_no_terminal_leakage(output, _DSV4_TERMINALS)


@pytest.mark.parametrize("delay", TEXT_HOLDBACK_DELAYS, ids=lambda d: f"delay{d}")
@pytest.mark.parametrize("sample", _deepseek_v4_samples, ids=lambda s: s.id)
class TestDeepSeekV4TextHoldback:
    """Replay DeepSeek V4 with production-like text/token-ID misalignment."""

    def test_replay(self, sample, delay):
        tokenizer = make_mock_tokenizer(sample)
        kwargs = {}
        if sample.chat_template_kwargs:
            kwargs["chat_template_kwargs"] = sample.chat_template_kwargs
        parser = DeepSeekV4Parser(tokenizer, sample.tools, **kwargs)
        deltas = replay_with_text_holdback(
            parser, sample.tokens, text_delay=delay, tools=sample.tools
        )
        output = collect_output(deltas)

        assert_parse_output(output, sample)
        assert_no_terminal_leakage(
            output, _DSV4_TERMINALS, context=f"text_delay={delay}"
        )


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
        parser = NemotronV3Parser(tokenizer)

        request = _test_request()

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
            first_text, first_ids, request, prompt_token_ids=[], finished=False
        )
        result2 = parser.parse_delta(
            last_text_missing, last_ids, request, finished=True
        )

        output = collect_output([result1, result2])

        tool_calls_only = dataclasses.replace(
            sample, expected_reasoning=None, expected_content=None
        )
        assert_parse_output(output, tool_calls_only)


_TOOL_CALL_SAMPLES = (
    [
        (Qwen3Parser, s)
        for s in _qwen3_samples
        if s.expected_tool_calls and s.expected_reasoning
    ]
    + [
        (NemotronV3Parser, s)
        for s in _nemotron_v3_samples
        if s.expected_tool_calls and s.expected_reasoning
    ]
    + [
        (Gemma4Parser, s)
        for s in _gemma4_samples
        if s.expected_tool_calls and s.expected_reasoning
    ]
    + [
        (DeepSeekV4Parser, s)
        for s in _deepseek_v4_samples
        if s.expected_tool_calls and s.expected_reasoning
    ]
)


def _suppressed_expectations(sample) -> tuple[str, str]:
    """Compute expected (reasoning, content) when tools are suppressed.

    When an explicit reasoning-end delimiter (``</think>``, ``<channel|>``)
    is present, reasoning ends there and the tool call block becomes content.
    When reasoning ends implicitly (the tool-start token triggers both
    REASONING_END and TOOL_CALL_START), reasoning still ends at the tool
    start and the raw tool call block becomes content text — only the
    structured tool parsing is suppressed, not the reasoning boundary.
    """
    full_text = "".join(text for _, text in sample.tokens)
    reasoning = sample.expected_reasoning
    idx = full_text.find(reasoning)
    if idx < 0:
        return (full_text, "")
    after_reasoning = full_text[idx + len(reasoning) :]
    for delim in ("</think>", "<channel|>"):
        pos = after_reasoning.find(delim)
        if pos >= 0:
            return (reasoning, after_reasoning[pos + len(delim) :])
    for delim in ("<tool_call>",):
        pos = after_reasoning.find(delim)
        if pos >= 0:
            return (reasoning, after_reasoning[pos:])
    return (full_text, "")


_DUMMY_TOOLS = [
    {
        "type": "function",
        "function": {"name": "stub", "parameters": {"type": "object"}},
    }
]


@pytest.mark.parametrize("chunk_size", [1, 5, None], ids=lambda c: f"chunk{c}")
@pytest.mark.parametrize(
    "parser_cls,sample",
    _TOOL_CALL_SAMPLES,
    ids=lambda v: v.id if hasattr(v, "id") else v.__name__,
)
class TestSkipToolParsingReplay:
    """Replay with skip_tool_parsing=True (tool_choice='none').

    Verifies that reasoning is extracted normally and the raw tool call
    block appears as content text with no tool calls parsed.
    """

    def test_replay(self, parser_cls, sample, chunk_size):
        tokenizer = make_mock_tokenizer(sample)
        kwargs = {}
        if sample.chat_template_kwargs:
            kwargs["chat_template_kwargs"] = sample.chat_template_kwargs
        parser = parser_cls(tokenizer, **kwargs)

        request = _test_request()
        request.tool_choice = "none"
        request.tools = _DUMMY_TOOLS

        all_ids = [tid for tid, _ in sample.tokens]
        all_texts = [text for _, text in sample.tokens]
        if chunk_size is None:
            chunk_size = len(all_ids)

        results = []
        chunks = list(range(0, len(all_ids), chunk_size))
        for i, start in enumerate(chunks):
            end = min(start + chunk_size, len(all_ids))
            is_last = i == len(chunks) - 1
            result = parser.parse_delta(
                "".join(all_texts[start:end]),
                all_ids[start:end],
                request,
                prompt_token_ids=[] if start == 0 else None,
                finished=is_last,
            )
            results.append(result)

        output = collect_output(results)

        expected_reasoning, expected_content = _suppressed_expectations(sample)

        assert output.reasoning == expected_reasoning, (
            f"Reasoning mismatch:\n"
            f"  expected: {expected_reasoning!r}\n"
            f"  actual:   {output.reasoning!r}"
        )
        assert output.tool_calls == [], (
            f"Expected no tool calls but got {output.tool_calls}"
        )
        assert output.content == expected_content, (
            f"Content mismatch:\n"
            f"  expected: {expected_content!r}\n"
            f"  actual:   {output.content!r}"
        )


class _NemotronV3DelegatingEngine(DelegatingParser):
    reasoning_parser_cls = NemotronV3ParserReasoningAdapter
    tool_parser_cls = NemotronV3ParserToolAdapter


_NEMOTRON_V3_DELEGATING_FAIL_IDS = frozenset(
    {
        "nemotron_v3-think-then-content",
        "nemotron_v3-content-only",
        "nemotron_v3-think-content-tool",
        "nemotron_v3-empty-reasoning-content",
    }
)

_nemotron_v3_delegating_fail_samples = [
    s for s in _nemotron_v3_samples if s.id in _NEMOTRON_V3_DELEGATING_FAIL_IDS
]


@pytest.mark.parametrize("chunk_size", NEMOTRON_CHUNK_SIZES, ids=lambda c: f"chunk{c}")
@pytest.mark.parametrize(
    "sample", _nemotron_v3_delegating_fail_samples, ids=lambda s: s.id
)
class TestNemotronV3DelegatingEngineReplay:
    """Replay nemotron_v3 through DelegatingParser with engine adapters.

    Exercises the adapter-based delegating path where reasoning and tool
    parsing use separate ParserEngine instances.
    """

    def test_replay(self, sample, chunk_size):
        tokenizer = make_mock_tokenizer(sample)
        parser = _NemotronV3DelegatingEngine(tokenizer, sample.tools)
        deltas = replay_streaming(
            parser, sample.tokens, chunk_size=chunk_size, tools=sample.tools
        )
        output = collect_output(deltas)

        assert_parse_output(output, sample)
        assert_no_terminal_leakage(output, _NEMOTRON_V3_TERMINALS)


class TestAdapterReferences:
    """Verify make_adapters sets reasoning/tool parser class refs on parser engine
    parser classes so the serving layer finds them and calls adjust_request."""

    @pytest.mark.parametrize(
        "parser_name",
        list(_ENGINE_PARSERS.keys()),
    )
    def test_adapter_cls_refs_set(self, parser_name):
        parser_cls = _ENGINE_PARSERS[parser_name]
        assert parser_cls.reasoning_parser_cls is not None, (
            f"{parser_name}: reasoning_parser_cls is None"
        )
        assert parser_cls.tool_parser_cls is not None, (
            f"{parser_name}: tool_parser_cls is None"
        )
