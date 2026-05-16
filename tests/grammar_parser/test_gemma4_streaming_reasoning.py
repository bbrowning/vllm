# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Reproduce streaming tool call failure when reasoning precedes a tool call.

When Gemma4 is run with both --reasoning-parser gemma4_grammar and
--tool-call-parser gemma4_grammar, streaming tool calls after reasoning
fail: the tool call body leaks as content and no tool_calls are returned.
Non-streaming works fine.

The root cause is a delta_text / delta_token_ids mismatch created by the
DelegatingParser: the reasoning parser's drop_tokens strip <|tool_call>
text from delta_text, but extract_content_ids() keeps the token ID.  The
tool parser's scanner defers the terminal (text not found), and the tool
call body becomes content.

This test feeds the exact model output through the full DelegatingParser
pipeline (both reasoning and tool parsers) to reproduce the failure.
Tokens are batched (stream_interval=10 style) so that <channel|> and
<|tool_call> land in the same parse_delta() call — the condition that
triggers the bug.
"""

import json
from unittest.mock import MagicMock

import pytest

from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
)
from vllm.entrypoints.openai.engine.protocol import DeltaMessage
from vllm.grammar_parser.registered_parsers import (
    GrammarGemma4ReasoningParser,
    GrammarGemma4ToolParser,
)
from vllm.parser.abstract_parser import _WrappedParser

# ── Special token IDs (arbitrary but consistent) ─────────────────────
CHANNEL_START_ID = 50  # <|channel>
CHANNEL_END_ID = 51  # <channel|>
TOOL_CALL_START_ID = 48  # <|tool_call>
TOOL_CALL_END_ID = 49  # <tool_call|>
QUOTED_ID = 52  # <|"|>

SPECIAL_TOKEN_MAP = {
    CHANNEL_START_ID: "<|channel>",
    CHANNEL_END_ID: "<channel|>",
    TOOL_CALL_START_ID: "<|tool_call>",
    TOOL_CALL_END_ID: "<tool_call|>",
    QUOTED_ID: '<|"|>',
}

SPECIAL_TEXT_TO_ID = {v: k for k, v in SPECIAL_TOKEN_MAP.items()}

# ── Model output (exact tokens from the user's dump) ────────────────
# <|channel>thought\n...reasoning...<channel|><|tool_call>
# call:get_current_weather{city:<|"|>Dallas<|"|>,state:<|"|>TX<|"|>,
# unit:<|"|>fahrenheit<|"|>}<tool_call|>

REASONING_TEXT = (
    "The user is asking for the current weather in Dallas, Texas, "
    "and specifically requests the temperature in Fahrenheit. "
    "I have a tool `get_current_weather` that can provide this "
    "information. I should call this tool with `city='Dallas'`, "
    "`state='TX'`, and `unit='fahrenheit'`."
)

# Break reasoning into word-level tokens (IDs 1000+)
_reasoning_words = REASONING_TEXT.split(" ")
_REGULAR_TOKEN_START = 1000
REASONING_TOKENS: list[tuple[int, str]] = []
for i, word in enumerate(_reasoning_words):
    prefix = " " if i > 0 else ""
    REASONING_TOKENS.append((_REGULAR_TOKEN_START + i, prefix + word))

# Tool call body tokens (IDs 2000+)
TOOL_BODY_TOKENS: list[tuple[int, str]] = [
    (2000, "call"),
    (2001, ":"),
    (2002, "get_current_weather"),
    (2003, "{"),
    (2004, "city"),
    (2005, ":"),
    # <|"|> Dallas <|"|>
    (2006, "Dallas"),
    (2007, ","),
    (2008, "state"),
    (2009, ":"),
    # <|"|> TX <|"|>
    (2010, "TX"),
    (2011, ","),
    (2012, "unit"),
    (2013, ":"),
    # <|"|> fahrenheit <|"|>
    (2014, "fahrenheit"),
    (2015, "}"),
]

# Full token sequence matching the model output
FULL_TOKEN_SEQUENCE: list[tuple[int, str]] = []

# <|channel>
FULL_TOKEN_SEQUENCE.append((CHANNEL_START_ID, "<|channel>"))
# thought\n
FULL_TOKEN_SEQUENCE.append((3000, "thought"))
FULL_TOKEN_SEQUENCE.append((3001, "\n"))
# reasoning text
FULL_TOKEN_SEQUENCE.extend(REASONING_TOKENS)
# <channel|>
FULL_TOKEN_SEQUENCE.append((CHANNEL_END_ID, "<channel|>"))
# <|tool_call>
FULL_TOKEN_SEQUENCE.append((TOOL_CALL_START_ID, "<|tool_call>"))
# call:get_current_weather{
FULL_TOKEN_SEQUENCE.extend(TOOL_BODY_TOKENS[:4])
# city:
FULL_TOKEN_SEQUENCE.extend(TOOL_BODY_TOKENS[4:6])
# <|"|>Dallas<|"|>
FULL_TOKEN_SEQUENCE.append((QUOTED_ID, '<|"|>'))
FULL_TOKEN_SEQUENCE.append(TOOL_BODY_TOKENS[6])  # Dallas
FULL_TOKEN_SEQUENCE.append((QUOTED_ID, '<|"|>'))
# ,state:
FULL_TOKEN_SEQUENCE.extend(TOOL_BODY_TOKENS[7:10])
# <|"|>TX<|"|>
FULL_TOKEN_SEQUENCE.append((QUOTED_ID, '<|"|>'))
FULL_TOKEN_SEQUENCE.append(TOOL_BODY_TOKENS[10])  # TX
FULL_TOKEN_SEQUENCE.append((QUOTED_ID, '<|"|>'))
# ,unit:
FULL_TOKEN_SEQUENCE.extend(TOOL_BODY_TOKENS[11:14])
# <|"|>fahrenheit<|"|>
FULL_TOKEN_SEQUENCE.append((QUOTED_ID, '<|"|>'))
FULL_TOKEN_SEQUENCE.append(TOOL_BODY_TOKENS[14])  # fahrenheit
FULL_TOKEN_SEQUENCE.append((QUOTED_ID, '<|"|>'))
# }
FULL_TOKEN_SEQUENCE.append(TOOL_BODY_TOKENS[15])  # }
# <tool_call|>
FULL_TOKEN_SEQUENCE.append((TOOL_CALL_END_ID, "<tool_call|>"))

# Build a complete token-id-to-text map for the mock tokenizer
_TOKEN_DECODE_MAP: dict[int, str] = {}
for tid, text in FULL_TOKEN_SEQUENCE:
    _TOKEN_DECODE_MAP[tid] = text


# ── Mock tokenizer ───────────────────────────────────────────────────


def _make_mock_tokenizer():
    tokenizer = MagicMock()

    vocab = dict(SPECIAL_TEXT_TO_ID)
    tokenizer.get_vocab.return_value = vocab
    tokenizer.encode.return_value = [tid for tid, _ in FULL_TOKEN_SEQUENCE]

    def decode(ids, skip_special_tokens=False):
        parts = []
        for tid in ids:
            if skip_special_tokens and tid in SPECIAL_TOKEN_MAP:
                continue
            text = _TOKEN_DECODE_MAP.get(tid, f"?{tid}?")
            parts.append(text)
        return "".join(parts)

    tokenizer.decode.side_effect = decode
    return tokenizer


# ── Helpers ──────────────────────────────────────────────────────────


def _stream_tokens_batched(
    parser, tokenizer, request, batch_size=10, prompt_token_ids=None
) -> list[DeltaMessage | None]:
    """Feed tokens in batches through parse_delta.

    Uses batch_size tokens per call (like --stream-interval 10) to
    reproduce the bug where <channel|> and <|tool_call> land in the
    same parse_delta() call.
    """
    token_ids = tokenizer.encode("", add_special_tokens=False)
    results: list[DeltaMessage | None] = []

    for start in range(0, len(token_ids), batch_size):
        batch_ids = token_ids[start : start + batch_size]
        delta_text = tokenizer.decode(batch_ids)
        result = parser.parse_delta(
            delta_text, batch_ids, request, prompt_token_ids=prompt_token_ids
        )
        prompt_token_ids = None
        results.append(result)
    return results


def _collect_fields(results):
    reasoning = "".join(r.reasoning for r in results if r and r.reasoning)
    content = "".join(r.content for r in results if r and r.content)
    tool_calls = [tc for r in results if r and r.tool_calls for tc in r.tool_calls]
    return reasoning, content, tool_calls


# ── Fixtures ─────────────────────────────────────────────────────────


@pytest.fixture
def mock_tokenizer():
    return _make_mock_tokenizer()


@pytest.fixture
def parser(mock_tokenizer):
    _WrappedParser.reasoning_parser_cls = GrammarGemma4ReasoningParser
    _WrappedParser.tool_parser_cls = GrammarGemma4ToolParser
    return _WrappedParser(mock_tokenizer)


@pytest.fixture
def request_obj():
    return ChatCompletionRequest(
        model="test-model",
        messages=[{"role": "user", "content": "hi"}],
    )


# ── Tests ────────────────────────────────────────────────────────────


class TestGemma4StreamingReasoningThenToolCall:
    """Reproduce streaming failure: reasoning prefix -> tool call.

    The model output is:
        <|channel>thought\\n...reasoning...<channel|>
        <|tool_call>call:get_current_weather{city:<|"|>Dallas<|"|>,...}<tool_call|>

    Non-streaming correctly extracts the tool call; streaming does not
    when tokens are batched such that <channel|> and <|tool_call> are
    in the same parse_delta() call.
    """

    def test_tool_call_extracted(self, parser, mock_tokenizer, request_obj):
        """Tool calls must be extracted from streaming output."""
        results = _stream_tokens_batched(
            parser,
            mock_tokenizer,
            request_obj,
            batch_size=10,
            prompt_token_ids=[],
        )

        reasoning, content, tool_calls = _collect_fields(results)

        # Tool call should be extracted
        assert len(tool_calls) > 0, (
            f"Expected tool_calls but got none. "
            f"content={content!r}, reasoning={reasoning[:80]!r}..."
        )

        # Verify the tool call details
        names = [
            tc.function.name for tc in tool_calls if tc.function and tc.function.name
        ]
        assert "get_current_weather" in names, (
            f"Expected get_current_weather, got {names}"
        )

        args_text = "".join(
            tc.function.arguments
            for tc in tool_calls
            if tc.function and tc.function.arguments
        )
        if args_text:
            parsed_args = json.loads(args_text)
            assert parsed_args.get("city") == "Dallas"
            assert parsed_args.get("state") == "TX"
            assert parsed_args.get("unit") == "fahrenheit"

    def test_tool_call_text_not_in_content(self, parser, mock_tokenizer, request_obj):
        """Tool call body must not leak into content."""
        results = _stream_tokens_batched(
            parser,
            mock_tokenizer,
            request_obj,
            batch_size=10,
            prompt_token_ids=[],
        )

        _, content, _ = _collect_fields(results)

        assert "call:" not in content, (
            f"Tool call text leaked into content: {content!r}"
        )
        assert "get_current_weather" not in content, (
            f"Function name leaked into content: {content!r}"
        )

    def test_reasoning_extracted(self, parser, mock_tokenizer, request_obj):
        """Reasoning content should be captured."""
        results = _stream_tokens_batched(
            parser,
            mock_tokenizer,
            request_obj,
            batch_size=10,
            prompt_token_ids=[],
        )

        reasoning, _, _ = _collect_fields(results)

        assert "weather" in reasoning.lower(), (
            f"Expected reasoning about weather, got: {reasoning[:100]!r}"
        )


# ── Second model output: two tool calls with holdback ────────────────

REASONING_TEXT_2 = (
    "The user wants me to:\n"
    "1. Perform some reasoning.\n"
    "2. Call a tool to fetch the hostname.\n"
    "3. Call a tool to fetch the current date.\n"
    "\n"
    "Since I am an AI assistant (opencode), I can use the "
    "`bash` tool to execute commands.\n"
    "To get the hostname, I can run `hostname`.\n"
    "To get the current date, I can run `date`.\n"
    "\n"
    "I should do this in a single response with "
    "multiple tool calls for efficiency."
)

_reasoning_words_2 = REASONING_TEXT_2.split(" ")
_R2_TOKEN_START = 4000
REASONING_TOKENS_2: list[tuple[int, str]] = []
for i, word in enumerate(_reasoning_words_2):
    prefix = " " if i > 0 else ""
    REASONING_TOKENS_2.append((_R2_TOKEN_START + i, prefix + word))

TOOL_BODY_TOKENS_2A: list[tuple[int, str]] = [
    (5000, "call"),
    (5001, ":"),
    (5002, "bash"),
    (5003, "{"),
    (5004, "command"),
    (5005, ":"),
    (5006, "hostname"),
    (5007, ","),
    (5008, "description"),
    (5009, ":"),
    (5010, "Fetch the hostname of the system."),
    (5011, "}"),
]

TOOL_BODY_TOKENS_2B: list[tuple[int, str]] = [
    (6000, "call"),
    (6001, ":"),
    (6002, "bash"),
    (6003, "{"),
    (6004, "command"),
    (6005, ":"),
    (6006, "date"),
    (6007, ","),
    (6008, "description"),
    (6009, ":"),
    (6010, "Fetch the current system date and time."),
    (6011, "}"),
]

FULL_TOKEN_SEQUENCE_2: list[tuple[int, str]] = []
FULL_TOKEN_SEQUENCE_2.append((CHANNEL_START_ID, "<|channel>"))
FULL_TOKEN_SEQUENCE_2.append((3000, "thought"))
FULL_TOKEN_SEQUENCE_2.append((3001, "\n"))
FULL_TOKEN_SEQUENCE_2.extend(REASONING_TOKENS_2)
FULL_TOKEN_SEQUENCE_2.append((CHANNEL_END_ID, "<channel|>"))
# First tool call
FULL_TOKEN_SEQUENCE_2.append((TOOL_CALL_START_ID, "<|tool_call>"))
FULL_TOKEN_SEQUENCE_2.extend(TOOL_BODY_TOKENS_2A[:6])
FULL_TOKEN_SEQUENCE_2.append((QUOTED_ID, '<|"|>'))
FULL_TOKEN_SEQUENCE_2.append(TOOL_BODY_TOKENS_2A[6])  # hostname
FULL_TOKEN_SEQUENCE_2.append((QUOTED_ID, '<|"|>'))
FULL_TOKEN_SEQUENCE_2.extend(TOOL_BODY_TOKENS_2A[7:10])
FULL_TOKEN_SEQUENCE_2.append((QUOTED_ID, '<|"|>'))
FULL_TOKEN_SEQUENCE_2.append(TOOL_BODY_TOKENS_2A[10])  # description value
FULL_TOKEN_SEQUENCE_2.append((QUOTED_ID, '<|"|>'))
FULL_TOKEN_SEQUENCE_2.append(TOOL_BODY_TOKENS_2A[11])  # }
FULL_TOKEN_SEQUENCE_2.append((TOOL_CALL_END_ID, "<tool_call|>"))
# Second tool call
FULL_TOKEN_SEQUENCE_2.append((TOOL_CALL_START_ID, "<|tool_call>"))
FULL_TOKEN_SEQUENCE_2.extend(TOOL_BODY_TOKENS_2B[:6])
FULL_TOKEN_SEQUENCE_2.append((QUOTED_ID, '<|"|>'))
FULL_TOKEN_SEQUENCE_2.append(TOOL_BODY_TOKENS_2B[6])  # date
FULL_TOKEN_SEQUENCE_2.append((QUOTED_ID, '<|"|>'))
FULL_TOKEN_SEQUENCE_2.extend(TOOL_BODY_TOKENS_2B[7:10])
FULL_TOKEN_SEQUENCE_2.append((QUOTED_ID, '<|"|>'))
FULL_TOKEN_SEQUENCE_2.append(TOOL_BODY_TOKENS_2B[10])  # description value
FULL_TOKEN_SEQUENCE_2.append((QUOTED_ID, '<|"|>'))
FULL_TOKEN_SEQUENCE_2.append(TOOL_BODY_TOKENS_2B[11])  # }
FULL_TOKEN_SEQUENCE_2.append((TOOL_CALL_END_ID, "<tool_call|>"))

_TOKEN_DECODE_MAP_2: dict[int, str] = {}
for tid, text in FULL_TOKEN_SEQUENCE_2:
    _TOKEN_DECODE_MAP_2[tid] = text


def _make_mock_tokenizer_2():
    tokenizer = MagicMock()

    vocab = dict(SPECIAL_TEXT_TO_ID)
    tokenizer.get_vocab.return_value = vocab
    tokenizer.encode.return_value = [tid for tid, _ in FULL_TOKEN_SEQUENCE_2]

    def decode(ids, skip_special_tokens=False):
        parts = []
        for tid in ids:
            if skip_special_tokens and tid in SPECIAL_TOKEN_MAP:
                continue
            text = _TOKEN_DECODE_MAP_2.get(tid, f"?{tid}?")
            parts.append(text)
        return "".join(parts)

    tokenizer.decode.side_effect = decode
    return tokenizer


def _stream_tokens_with_holdback(
    parser,
    tokenizer,
    request,
    batch_size=10,
    holdback_chars=12,
    prompt_token_ids=None,
) -> list[DeltaMessage | None]:
    """Feed tokens in batches with simulated detokenizer holdback.

    Instead of decoding each batch independently, simulates incremental
    decoding where the detokenizer holds back the last N characters of
    decoded text until the next batch arrives.  This reproduces the
    delta_text / delta_token_ids mismatch that occurs in production.
    """
    token_ids = tokenizer.encode("", add_special_tokens=False)
    results: list[DeltaMessage | None] = []
    prev_safe_text = ""

    for start in range(0, len(token_ids), batch_size):
        batch_end = min(start + batch_size, len(token_ids))
        batch_ids = token_ids[start:batch_end]

        full_decoded = tokenizer.decode(token_ids[:batch_end])

        if batch_end < len(token_ids):
            safe_len = max(0, len(full_decoded) - holdback_chars)
            safe_text = full_decoded[:safe_len]
        else:
            safe_text = full_decoded

        delta_text = safe_text[len(prev_safe_text) :]
        prev_safe_text = safe_text

        result = parser.parse_delta(
            delta_text,
            batch_ids,
            request,
            prompt_token_ids=prompt_token_ids,
        )
        prompt_token_ids = None
        results.append(result)
    return results


class TestGemma4ReasoningTruncationWithHoldback:
    """Reproduce reasoning text truncation when detokenizer holds back text.

    With stream_interval=10 and SentencePiece holdback, the last word(s)
    before <channel|> can be lost:
    - delta_text is shorter than what delta_token_ids decode to
    - Scanner defers the <channel|> terminal (text not in delta_text)
    - DelegatingParser sees CHANNEL_END_ID in raw token IDs and
      transitions to tool mode prematurely
    - Holdback text ("efficiency.") arrives in the next batch but goes
      to the tool parser instead of the reasoning parser
    """

    @pytest.fixture
    def tokenizer_2(self):
        return _make_mock_tokenizer_2()

    @pytest.fixture
    def parser_2(self, tokenizer_2):
        _WrappedParser.reasoning_parser_cls = GrammarGemma4ReasoningParser
        _WrappedParser.tool_parser_cls = GrammarGemma4ToolParser
        return _WrappedParser(tokenizer_2)

    def test_reasoning_not_truncated(self, parser_2, tokenizer_2, request_obj):
        """Reasoning must include the full text up to <channel|>."""
        results = _stream_tokens_with_holdback(
            parser_2,
            tokenizer_2,
            request_obj,
            batch_size=10,
            holdback_chars=12,
            prompt_token_ids=[],
        )

        reasoning, content, tool_calls = _collect_fields(results)

        assert "efficiency" in reasoning, (
            f"Reasoning truncated — missing 'efficiency'. "
            f"Reasoning ends with: {reasoning[-60:]!r}"
        )

    def test_both_tool_calls_extracted(self, parser_2, tokenizer_2, request_obj):
        """Both bash tool calls must be extracted."""
        results = _stream_tokens_with_holdback(
            parser_2,
            tokenizer_2,
            request_obj,
            batch_size=10,
            holdback_chars=12,
            prompt_token_ids=[],
        )

        _, _, tool_calls = _collect_fields(results)

        names = [
            tc.function.name for tc in tool_calls if tc.function and tc.function.name
        ]
        assert len(names) >= 2, f"Expected 2 tool calls, got {len(names)}: {names}"
        assert names.count("bash") >= 2, f"Expected 2 bash tool calls, got {names}"

    def test_tool_call_text_not_in_content(self, parser_2, tokenizer_2, request_obj):
        """Tool call body must not leak into content."""
        results = _stream_tokens_with_holdback(
            parser_2,
            tokenizer_2,
            request_obj,
            batch_size=10,
            holdback_chars=12,
            prompt_token_ids=[],
        )

        _, content, _ = _collect_fields(results)

        assert "call:" not in content, (
            f"Tool call text leaked into content: {content!r}"
        )
