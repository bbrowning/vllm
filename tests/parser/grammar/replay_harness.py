# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Data-driven replay harness for grammar parser testing.

Loads token sequences from JSONL files and replays them through parsers
at different chunk sizes to verify chunk-size invariance: the same
token sequence must produce identical output regardless of how tokens
are batched.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from unittest.mock import MagicMock

from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
)
from vllm.entrypoints.openai.engine.protocol import DeltaMessage

DATA_DIR = Path(__file__).parent / "data"


@dataclass
class Sample:
    """One test sample loaded from a JSONL file."""

    id: str
    description: str
    source: str
    vocab: dict[str, int]
    tokens: list[tuple[int, str]]
    expected_reasoning: str | None
    expected_content: str | None
    expected_tool_calls: list[dict] | None
    tools: list[dict] | None = None


@dataclass
class ParseOutput:
    """Accumulated parse output from replaying a token stream."""

    reasoning: str = ""
    content: str = ""
    tool_calls: list[dict] = field(default_factory=list)


def load_samples_from_path(path: Path) -> list[Sample]:
    """Load all samples from a JSONL file."""
    if not path.exists():
        return []

    samples = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        data = json.loads(line)
        tokens = [(t[0], t[1]) for t in data["tokens"]]
        expected = data.get("expected", {})
        serving = data.get("serving", {})
        samples.append(
            Sample(
                id=data["id"],
                description=data.get("description", ""),
                source=data.get("source", ""),
                vocab=data.get("vocab", {}),
                tokens=tokens,
                expected_reasoning=expected.get("reasoning"),
                expected_content=expected.get("content"),
                expected_tool_calls=expected.get("tool_calls"),
                tools=serving.get("tools"),
            )
        )
    return samples


def load_samples(model: str) -> list[Sample]:
    """Load all samples from ``tests/parser/grammar/data/{model}.jsonl``."""
    return load_samples_from_path(DATA_DIR / f"{model}.jsonl")


def make_mock_tokenizer(sample: Sample) -> MagicMock:
    """Build a mock tokenizer from a sample's vocab and token data."""
    token_decode_map: dict[int, str] = {}
    for tid, text in sample.tokens:
        token_decode_map[tid] = text

    special_text_to_id = dict(sample.vocab)

    tokenizer = MagicMock()
    tokenizer.get_vocab.return_value = dict(special_text_to_id)
    tokenizer.encode.return_value = [tid for tid, _ in sample.tokens]

    def decode(ids, skip_special_tokens=False):
        parts = []
        for tid in ids:
            if skip_special_tokens and tid in _inv_special(special_text_to_id):
                continue
            text = token_decode_map.get(tid, f"?{tid}?")
            parts.append(text)
        return "".join(parts)

    tokenizer.decode.side_effect = decode
    return tokenizer


def _inv_special(vocab: dict[str, int]) -> set[int]:
    return set(vocab.values())


def _test_request() -> ChatCompletionRequest:
    return ChatCompletionRequest(
        model="test-model",
        messages=[{"role": "user", "content": "test"}],
    )


def replay_streaming(
    parser,
    tokens: list[tuple[int, str]],
    chunk_size: int | None = None,
    holdback_chars: int = 0,
    finished_on_last: bool = False,
) -> list[DeltaMessage | None]:
    """Feed tokens through ``parser.parse_delta()`` at a given chunk size.

    Args:
        parser: A :class:`Parser` instance with ``parse_delta()`` method.
        tokens: List of ``(token_id, decoded_text)`` pairs.
        chunk_size: Number of tokens per batch. ``None`` means all at once.
        holdback_chars: Simulate detokenizer holdback by holding back
            this many characters of decoded text between batches.
        finished_on_last: When True, pass ``finished=True`` on the last
            ``parse_delta()`` call, matching real server behavior.

    Returns:
        List of ``DeltaMessage`` results from each ``parse_delta()`` call.
    """
    if chunk_size is None:
        chunk_size = len(tokens)

    results: list[DeltaMessage | None] = []
    all_ids = [tid for tid, _ in tokens]
    all_texts = [text for _, text in tokens]

    request = _test_request()

    if holdback_chars <= 0:
        chunks = list(range(0, len(tokens), chunk_size))
        for i, start in enumerate(chunks):
            batch_end = min(start + chunk_size, len(tokens))
            batch_ids = all_ids[start:batch_end]
            delta_text = "".join(all_texts[start:batch_end])
            is_last = i == len(chunks) - 1

            result = parser.parse_delta(
                delta_text,
                batch_ids,
                request,
                prompt_token_ids=[] if start == 0 else None,
                finished=finished_on_last and is_last,
            )
            results.append(result)
        return results

    emitted_up_to = 0
    is_first = True

    for start in range(0, len(tokens), chunk_size):
        batch_end = min(start + chunk_size, len(tokens))

        if batch_end < len(tokens):
            held_chars = 0
            safe_end = batch_end
            while safe_end > emitted_up_to and held_chars < holdback_chars:
                safe_end -= 1
                held_chars += len(all_texts[safe_end])
        else:
            safe_end = batch_end

        if safe_end <= emitted_up_to:
            continue

        batch_ids = all_ids[emitted_up_to:safe_end]
        delta_text = "".join(all_texts[emitted_up_to:safe_end])
        emitted_up_to = safe_end

        is_last_chunk = batch_end >= len(tokens)
        result = parser.parse_delta(
            delta_text,
            batch_ids,
            request,
            prompt_token_ids=[] if is_first else None,
            finished=finished_on_last and is_last_chunk,
        )
        results.append(result)
        is_first = False

    if emitted_up_to < len(tokens):
        batch_ids = all_ids[emitted_up_to:]
        delta_text = "".join(all_texts[emitted_up_to:])
        result = parser.parse_delta(
            delta_text,
            batch_ids,
            request,
            prompt_token_ids=[] if is_first else None,
            finished=finished_on_last,
        )
        results.append(result)

    return results


def replay_with_text_holdback(
    parser,
    tokens: list[tuple[int, str]],
    text_delay: int = 1,
) -> list[DeltaMessage | None]:
    """Replay token-by-token with text arriving *text_delay* steps late.

    Simulates the production detokenizer holdback where token IDs arrive
    immediately but decoded text is delayed.  On the last token all
    remaining held-back text is flushed, matching real server behavior::

        step 0:   ids=[tok0], text=""               (held back)
        step 1:   ids=[tok1], text=tok0_text         (tok0 released)
        ...
        step N-1: ids=[tokN-1], text=remaining_texts (flush all)

    This exercises the TokenIDScanner deferred-terminal path that
    ``replay_streaming`` (which keeps text and IDs aligned) does not.
    """
    results: list[DeltaMessage | None] = []
    request = _test_request()

    n = len(tokens)
    held_texts: list[str] = []

    for i in range(n):
        token_id = tokens[i][0]
        held_texts.append(tokens[i][1])

        is_last = i == n - 1
        if is_last:
            delta_text = "".join(held_texts)
            held_texts.clear()
        elif len(held_texts) > text_delay:
            delta_text = held_texts.pop(0)
        else:
            delta_text = ""

        result = parser.parse_delta(
            delta_text,
            [token_id],
            request,
            prompt_token_ids=[] if i == 0 else None,
            finished=is_last,
        )
        results.append(result)

    return results


def collect_output(results: list[DeltaMessage | None]) -> ParseOutput:
    """Accumulate ``DeltaMessage`` results into a :class:`ParseOutput`."""
    output = ParseOutput()

    for r in results:
        if r is None:
            continue
        if r.reasoning:
            output.reasoning += r.reasoning
        if r.content:
            output.content += r.content
        if r.tool_calls:
            for tc in r.tool_calls:
                if tc.function and tc.function.name:
                    existing = None
                    for existing_tc in output.tool_calls:
                        if existing_tc.get("_index") == tc.index:
                            existing = existing_tc
                            break

                    if existing is None:
                        output.tool_calls.append(
                            {
                                "_index": tc.index,
                                "name": tc.function.name,
                                "arguments": tc.function.arguments or "",
                            }
                        )
                    else:
                        existing["arguments"] += tc.function.arguments or ""
                elif tc.function and tc.function.arguments:
                    for existing_tc in output.tool_calls:
                        if existing_tc.get("_index") == tc.index:
                            existing_tc["arguments"] += tc.function.arguments
                            break

    for tc in output.tool_calls:
        tc.pop("_index", None)

    return output


def assert_parse_output(actual: ParseOutput, sample: Sample) -> None:
    """Compare actual parse output against expected values from a sample."""
    if sample.expected_reasoning is not None:
        assert actual.reasoning == sample.expected_reasoning, (
            f"Reasoning mismatch:\n"
            f"  expected: {sample.expected_reasoning!r}\n"
            f"  actual:   {actual.reasoning!r}"
        )

    if sample.expected_content is not None:
        assert actual.content == sample.expected_content, (
            f"Content mismatch:\n"
            f"  expected: {sample.expected_content!r}\n"
            f"  actual:   {actual.content!r}"
        )
    if sample.expected_tool_calls is not None:
        assert len(actual.tool_calls) == len(sample.expected_tool_calls), (
            f"Tool call count mismatch: "
            f"expected {len(sample.expected_tool_calls)}, "
            f"got {len(actual.tool_calls)}"
        )
        for i, (expected_tc, actual_tc) in enumerate(
            zip(sample.expected_tool_calls, actual.tool_calls)
        ):
            assert actual_tc["name"] == expected_tc["name"], (
                f"Tool call {i} name mismatch: "
                f"expected {expected_tc['name']!r}, "
                f"got {actual_tc['name']!r}"
            )
            if "arguments" in expected_tc:
                expected_args = expected_tc["arguments"]
                actual_args_str = actual_tc.get("arguments", "{}")
                if isinstance(expected_args, dict):
                    try:
                        actual_args = json.loads(actual_args_str)
                    except json.JSONDecodeError as e:
                        raise AssertionError(
                            f"Tool call {i} arguments not valid JSON: "
                            f"{actual_args_str!r}"
                        ) from e
                    assert actual_args == expected_args, (
                        f"Tool call {i} arguments mismatch:\n"
                        f"  expected: {expected_args}\n"
                        f"  actual:   {actual_args}"
                    )


def assert_no_terminal_leakage(
    actual: ParseOutput,
    terminals: list[str],
    context: str = "",
) -> None:
    """Assert that none of *terminals* appear in reasoning or content."""
    suffix = f" ({context})" if context else ""
    for terminal in terminals:
        assert terminal not in actual.reasoning, (
            f"{terminal!r} leaked into reasoning{suffix}"
        )
        assert terminal not in actual.content, (
            f"{terminal!r} leaked into content{suffix}"
        )
