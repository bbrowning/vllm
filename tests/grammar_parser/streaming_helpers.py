# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared streaming simulation helpers for grammar parser tests."""

from __future__ import annotations

from typing import Any

from vllm.entrypoints.openai.engine.protocol import DeltaMessage


def simulate_tool_streaming(
    parser,
    request,
    chunks: list[str],
) -> list[tuple[DeltaMessage | None, str]]:
    """Feed text chunks through ``extract_tool_calls_streaming()``.

    Uses dummy token IDs (``[0]`` per chunk).  For tests that need
    explicit token IDs, use :func:`simulate_tool_streaming_with_ids`.
    """
    results: list[tuple[Any, str]] = []
    previous_text = ""
    previous_token_ids: list[int] = []

    for chunk in chunks:
        current_text = previous_text + chunk
        delta_token_ids: list[int] = [0]
        current_token_ids = previous_token_ids + delta_token_ids

        delta = parser.extract_tool_calls_streaming(
            previous_text=previous_text,
            current_text=current_text,
            delta_text=chunk,
            previous_token_ids=tuple(previous_token_ids),
            current_token_ids=tuple(current_token_ids),
            delta_token_ids=tuple(delta_token_ids),
            request=request,
        )
        results.append((delta, current_text))
        previous_text = current_text
        previous_token_ids = list(current_token_ids)

    return results


def simulate_tool_streaming_with_ids(
    parser,
    request,
    deltas: list[tuple[str, list[int]]],
) -> list[tuple[DeltaMessage | None, str]]:
    """Feed ``(delta_text, delta_token_ids)`` pairs through streaming."""
    results: list[tuple[Any, str]] = []
    previous_text = ""
    previous_token_ids: list[int] = []

    for delta_text, delta_tids in deltas:
        current_text = previous_text + delta_text
        current_token_ids = previous_token_ids + delta_tids

        delta = parser.extract_tool_calls_streaming(
            previous_text=previous_text,
            current_text=current_text,
            delta_text=delta_text,
            previous_token_ids=tuple(previous_token_ids),
            current_token_ids=tuple(current_token_ids),
            delta_token_ids=tuple(delta_tids),
            request=request,
        )
        results.append((delta, current_text))
        previous_text = current_text
        previous_token_ids = list(current_token_ids)

    return results


def collect_tool_arguments(
    results: list[tuple[DeltaMessage | None, str]],
) -> str:
    """Concatenate all streamed argument fragments."""
    args_text = ""
    for delta, _ in results:
        if delta and delta.tool_calls:
            for tc in delta.tool_calls:
                if tc.function and tc.function.arguments:
                    args_text += tc.function.arguments
    return args_text


def collect_function_name(
    results: list[tuple[DeltaMessage | None, str]],
) -> str | None:
    """Return first function name from deltas."""
    for delta, _ in results:
        if delta and delta.tool_calls:
            for tc in delta.tool_calls:
                if tc.function and tc.function.name:
                    return tc.function.name
    return None


def simulate_reasoning_streaming(
    parser,
    chunks: list[str],
    delta_token_ids_per_chunk: list[tuple[int, ...]] | None = None,
) -> tuple[str, str]:
    """Feed chunks through ``extract_reasoning_streaming()``.

    Returns ``(reasoning_text, content_text)`` tuple.
    """
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
