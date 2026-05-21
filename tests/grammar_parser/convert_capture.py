#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Convert raw parser token captures to replay test fixtures.

Reads JSONL captured via ``VLLM_DUMP_PARSER_TOKENS`` and produces
fixture JSONL suitable for ``tests/grammar_parser/data/``.

The conversion uses token IDs (not the incremental lexer) to partition
the stream, so it serves as an independent oracle for what the parser
should produce.

Usage::

    python tests/grammar_parser/convert_capture.py \\
        --input qwen36_capture.jsonl \\
        --start-number 6 \\
        --prefix qwen3-live
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import regex as re

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from vllm.grammar_parser.parsers.qwen3 import _qwen3xml_arg_converter

_FUNC_NAME_RE = re.compile(r"<function=([^>]+)>")


@dataclass
class _ToolCall:
    name: str = ""
    raw_args: str = ""


@dataclass
class _ParsedEntry:
    reasoning: str = ""
    content: str = ""
    tool_calls: list[_ToolCall] = field(default_factory=list)


def _partition_by_token_ids(
    tokens: list[list],
    vocab: dict[str, int],
) -> _ParsedEntry:
    """Partition a token stream using special token IDs."""
    think_start_id = vocab.get("<think>")
    think_end_id = vocab.get("</think>")
    tool_start_id = vocab.get("<tool_call>")
    tool_end_id = vocab.get("</tool_call>")

    REASONING = "reasoning"
    CONTENT = "content"
    TOOL_BODY = "tool_body"

    state = REASONING
    result = _ParsedEntry()
    current_tool_body = ""

    for tid, text in tokens:
        if tid == think_start_id:
            continue
        elif tid == think_end_id:
            state = CONTENT
            continue
        elif tid == tool_start_id:
            state = TOOL_BODY
            current_tool_body = ""
            continue
        elif tid == tool_end_id:
            if current_tool_body:
                _finalize_tool_call(result, current_tool_body)
            current_tool_body = ""
            state = CONTENT
            continue

        if state == REASONING:
            result.reasoning += text
        elif state == CONTENT:
            result.content += text
        elif state == TOOL_BODY:
            current_tool_body += text

    return result


def _finalize_tool_call(result: _ParsedEntry, raw_body: str) -> None:
    """Extract function name and arguments from raw tool call body."""
    m = _FUNC_NAME_RE.search(raw_body)
    if not m:
        return

    func_name = m.group(1)
    close_angle_pos = m.end()

    func_end_marker = "</function>"
    func_end_pos = raw_body.rfind(func_end_marker)
    if func_end_pos == -1:
        raw_args = raw_body[close_angle_pos:]
    else:
        raw_args = raw_body[close_angle_pos:func_end_pos]

    tc = _ToolCall(name=func_name, raw_args=raw_args)
    result.tool_calls.append(tc)


def _strip_stop_tokens(
    tokens: list[list],
    stop_token_ids: set[int] | None = None,
) -> list[list]:
    """Remove stop tokens (e.g. <|im_end|>) from the end of token list."""
    if stop_token_ids is None:
        stop_token_ids = {248046}  # <|im_end|> for Qwen3

    while tokens and tokens[-1][0] in stop_token_ids:
        tokens = tokens[:-1]
    return tokens


_DESCRIPTIONS = [
    "Live: reasoning then content text then Bash tool call (2 params)",
    "Live: brief reasoning then markdown content (no tools)",
    "Live: reasoning then Read tool call (1 param, whitespace content)",
    "Live: reasoning then content explanation (no tools)",
    "Live: reasoning then Edit tool call (3 params, large code blocks)",
    "Live: reasoning then content text then Bash verify (2 params)",
    "Live: brief reasoning then done content message (no tools)",
]

_SUFFIXES = [
    "reasoning-content-bash",
    "content-only",
    "reasoning-read",
    "reasoning-content",
    "edit-tool",
    "content-bash-verify",
    "done-content",
]


def convert_entry(
    data: dict,
    entry_index: int,
    start_number: int,
    prefix: str,
) -> dict:
    """Convert a single captured entry to a test fixture."""
    tokens = _strip_stop_tokens(data["tokens"])
    vocab = data["vocab"]

    num = start_number + entry_index
    suffix = _SUFFIXES[entry_index] if entry_index < len(_SUFFIXES) else "entry"
    fixture_id = f"{prefix}-{suffix}-{num:03d}"
    description = (
        _DESCRIPTIONS[entry_index]
        if entry_index < len(_DESCRIPTIONS)
        else data.get("description", "")
    )

    if "parsed" in data and data["parsed"] is not None:
        pr = data["parsed"]
        expected: dict = {
            "reasoning": pr.get("reasoning"),
            "content": pr.get("content"),
            "tool_calls": pr.get("tool_calls", []),
        }
    else:
        parsed = _partition_by_token_ids(tokens, vocab)
        expected_tool_calls = []
        for tc in parsed.tool_calls:
            args_json = _qwen3xml_arg_converter(tc.raw_args, partial=False)
            args_dict = json.loads(args_json)
            expected_tool_calls.append({"name": tc.name, "arguments": args_dict})
        expected = {
            "reasoning": parsed.reasoning or None,
            "content": parsed.content or None,
            "tool_calls": expected_tool_calls if expected_tool_calls else [],
        }

    return {
        "id": fixture_id,
        "description": description,
        "source": data.get("source", "live-capture"),
        "vocab": vocab,
        "tokens": tokens,
        "expected": expected,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Path to capture JSONL")
    parser.add_argument(
        "--start-number",
        type=int,
        default=6,
        help="Starting sequence number for IDs",
    )
    parser.add_argument("--prefix", default="qwen3-live", help="ID prefix")
    parser.add_argument("--output", help="Output file (default: stdout)")
    args = parser.parse_args()

    entries = []
    for line in Path(args.input).read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        entries.append(json.loads(line))

    if args.output:
        with open(args.output, "w") as out:
            for i, entry in enumerate(entries):
                fixture = convert_entry(entry, i, args.start_number, args.prefix)
                out.write(json.dumps(fixture, ensure_ascii=False) + "\n")
    else:
        for i, entry in enumerate(entries):
            fixture = convert_entry(entry, i, args.start_number, args.prefix)
            sys.stdout.write(json.dumps(fixture, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
