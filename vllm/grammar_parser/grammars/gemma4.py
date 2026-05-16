# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Grammar configuration for Google Gemma4 tool call format.

Gemma4 format::

    <|tool_call>call:func_name{key:<|"|>value<|"|>,num:42}<tool_call|>

The argument body uses a custom serialization (not JSON): unquoted keys,
``<|"|>`` string delimiters, and bare values for numbers/booleans.
The ``arg_converter`` hooks into the existing ``_parse_gemma4_args``
function to convert this to JSON for the OpenAI protocol.
"""

from __future__ import annotations

import json

from vllm.grammar_parser.events import EventType
from vllm.grammar_parser.grammar_config import (
    GrammarConfig,
    ParserState,
    Transition,
)
from vllm.grammar_parser.grammars.gemma4_tokens import GEMMA4_DROP_TOKENS
from vllm.tool_parsers.gemma4_tool_parser import (
    _parse_gemma4_args,
)

TOOL_CALL_START = "<|tool_call>"
TOOL_CALL_END = "<tool_call|>"


def _gemma4_arg_converter(raw_args: str, partial: bool) -> str:
    """Convert Gemma4 custom arg format to JSON string.

    The raw text is everything between ``{`` and the closing ``}``
    (inclusive of any trailing ``}`` from the format).  We strip the
    trailing ``}`` before parsing.
    """
    text = raw_args.strip()
    if text.endswith("}"):
        text = text[:-1]

    parsed = _parse_gemma4_args(text, partial=partial)
    return json.dumps(parsed, ensure_ascii=False)


def gemma4_config() -> GrammarConfig:
    """Return the grammar config for Gemma4 tool calls."""
    return GrammarConfig(
        name="gemma4",
        terminals={
            "TOOL_START": TOOL_CALL_START,
            "TOOL_END": TOOL_CALL_END,
            "CALL_PREFIX": "call:",
            "OPEN_BRACE": "{",
        },
        token_id_terminals={
            "TOOL_START": TOOL_CALL_START,
            "TOOL_END": TOOL_CALL_END,
        },
        transitions={
            (ParserState.CONTENT, "TOOL_START"): Transition(
                ParserState.TOOL_PREAMBLE,
                [EventType.TOOL_CALL_START],
            ),
            (ParserState.TOOL_PREAMBLE, "CALL_PREFIX"): Transition(
                ParserState.TOOL_NAME,
                [],
            ),
            (ParserState.TOOL_NAME, "OPEN_BRACE"): Transition(
                ParserState.TOOL_ARGS,
                [],
            ),
            (ParserState.TOOL_ARGS, "TOOL_END"): Transition(
                ParserState.CONTENT,
                [EventType.TOOL_CALL_END],
            ),
        },
        content_events={
            ParserState.CONTENT: EventType.TEXT_CHUNK,
            ParserState.TOOL_NAME: EventType.TOOL_NAME,
            ParserState.TOOL_ARGS: EventType.ARG_VALUE_CHUNK,
        },
        arg_converter=_gemma4_arg_converter,
        tool_args_json=False,
        arg_structural_chars=frozenset(",:{}[]<"),
        drop_tokens=GEMMA4_DROP_TOKENS - {"<|tool_call>", "<tool_call|>", '<|"|>'},
    )
