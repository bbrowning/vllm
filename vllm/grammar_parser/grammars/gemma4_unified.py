# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unified grammar configuration for Gemma4: channel-based reasoning
plus custom tool call format in a single state machine.

A single :class:`StreamingParserEngine` handles the complete model
output::

    <|channel>thought
    ...reasoning...<channel|>
    <|tool_call>call:func{key:<|"|>val<|"|>}<tool_call|>
"""

from __future__ import annotations

from vllm.grammar_parser.events import EventType
from vllm.grammar_parser.grammar_config import (
    GrammarConfig,
    ParserState,
    Transition,
)
from vllm.grammar_parser.grammars.gemma4 import _gemma4_arg_converter
from vllm.grammar_parser.grammars.gemma4_tokens import GEMMA4_DROP_TOKENS

CHANNEL_START = "<|channel>"
CHANNEL_END = "<channel|>"
TOOL_CALL_START = "<|tool_call>"
TOOL_CALL_END = "<tool_call|>"


def gemma4_unified_config() -> GrammarConfig:
    """Return a unified grammar config for Gemma4 reasoning + tool calls."""
    used_tokens = {
        CHANNEL_START,
        CHANNEL_END,
        TOOL_CALL_START,
        TOOL_CALL_END,
        '<|"|>',
    }

    return GrammarConfig(
        name="gemma4_unified",
        initial_state=ParserState.CONTENT,
        terminals={
            "THINK_START": CHANNEL_START,
            "THINK_END": CHANNEL_END,
            "TOOL_START": TOOL_CALL_START,
            "TOOL_END": TOOL_CALL_END,
            "CALL_PREFIX": "call:",
            "OPEN_BRACE": "{",
        },
        token_id_terminals={
            "THINK_START": CHANNEL_START,
            "THINK_END": CHANNEL_END,
            "TOOL_START": TOOL_CALL_START,
            "TOOL_END": TOOL_CALL_END,
        },
        transitions={
            # -- Reasoning transitions --
            (ParserState.CONTENT, "THINK_START"): Transition(
                ParserState.REASONING,
                [EventType.REASONING_START],
            ),
            (ParserState.REASONING, "THINK_END"): Transition(
                ParserState.CONTENT,
                [EventType.REASONING_END],
            ),
            # Tool call directly from reasoning (no explicit <channel|>)
            (ParserState.REASONING, "TOOL_START"): Transition(
                ParserState.TOOL_PREAMBLE,
                [EventType.REASONING_END, EventType.TOOL_CALL_START],
            ),
            # -- Tool call transitions --
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
            # Back-to-back tool calls
            (ParserState.CONTENT, "TOOL_END"): Transition(
                ParserState.CONTENT,
                [],
            ),
            # Absorb a bare <channel|> in content state.  This can occur when
            # holdback-released bytes reconstruct the token after a premature
            # THINK_END firing has already transitioned the state machine to
            # CONTENT.  Silently drop it rather than leaking it as TEXT_CHUNK.
            (ParserState.CONTENT, "THINK_END"): Transition(
                ParserState.CONTENT,
                [],
            ),
        },
        content_events={
            ParserState.CONTENT: EventType.TEXT_CHUNK,
            ParserState.REASONING: EventType.REASONING_CHUNK,
            ParserState.TOOL_NAME: EventType.TOOL_NAME,
            ParserState.TOOL_ARGS: EventType.ARG_VALUE_CHUNK,
        },
        arg_converter=_gemma4_arg_converter,
        tool_args_json=False,
        arg_structural_chars=frozenset(",:{}[]<"),
        drop_tokens=GEMMA4_DROP_TOKENS - used_tokens,
        token_id_text_in_delta=True,
    )
