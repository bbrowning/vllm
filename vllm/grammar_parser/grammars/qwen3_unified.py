# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unified grammar configuration for Qwen3: ``<think>``/``</think>``
reasoning plus ``<tool_call>`` XML tool calls in a single state machine.

Merges ``_qwen3_reasoning_config()`` (reasoning) and ``qwen3xml_config()``
(tool calls).  Starts in REASONING state because Qwen3.5+ chat
templates place ``<think>`` in the prompt.
"""

from __future__ import annotations

from vllm.grammar_parser.events import EventType
from vllm.grammar_parser.grammar_config import (
    GrammarConfig,
    ParserState,
    Transition,
)
from vllm.grammar_parser.grammars.qwen3xml import _qwen3xml_arg_converter


def qwen3_unified_config() -> GrammarConfig:
    """Return a unified grammar config for Qwen3 reasoning + tool calls."""
    return GrammarConfig(
        name="qwen3_unified",
        initial_state=ParserState.REASONING,
        terminals={
            # Reasoning terminals
            "THINK_START": "<think>",
            "THINK_END": "</think>",
            # Tool call terminals
            "TOOL_START": "<tool_call>",
            "TOOL_END": "</tool_call>",
            "FUNC_PREFIX": "<function=",
            "FUNC_END": "</function>",
            "CLOSE_ANGLE": ">",
        },
        token_id_terminals={
            "THINK_START": "<think>",
            "THINK_END": "</think>",
            "TOOL_START": "<tool_call>",
            "TOOL_END": "</tool_call>",
        },
        transitions={
            # -- Reasoning transitions --
            (ParserState.REASONING, "THINK_START"): Transition(
                ParserState.REASONING,
                [],
            ),
            (ParserState.REASONING, "THINK_END"): Transition(
                ParserState.CONTENT,
                [EventType.REASONING_END],
            ),
            # Tool call directly from reasoning (implicit end)
            (ParserState.REASONING, "TOOL_START"): Transition(
                ParserState.TOOL_PREAMBLE,
                [EventType.REASONING_END, EventType.TOOL_CALL_START],
            ),
            # -- Tool call transitions --
            (ParserState.CONTENT, "TOOL_START"): Transition(
                ParserState.TOOL_PREAMBLE,
                [EventType.TOOL_CALL_START],
            ),
            (ParserState.TOOL_PREAMBLE, "FUNC_PREFIX"): Transition(
                ParserState.TOOL_NAME,
                [],
            ),
            (ParserState.TOOL_NAME, "CLOSE_ANGLE"): Transition(
                ParserState.TOOL_ARGS,
                [],
            ),
            (ParserState.TOOL_ARGS, "FUNC_END"): Transition(
                ParserState.TOOL_BETWEEN,
                [EventType.TOOL_CALL_END],
            ),
            (ParserState.TOOL_BETWEEN, "TOOL_END"): Transition(
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
        arg_converter=_qwen3xml_arg_converter,
        strip_trailing_quotes=False,
        tool_args_json=False,
    )
