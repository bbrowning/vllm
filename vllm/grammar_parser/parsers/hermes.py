# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Hermes grammar parser for JSON tool call formats.

Hermes format::

    <tool_call>{"name": "func", "arguments": {"key": "val"}}</tool_call>

The JSON body contains both the function name and arguments.
The ``arg_converter`` extracts just the arguments portion for
streaming, while ``_try_extract_name`` in the adapter handles
the function name.

This config also works for other tag-delimited JSON body formats
by parameterizing the start/end tags.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from vllm.grammar_parser.events import EventType
from vllm.grammar_parser.grammar_config import (
    GrammarConfig,
    ParserState,
    Transition,
)
from vllm.grammar_parser.unified_parser import GrammarParser

if TYPE_CHECKING:
    from vllm.tokenizers import TokenizerLike
    from vllm.tool_parsers.abstract_tool_parser import Tool


def _json_body_arg_converter(raw_args: str, partial: bool) -> str:
    """Extract arguments from a JSON body like
    ``{"name": "func", "arguments": {...}}``."""
    raw_args = raw_args.strip()
    if not raw_args:
        return "{}"

    try:
        import partial_json_parser
        from partial_json_parser.core.options import Allow

        parsed = partial_json_parser.loads(raw_args, Allow.ALL)
    except Exception:
        return "{}"

    if not isinstance(parsed, dict):
        return "{}"

    for key in ("arguments", "parameters"):
        if key in parsed:
            val = parsed[key]
            if isinstance(val, str):
                return val
            return json.dumps(val, ensure_ascii=False)

    return "{}"


def hermes_config(
    start_tag: str = "<tool_call>",
    end_tag: str = "</tool_call>",
    name: str = "hermes",
) -> GrammarConfig:
    """Return a grammar config for Hermes-style JSON tool calls.

    Args:
        start_tag: The tag that opens a tool call block.
        end_tag: The tag that closes a tool call block.
        name: Name for this grammar config instance.
    """
    return GrammarConfig(
        name=name,
        terminals={
            "TOOL_START": start_tag,
            "TOOL_END": end_tag,
        },
        token_id_terminals={
            "TOOL_START": start_tag,
            "TOOL_END": end_tag,
        },
        transitions={
            (ParserState.CONTENT, "TOOL_START"): Transition(
                ParserState.TOOL_ARGS,
                [EventType.TOOL_CALL_START],
            ),
            (ParserState.TOOL_ARGS, "TOOL_END"): Transition(
                ParserState.CONTENT,
                [EventType.TOOL_CALL_END],
            ),
        },
        content_events={
            ParserState.CONTENT: EventType.TEXT_CHUNK,
            ParserState.TOOL_ARGS: EventType.ARG_VALUE_CHUNK,
        },
        arg_converter=_json_body_arg_converter,
        tool_args_json=False,
    )


class HermesGrammarParser(GrammarParser):
    """Hermes parser: ``<tool_call>``/``</tool_call>`` JSON tool calls."""

    def __init__(
        self,
        tokenizer: TokenizerLike,
        tools: list[Tool] | None = None,
        **kwargs,
    ) -> None:
        super().__init__(tokenizer, tools, grammar_config=hermes_config(), **kwargs)
