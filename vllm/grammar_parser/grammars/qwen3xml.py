# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Grammar configuration for Qwen3 XML tool call format.

Qwen3 XML format::

    <tool_call>
    <function=func_name>
    <parameter=key>value</parameter>
    </function>
    </tool_call>

The argument body consists of ``<parameter=NAME>VALUE</parameter>`` tags.
The ``arg_converter`` parses these into a JSON object.
"""

from __future__ import annotations

import ast
import json

import regex as re

from vllm.grammar_parser.events import EventType
from vllm.grammar_parser.grammar_config import (
    GrammarConfig,
    ParserState,
    Transition,
)

TOOL_CALL_START = "<tool_call>"
TOOL_CALL_END = "</tool_call>"
FUNC_PREFIX = "<function="
FUNC_END = "</function>"

_PARAM_RE = re.compile(r"<parameter=([^>]*)>(.*?)</parameter>", re.DOTALL)
_PARTIAL_PARAM_RE = re.compile(r"<parameter=([^>]+)>([^<]*)$", re.DOTALL)


def _coerce_value(text: str):
    """Best-effort type coercion for parameter values."""
    stripped = text.strip()
    if not stripped:
        return ""

    lower = stripped.lower()
    if lower == "true":
        return True
    if lower == "false":
        return False
    if lower in ("null", "none", "nil"):
        return None

    try:
        return int(stripped)
    except ValueError:
        pass
    try:
        return float(stripped)
    except ValueError:
        pass

    try:
        return json.loads(stripped)
    except (json.JSONDecodeError, ValueError):
        pass

    try:
        result = ast.literal_eval(stripped)
        return result
    except (ValueError, SyntaxError):
        pass

    return stripped


def _qwen3xml_arg_converter(raw_args: str, partial: bool) -> str:
    """Convert Qwen3 XML parameter tags to a JSON string."""
    params: dict[str, object] = {}

    for match in _PARAM_RE.finditer(raw_args):
        name = match.group(1)
        value = match.group(2)
        params[name] = _coerce_value(value)

    if partial:
        remaining = _PARAM_RE.sub("", raw_args)
        m = _PARTIAL_PARAM_RE.search(remaining)
        if m:
            name = m.group(1)
            value = m.group(2)
            if name:
                params[name] = value

    return json.dumps(params, ensure_ascii=False)


def qwen3xml_config() -> GrammarConfig:
    """Return the grammar config for Qwen3 XML tool calls."""
    return GrammarConfig(
        name="qwen3xml",
        terminals={
            "TOOL_START": TOOL_CALL_START,
            "TOOL_END": TOOL_CALL_END,
            "FUNC_PREFIX": FUNC_PREFIX,
            "FUNC_END": FUNC_END,
            "CLOSE_ANGLE": ">",
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
            ParserState.TOOL_NAME: EventType.TOOL_NAME,
            ParserState.TOOL_ARGS: EventType.ARG_VALUE_CHUNK,
        },
        arg_converter=_qwen3xml_arg_converter,
        strip_trailing_quotes=False,
        tool_args_json=False,
    )
