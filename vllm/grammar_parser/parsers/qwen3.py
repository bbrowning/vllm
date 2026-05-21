# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Qwen3 grammar parsers for tool calls and reasoning.

Two parsers are provided:

* ``Qwen3XMLGrammarParser`` — tool calls only (XML ``<tool_call>`` format)
* ``Qwen3GrammarParser`` — reasoning (``<think>``/``</think>``) **plus**
  tool calls in a single state machine.  Starts in REASONING state
  because Qwen3.5+ chat templates place ``<think>`` in the prompt.

Qwen3 XML tool call format::

    <tool_call>
    <function=func_name>
    <parameter=key>value</parameter>
    </function>
    </tool_call>

The argument body consists of ``<parameter=NAME>VALUE</parameter>`` tags.
The ``_qwen3xml_arg_converter`` parses these into a JSON object.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import regex as re

from vllm.grammar_parser.events import EventType
from vllm.grammar_parser.grammar_config import (
    GrammarConfig,
    ParserState,
    Transition,
)
from vllm.grammar_parser.unified_parser import GrammarParser
from vllm.tool_parsers.utils import safe_literal_eval

if TYPE_CHECKING:
    from vllm.tokenizers import TokenizerLike
    from vllm.tool_parsers.abstract_tool_parser import Tool

TOOL_CALL_START = "<tool_call>"
TOOL_CALL_END = "</tool_call>"
FUNC_PREFIX = "<function="
FUNC_END = "</function>"

_PARAM_RE = re.compile(
    r"<\s*parameter\s*=\s*([^>]*)>(.*?)<\s*/\s*parameter\s*>", re.DOTALL
)
_PARTIAL_PARAM_RE = re.compile(r"<\s*parameter\s*=\s*([^>]+)>([^<]*)$", re.DOTALL)


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
        result = safe_literal_eval(stripped)
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
    """Return the grammar config for Qwen3 XML tool calls (no reasoning)."""
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


def qwen3_config() -> GrammarConfig:
    """Return the grammar config for Qwen3 reasoning + tool calls."""
    return GrammarConfig(
        name="qwen3",
        initial_state=ParserState.REASONING,
        terminals={
            # Reasoning terminals
            "THINK_START": "<think>",
            "THINK_END": "</think>",
            # Tool call terminals
            "TOOL_START": TOOL_CALL_START,
            "TOOL_END": TOOL_CALL_END,
            "FUNC_PREFIX": FUNC_PREFIX,
            "FUNC_END": FUNC_END,
            "CLOSE_ANGLE": ">",
        },
        token_id_terminals={
            "THINK_START": "<think>",
            "THINK_END": "</think>",
            "TOOL_START": TOOL_CALL_START,
            "TOOL_END": TOOL_CALL_END,
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


class Qwen3GrammarParser(GrammarParser):
    """Qwen3 parser: ``<think>``/``</think>`` reasoning +
    ``<tool_call>`` XML tool calls in a single engine.

    - Starts in REASONING state (Qwen3.5+ puts ``<think>`` in prompt)
    - ``<tool_call>`` as implicit reasoning end
    - Unpaired ``<tool_call>`` token ID detection for ``is_reasoning_end``
    """

    def __init__(
        self,
        tokenizer: TokenizerLike,
        tools: list[Tool] | None = None,
        **kwargs,
    ) -> None:
        kwargs.setdefault("grammar_config", qwen3_config())
        super().__init__(
            tokenizer,
            tools,
            **kwargs,
        )
        vocab = self.vocab
        self._tool_call_token_id: int | None = vocab.get("<tool_call>")
        self._tool_call_end_token_id: int | None = vocab.get("</tool_call>")

    def is_reasoning_end(self, input_ids: list[int]) -> bool:
        if super().is_reasoning_end(input_ids):
            return True
        tool_call_id = self._tool_call_token_id
        tool_call_end_id = self._tool_call_end_token_id
        if tool_call_id is not None:
            for i in range(len(input_ids) - 1, -1, -1):
                if input_ids[i] == tool_call_id:
                    if tool_call_end_id is not None and any(
                        input_ids[j] == tool_call_end_id
                        for j in range(i + 1, len(input_ids))
                    ):
                        continue
                    return True
        return False


class Qwen3XMLGrammarParser(GrammarParser):
    """Qwen3 XML parser: ``<tool_call><function=...>`` tool calls."""

    def __init__(
        self,
        tokenizer: TokenizerLike,
        tools: list[Tool] | None = None,
        **kwargs,
    ) -> None:
        super().__init__(tokenizer, tools, grammar_config=qwen3xml_config(), **kwargs)
