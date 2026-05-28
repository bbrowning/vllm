# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek V4 grammar parser: ``<think>``/``</think>``
reasoning plus DSML tool calls in a single state machine.

DeepSeek V4 output format::

    <think>
    ...reasoning...
    </think>
    <｜DSML｜tool_calls>
    <｜DSML｜invoke name="func_name">
    <｜DSML｜parameter name="location" string="true">杭州</｜DSML｜parameter>
    <｜DSML｜parameter name="count" string="false">5</｜DSML｜parameter>
    </｜DSML｜invoke>
    </｜DSML｜tool_calls>

The ``string`` attribute on each parameter tag controls type coercion:
``string="true"`` keeps the value as a raw string; ``string="false"`` parses
it as JSON (number, boolean, array, or object).

Initial state depends on thinking mode:
- In thinking mode the prompt pre-fills ``<think>``, so the model output
  starts inside a reasoning block.  Initial state is REASONING.
- In chat mode the prompt pre-fills ``</think>`` so the model starts
  outputting content directly.  Initial state is CONTENT.

A bare ``</think>`` in CONTENT state (no preceding ``<think>``) is silently
absorbed so that models which suppress their thinking by emitting ``</think>``
immediately do not leak the tag into text content.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import regex as re

from vllm.parser.grammar.events import EventType
from vllm.parser.grammar.grammar_config import (
    GrammarConfig,
    ParserState,
    Transition,
)
from vllm.parser.grammar.unified_parser import GrammarParser

if TYPE_CHECKING:
    from vllm.tokenizers import TokenizerLike
    from vllm.tool_parsers.abstract_tool_parser import Tool

_DSML = "｜DSML｜"

DSML_THINK_START = "<think>"
DSML_THINK_END = "</think>"
DSML_TOOL_CALLS_START = f"<{_DSML}tool_calls>"
DSML_TOOL_CALLS_END = f"</{_DSML}tool_calls>"
DSML_INVOKE_PREFIX = f'<{_DSML}invoke name="'
DSML_INVOKE_NAME_END = '">'
DSML_INVOKE_END = f"</{_DSML}invoke>"

_ESCAPED_DSML = re.escape(_DSML)
_PARAM_RE = re.compile(
    rf'<{_ESCAPED_DSML}parameter\s+name="([^"]+)"\s+string="(true|false)">(.*?)</{_ESCAPED_DSML}parameter>',
    re.DOTALL,
)
_PARTIAL_PARAM_RE = re.compile(
    rf'<{_ESCAPED_DSML}parameter\s+name="([^"]+)"\s+string="(true|false)">([^<]*)$',
    re.DOTALL,
)


def _dsml_arg_converter(raw_args: str, partial: bool) -> str:
    """Convert DSML parameter tags to a JSON object string.

    Handles both complete (partial=False) and in-progress (partial=True)
    argument bodies captured in the TOOL_ARGS parser state.
    """
    params: dict[str, object] = {}

    for m in _PARAM_RE.finditer(raw_args):
        name, is_str, value = m.group(1), m.group(2), m.group(3)
        if is_str == "true":
            params[name] = value
        else:
            try:
                params[name] = json.loads(value)
            except (json.JSONDecodeError, ValueError):
                params[name] = value

    if partial:
        pm = _PARTIAL_PARAM_RE.search(raw_args)
        if pm:
            name = pm.group(1)
            value = pm.group(3)
            if name:
                params[name] = value

    return json.dumps(params, ensure_ascii=False)


_DEEPSEEK_V4_CONFIGS: dict[bool, GrammarConfig] = {}


def deepseek_v4_config(thinking: bool = False) -> GrammarConfig:
    """Return the grammar config for DeepSeek V4 reasoning + tool calls."""
    cached = _DEEPSEEK_V4_CONFIGS.get(thinking)
    if cached is not None:
        return cached
    config = GrammarConfig(
        name="deepseek_v4",
        initial_state=ParserState.REASONING if thinking else ParserState.CONTENT,
        terminals={
            "THINK_START": DSML_THINK_START,
            "THINK_END": DSML_THINK_END,
            "TOOL_CALLS_START": DSML_TOOL_CALLS_START,
            "TOOL_CALLS_END": DSML_TOOL_CALLS_END,
            "INVOKE_PREFIX": DSML_INVOKE_PREFIX,
            "INVOKE_NAME_END": DSML_INVOKE_NAME_END,
            "INVOKE_END": DSML_INVOKE_END,
        },
        token_id_terminals={
            "THINK_START": DSML_THINK_START,
            "THINK_END": DSML_THINK_END,
        },
        transitions={
            # -- Reasoning transitions --
            (ParserState.CONTENT, "THINK_START"): Transition(
                ParserState.REASONING,
                [EventType.REASONING_START],
            ),
            # Absorb a bare </think> with no prior <think> (model skips reasoning)
            (ParserState.CONTENT, "THINK_END"): Transition(
                ParserState.CONTENT,
                [],
            ),
            # Absorb a duplicate <think> while already reasoning
            (ParserState.REASONING, "THINK_START"): Transition(
                ParserState.REASONING,
                [],
            ),
            (ParserState.REASONING, "THINK_END"): Transition(
                ParserState.CONTENT,
                [EventType.REASONING_END],
            ),
            # Tool call beginning while still inside <think> (implicit end)
            (ParserState.REASONING, "TOOL_CALLS_START"): Transition(
                ParserState.TOOL_PREAMBLE,
                [EventType.REASONING_END],
            ),
            # -- Tool call transitions --
            (ParserState.CONTENT, "TOOL_CALLS_START"): Transition(
                ParserState.TOOL_PREAMBLE,
                [],
            ),
            (ParserState.TOOL_PREAMBLE, "INVOKE_PREFIX"): Transition(
                ParserState.TOOL_NAME,
                [EventType.TOOL_CALL_START],
            ),
            (ParserState.TOOL_NAME, "INVOKE_NAME_END"): Transition(
                ParserState.TOOL_ARGS,
                [],
            ),
            (ParserState.TOOL_ARGS, "INVOKE_END"): Transition(
                ParserState.TOOL_BETWEEN,
                [EventType.TOOL_CALL_END],
            ),
            # Parallel tool calls: another invoke after the previous one ends
            (ParserState.TOOL_BETWEEN, "INVOKE_PREFIX"): Transition(
                ParserState.TOOL_NAME,
                [EventType.TOOL_CALL_START],
            ),
            (ParserState.TOOL_BETWEEN, "TOOL_CALLS_END"): Transition(
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
        arg_converter=_dsml_arg_converter,
        strip_trailing_quotes=False,
        tool_args_json=False,
    )
    _DEEPSEEK_V4_CONFIGS[thinking] = config
    return config


class DeepSeekV4GrammarParser(GrammarParser):
    """DeepSeek V4 parser: ``<think>``/``</think>`` reasoning +
    DSML tool calls (``<｜DSML｜tool_calls>``/``<｜DSML｜invoke>``) in a
    single state machine.

    Initial state is REASONING in thinking mode (the prompt pre-fills
    ``<think>``); CONTENT otherwise (chat mode pre-fills ``</think>``).
    """

    def __init__(
        self,
        tokenizer: TokenizerLike,
        tools: list[Tool] | None = None,
        **kwargs,
    ) -> None:
        chat_kwargs = kwargs.pop("chat_template_kwargs", None) or {}
        thinking = bool(
            chat_kwargs.get("thinking") or chat_kwargs.get("enable_thinking")
        )
        super().__init__(
            tokenizer,
            tools,
            grammar_config=deepseek_v4_config(thinking=thinking),
            **kwargs,
        )
