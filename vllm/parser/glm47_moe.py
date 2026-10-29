# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GLM-4.7 parser for reasoning and tool calls.

GLM-4.7 uses XML-like tool calls::

    <tool_call>func_name<arg_key>key</arg_key><arg_value>value</arg_value></tool_call>

The function name can be followed directly by the first ``<arg_key>`` tag,
and tool calls may have no arguments.
"""

from __future__ import annotations

import functools
import json
from typing import TYPE_CHECKING

import regex as re

from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
from vllm.logger import init_logger
from vllm.parser.engine.events import EventType
from vllm.parser.engine.parser_engine import ParserEngine
from vllm.parser.engine.parser_engine_config import (
    ArgValueExtender,
    ParserEngineConfig,
    ParserState,
    Transition,
)

logger = init_logger(__name__)

if TYPE_CHECKING:
    from vllm.tokenizers import TokenizerLike
    from vllm.tool_parsers.abstract_tool_parser import Tool

THINK_START = "<think>"
THINK_END = "</think>"
TOOL_CALL_START = "<tool_call>"
TOOL_CALL_END = "</tool_call>"
ARG_KEY_START = "<arg_key>"
ARG_KEY_END = "</arg_key>"
ARG_VALUE_START = "<arg_value>"
ARG_VALUE_END = "</arg_value>"

# Special tokens that delimit conversation turns in a rendered GLM prompt.
# Reasoning markers belonging to earlier turns must not be mistaken for the
# state of the turn currently being generated.
GLM_TURN_BOUNDARIES = frozenset(
    ("<|system|>", "<|user|>", "<|assistant|>", "<|observation|>")
)

_ARG_RE = re.compile(
    r"<arg_key>(?P<key>.*?)</arg_key>\s*"
    r"<arg_value>(?P<value>.*?)</arg_value>",
    re.DOTALL,
)
_PARTIAL_ARG_RE = re.compile(
    r"<arg_key>(?P<key>.*?)</arg_key>\s*"
    r"<arg_value>(?P<value>.*)$",
    re.DOTALL,
)


class Glm47ArgConverter:
    """Converts one call's arguments, parsing each closed pair once.

    Only the text after the last closed value is kept: a closed pair is
    never rescanned, and the open trailing value can only start there.
    """

    __slots__ = ("_params", "_pending_parts")

    def __init__(self) -> None:
        self._params: dict[str, object] = {}
        self._pending_parts: list[str] = []

    def feed(self, delta: str) -> None:
        self._pending_parts.append(delta)

    def _pending_text(self) -> str:
        pending = "".join(self._pending_parts)
        self._pending_parts = [pending]
        return pending

    def _parse(self, partial: bool) -> tuple[dict[str, object], str | None]:
        """Return the arguments and the key of the open trailing value, if any."""
        params = self._params
        pending = self._pending_text()
        pos = 0
        for match in _ARG_RE.finditer(pending):
            params[match.group("key").strip()] = match.group("value")
            pos = match.end()
        if pos:
            pending = pending[pos:]
            self._pending_parts = [pending]

        open_key = None
        if partial:
            match = _PARTIAL_ARG_RE.search(pending)
            if match:
                key = match.group("key").strip()
                if key:
                    params = {**params, key: match.group("value")}
                    open_key = key

        return params, open_key

    def convert(self, partial: bool) -> str:
        return json.dumps(self._parse(partial)[0], ensure_ascii=False)

    def extender(self) -> ArgValueExtender | None:
        # The open value runs to the end of the arguments and is the last
        # JSON value, so a delta appends its JSON-escaped text unless it
        # closes the value.
        params, open_key = self._parse(partial=True)
        if open_key is None or next(reversed(params)) != open_key:
            return None

        keep = len(ARG_VALUE_END) - 1
        tail = self._pending_text()[-keep:]

        def extend(delta: str) -> str | None:
            nonlocal tail
            text = tail + delta
            if ARG_VALUE_END in text:
                return None
            tail = text[-keep:]
            return json.dumps(delta, ensure_ascii=False)[1:-1]

        return extend


def _glm53_always_thinks(tokenizer: TokenizerLike) -> bool:
    # GLM-5.3 keeps the GLM-4.x prompt and tool-call format but replaces the
    # enable_thinking toggle with an always-on "Reasoning Effort:" header, so
    # its generation prompt always opens <think> and the model output carries
    # no opening tag.
    template = getattr(tokenizer, "chat_template", None)
    return (
        isinstance(template, str)
        and "[gMASK]<sop>" in template
        and "Reasoning Effort:" in template
        and "enable_thinking" not in template
        and "<tool_call>" in template
        and "<arg_key>" in template
        and "<arg_value>" in template
    )


@functools.cache
def glm47_moe_config(thinking: bool = True) -> ParserEngineConfig:
    arg_tag_transitions = {
        (ParserState.TOOL_ARGS, terminal): Transition(
            ParserState.TOOL_ARGS,
            (EventType.ARG_VALUE_CHUNK,),
        )
        for terminal in (
            "ARG_KEY_START",
            "ARG_KEY_END",
            "ARG_VALUE_START",
            "ARG_VALUE_END",
        )
    }

    reasoning_terminals = (
        {
            "THINK_START": THINK_START,
            "THINK_END": THINK_END,
        }
        if thinking
        else {}
    )
    reasoning_token_id_terminals = (
        {
            "THINK_START": THINK_START,
            "THINK_END": THINK_END,
        }
        if thinking
        else {}
    )
    reasoning_transitions = (
        {
            (ParserState.CONTENT, "THINK_START"): Transition(
                ParserState.REASONING,
                (EventType.REASONING_START,),
            ),
            (ParserState.REASONING, "THINK_END"): Transition(
                ParserState.CONTENT,
                (EventType.REASONING_END,),
            ),
            (ParserState.CONTENT, "THINK_END"): Transition(
                ParserState.CONTENT,
                (),
            ),
        }
        if thinking
        else {}
    )

    return ParserEngineConfig(
        name="glm47_moe",
        initial_state=ParserState.REASONING if thinking else ParserState.CONTENT,
        terminals={
            **reasoning_terminals,
            "TOOL_START": TOOL_CALL_START,
            "TOOL_END": TOOL_CALL_END,
            "ARG_KEY_START": ARG_KEY_START,
            "ARG_KEY_END": ARG_KEY_END,
            "ARG_VALUE_START": ARG_VALUE_START,
            "ARG_VALUE_END": ARG_VALUE_END,
        },
        token_id_terminals={
            **reasoning_token_id_terminals,
            "TOOL_START": TOOL_CALL_START,
            "TOOL_END": TOOL_CALL_END,
        },
        transitions={
            **reasoning_transitions,
            (ParserState.REASONING, "THINK_START"): Transition(
                ParserState.REASONING,
                (),
            ),
            (ParserState.REASONING, "TOOL_START"): Transition(
                ParserState.TOOL_NAME,
                (EventType.REASONING_END, EventType.TOOL_CALL_START),
            ),
            (ParserState.CONTENT, "TOOL_START"): Transition(
                ParserState.TOOL_NAME,
                (EventType.TOOL_CALL_START,),
            ),
            (ParserState.TOOL_NAME, "ARG_KEY_START"): Transition(
                ParserState.TOOL_ARGS,
                (EventType.ARG_VALUE_CHUNK,),
            ),
            (ParserState.TOOL_NAME, "TOOL_END"): Transition(
                ParserState.CONTENT,
                (EventType.TOOL_CALL_END,),
            ),
            (ParserState.TOOL_ARGS, "TOOL_END"): Transition(
                ParserState.CONTENT,
                (EventType.TOOL_CALL_END,),
            ),
            **arg_tag_transitions,
        },
        turn_boundary_tokens=GLM_TURN_BOUNDARIES,
        slot_arg_converter=Glm47ArgConverter,
        arg_structural_chars=frozenset(">"),
        stream_arg_deltas=True,
        tool_args_json=False,
        validate_tool_names=True,
    )


class Glm47MoeParser(ParserEngine):
    """GLM-4.7 parser backed by the declarative parser engine."""

    def __init__(
        self,
        tokenizer: TokenizerLike,
        tools: list[Tool] | None = None,
        **kwargs,
    ) -> None:
        chat_kwargs = kwargs.get("chat_template_kwargs", {}) or {}
        thinking = chat_kwargs.get("thinking", None)
        enable_thinking = chat_kwargs.get("enable_thinking", None)
        if (thinking is False or enable_thinking is False) and _glm53_always_thinks(
            tokenizer
        ):
            logger.warning_once(
                "Ignoring enable_thinking/thinking: the GLM-5.3 chat template "
                "has no thinking switch, so reasoning is always on and "
                "disabling extraction would leak it into the content."
            )
            thinking = None
            enable_thinking = None
        self.thinking_enabled = (
            True
            if thinking is None and enable_thinking is None
            else bool(thinking) or bool(enable_thinking)
        )
        kwargs.setdefault(
            "parser_engine_config",
            glm47_moe_config(thinking=self.thinking_enabled),
        )
        super().__init__(tokenizer, tools, **kwargs)

    def _emit_name_delta(self, idx: int, deltas, name: str | None) -> None:
        if name is not None:
            name = name.strip()
        super()._emit_name_delta(idx, deltas, name)

    def _handle_tool_end(self, event, deltas) -> None:
        idx = event.tool_index
        if 0 <= idx < len(self._tool_slots):
            self._tool_slots[idx].name = self._tool_slots[idx].name.strip()
        super()._handle_tool_end(event, deltas)

    def extract_reasoning(
        self,
        model_output: str,
        request: ChatCompletionRequest | ResponsesRequest,
    ) -> tuple[str | None, str | None]:
        if not self.thinking_enabled:
            return None, model_output
        return super().extract_reasoning(model_output, request)
