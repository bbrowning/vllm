# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Gemma4 grammar parser.

Handles channel-based reasoning plus custom tool call format in a single
state machine::

    <|channel>thought
    ...reasoning...<channel|>
    <|tool_call>call:func_name{key:<|"|>value<|"|>,num:42}<tool_call|>

Gemma4 requires ``skip_special_tokens=False`` because its arg format uses
``<|"|>`` (a special token) as a string delimiter.  The side-effect is that
*all* special tokens become visible in ``delta_text``, including ones the
parser doesn't use.  The model actively generates some of these (e.g.
``<turn|>`` after content, ``<|tool_response>`` after tool calls — see replay
test cases 005/006), so they must be explicitly dropped to prevent leaking
into response content.

``_GEMMA4_MODEL_DROP_TOKENS`` lists the Gemma4-specific tokens to drop.
Universal structural tokens (``<eos>``, ``<bos>``, etc.) are handled
separately via :data:`~.grammar_config.STRUCTURAL_DROP_TOKENS`.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from vllm.entrypoints.openai.engine.protocol import DeltaMessage
from vllm.parser.grammar.events import EventType, SemanticEvent
from vllm.parser.grammar.grammar_config import (
    GrammarConfig,
    ParserState,
    Transition,
)
from vllm.parser.grammar.unified_parser import GrammarParser
from vllm.tool_parsers.gemma4_tool_parser import (
    _parse_gemma4_args,
)

if TYPE_CHECKING:
    from vllm.entrypoints.openai.chat_completion.protocol import (
        ChatCompletionRequest,
    )
    from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
    from vllm.tokenizers import TokenizerLike
    from vllm.tool_parsers.abstract_tool_parser import Tool

_GEMMA4_MODEL_DROP_TOKENS: set[str] = {
    # Turn boundaries (model generates <turn|> after content — test case 006)
    "<|turn>",
    "<turn|>",
    # Channel / reasoning
    "<|channel>",
    "<channel|>",
    # Tool protocol tokens (model generates <|tool_response> — test case 005)
    "<|tool>",
    "<tool|>",
    "<|tool_call>",
    "<tool_call|>",
    "<|tool_response>",
    "<tool_response|>",
    '<|"|>',
    # Thinking
    "<|think|>",
    # Multi-modal (defensive — not expected during text completion)
    "<|image>",
    "<|image|>",
    "<image|>",
    "<|audio>",
    "<|audio|>",
    "<audio|>",
    "<|video|>",
}

CHANNEL_START = "<|channel>"
CHANNEL_END = "<channel|>"
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


_GEMMA4_CONFIG: GrammarConfig | None = None


def gemma4_config() -> GrammarConfig:
    """Return the grammar config for Gemma4 reasoning + tool calls."""
    global _GEMMA4_CONFIG
    if _GEMMA4_CONFIG is not None:
        return _GEMMA4_CONFIG

    used_tokens = {
        CHANNEL_START,
        CHANNEL_END,
        TOOL_CALL_START,
        TOOL_CALL_END,
        '<|"|>',
    }

    _GEMMA4_CONFIG = GrammarConfig(
        name="gemma4",
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
        drop_tokens=_GEMMA4_MODEL_DROP_TOKENS - used_tokens,
    )
    return _GEMMA4_CONFIG


_GEMMA4_THOUGHT_PREFIX = "thought\n"


class Gemma4GrammarParser(GrammarParser):
    """Gemma4 parser: ``<|channel>`` reasoning + ``<|tool_call>``
    tool calls in a single engine.

    - Strips the ``thought\\n`` prefix from reasoning content
    - Sets ``skip_special_tokens=False`` so boundary tokens are visible
    - Detects ``<|tool_call>`` token as implicit reasoning end
    """

    def __init__(
        self,
        tokenizer: TokenizerLike,
        tools: list[Tool] | None = None,
        **kwargs,
    ) -> None:
        super().__init__(
            tokenizer,
            tools,
            grammar_config=gemma4_config(),
            **kwargs,
        )
        vocab = self.vocab
        self._tool_call_token_id: int | None = vocab.get("<|tool_call>")
        self._new_turn_token_id: int | None = vocab.get("<|turn>")
        self._tool_response_token_id: int | None = vocab.get("<|tool_response>")
        self._reasoning_text: str = ""
        self._prefix_stripped: bool = False

    def _reset(self, initial_state=None) -> None:
        super()._reset(initial_state=initial_state)
        self._reasoning_text = ""
        self._prefix_stripped = False

    def is_reasoning_end(self, input_ids: list[int]) -> bool:
        end_id = self._reasoning_end_token_id
        start_id = self._reasoning_start_token_id
        tool_call_id = self._tool_call_token_id
        new_turn_id = self._new_turn_token_id
        tool_response_id = self._tool_response_token_id

        for i in range(len(input_ids) - 1, -1, -1):
            tid = input_ids[i]
            if start_id is not None and tid == start_id:
                return False
            if tool_call_id is not None and tid == tool_call_id:
                return True
            if new_turn_id is not None and tid == new_turn_id:
                return False
            if tool_response_id is not None and tid == tool_response_id:
                return False
            if end_id is not None and tid == end_id:
                return True
        return self._reasoning_ended

    def _events_to_delta(
        self,
        events: list[SemanticEvent],
        finished: bool = False,
    ) -> DeltaMessage | None:
        delta = super()._events_to_delta(events, finished=finished)
        if delta is None or delta.reasoning is None:
            return delta

        self._reasoning_text += delta.reasoning
        if self._prefix_stripped:
            return delta

        if self._reasoning_text.startswith(_GEMMA4_THOUGHT_PREFIX):
            prefix_len = len(_GEMMA4_THOUGHT_PREFIX)
            prev_reasoning_len = len(self._reasoning_text) - len(delta.reasoning)
            if prev_reasoning_len >= prefix_len:
                self._prefix_stripped = True
                return delta
            chars_of_prefix_in_delta = prefix_len - prev_reasoning_len
            stripped = delta.reasoning[chars_of_prefix_in_delta:]
            if stripped:
                self._prefix_stripped = True
                delta.reasoning = stripped
                return delta
            if len(self._reasoning_text) >= prefix_len:
                self._prefix_stripped = True
                delta.reasoning = None
                if delta.content is not None or delta.tool_calls:
                    return delta
                return None
            return None

        if _GEMMA4_THOUGHT_PREFIX.startswith(self._reasoning_text):
            return None

        self._prefix_stripped = True
        delta.reasoning = self._reasoning_text
        return delta

    def extract_reasoning(
        self,
        model_output: str,
        request: ChatCompletionRequest | ResponsesRequest,
    ) -> tuple[str | None, str | None]:
        reasoning, content = super().extract_reasoning(model_output, request)
        if reasoning and reasoning.startswith(_GEMMA4_THOUGHT_PREFIX):
            reasoning = reasoning[len(_GEMMA4_THOUGHT_PREFIX) :]
        return reasoning or None, content
