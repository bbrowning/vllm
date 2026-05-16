# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Registered grammar-based parser classes.

Each class is a thin wrapper that passes the appropriate
:class:`GrammarConfig` to the generic :class:`GrammarToolParser` or
:class:`GrammarReasoningParser` adapters, so they can be instantiated
with just ``(tokenizer)`` by the manager registries.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from vllm.entrypoints.openai.engine.protocol import DeltaMessage
from vllm.grammar_parser.adapter import GrammarReasoningParser, GrammarToolParser
from vllm.grammar_parser.events import EventType
from vllm.grammar_parser.grammar_config import (
    GrammarConfig,
    ParserState,
    Transition,
)
from vllm.grammar_parser.grammars.gemma4 import gemma4_config
from vllm.grammar_parser.grammars.gemma4_channel import gemma4_channel_config
from vllm.grammar_parser.grammars.hermes import hermes_config
from vllm.grammar_parser.grammars.qwen3xml import qwen3xml_config
from vllm.grammar_parser.grammars.think_tag import think_tag_config

if TYPE_CHECKING:
    from vllm.entrypoints.openai.chat_completion.protocol import (
        ChatCompletionRequest,
    )
    from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
    from vllm.tokenizers import TokenizerLike
    from vllm.tool_parsers.abstract_tool_parser import Tool

_GEMMA4_THOUGHT_PREFIX = "thought\n"


class GrammarQwen3CoderToolParser(GrammarToolParser):
    def __init__(
        self,
        tokenizer: TokenizerLike,
        tools: list[Tool] | None = None,
    ) -> None:
        super().__init__(tokenizer, tools, grammar_config=qwen3xml_config())


def _qwen3_reasoning_config() -> GrammarConfig:
    """Grammar config for Qwen3 reasoning.

    Starts in REASONING state because Qwen3.5+ chat templates place
    ``<think>`` in the prompt, so generated output begins mid-reasoning.
    ``<think>`` in output (old templates) is consumed as a no-op.
    ``<tool_call>`` acts as an implicit reasoning end — the model may
    emit it inside thinking without closing ``</think>`` first.
    """
    return GrammarConfig(
        name="qwen3_reasoning",
        initial_state=ParserState.REASONING,
        terminals={
            "THINK_START": "<think>",
            "THINK_END": "</think>",
            "TOOL_CALL_START": "<tool_call>",
        },
        token_id_terminals={
            "THINK_START": "<think>",
            "THINK_END": "</think>",
            "TOOL_CALL_START": "<tool_call>",
        },
        transitions={
            (ParserState.REASONING, "THINK_START"): Transition(
                ParserState.REASONING,
                [],
            ),
            (ParserState.REASONING, "THINK_END"): Transition(
                ParserState.CONTENT,
                [EventType.REASONING_END],
            ),
            (ParserState.REASONING, "TOOL_CALL_START"): Transition(
                ParserState.CONTENT,
                [EventType.REASONING_END],
            ),
        },
        content_events={
            ParserState.CONTENT: EventType.TEXT_CHUNK,
            ParserState.REASONING: EventType.REASONING_CHUNK,
        },
    )


class GrammarQwen3ReasoningParser(GrammarReasoningParser):
    """Reasoning parser for Qwen3 ``<think>``/``</think>`` format.

    Uses ``initial_state=REASONING`` so output without ``<think>``
    (Qwen3.5+ style where ``<think>`` is in the prompt) is correctly
    treated as reasoning. ``<think>`` in output (old template) is
    consumed as a no-op transition. ``<tool_call>`` is handled as an
    implicit reasoning end by the grammar config.

    Extends :class:`GrammarReasoningParser` with:
    - ``is_reasoning_end`` that detects unpaired ``<tool_call>`` tokens
    """

    def __init__(self, tokenizer: TokenizerLike, **kwargs) -> None:
        super().__init__(
            tokenizer,
            grammar_config=_qwen3_reasoning_config(),
            **kwargs,
        )
        vocab = self.vocab
        self._tool_call_token_id: int | None = vocab.get("<tool_call>")
        self._tool_call_end_token_id: int | None = vocab.get("</tool_call>")

    def is_reasoning_end(self, input_ids: Sequence[int]) -> bool:
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


class GrammarGemma4ToolParser(GrammarToolParser):
    def __init__(
        self,
        tokenizer: TokenizerLike,
        tools: list[Tool] | None = None,
    ) -> None:
        super().__init__(tokenizer, tools, grammar_config=gemma4_config())


class GrammarQwen3XMLToolParser(GrammarToolParser):
    def __init__(
        self,
        tokenizer: TokenizerLike,
        tools: list[Tool] | None = None,
    ) -> None:
        super().__init__(tokenizer, tools, grammar_config=qwen3xml_config())


class GrammarHermesToolParser(GrammarToolParser):
    def __init__(
        self,
        tokenizer: TokenizerLike,
        tools: list[Tool] | None = None,
    ) -> None:
        super().__init__(tokenizer, tools, grammar_config=hermes_config())


class GrammarThinkTagReasoningParser(GrammarReasoningParser):
    def __init__(self, tokenizer: TokenizerLike, **kwargs) -> None:
        super().__init__(tokenizer, grammar_config=think_tag_config(), **kwargs)


class GrammarGemma4ReasoningParser(GrammarReasoningParser):
    """Reasoning parser for Gemma4 ``<|channel>``/``<channel|>`` format.

    Mirrors the behavior of the original ``Gemma4ReasoningParser``:
    - Uses ``<|channel>``/``<channel|>`` token boundaries
    - Strips the ``thought\\n`` prefix from reasoning content
    - Sets ``skip_special_tokens=False`` so boundary tokens are visible
    - Detects reasoning end via ``<|tool_call>`` token as well
    """

    def __init__(self, tokenizer: TokenizerLike, **kwargs) -> None:
        super().__init__(
            tokenizer,
            grammar_config=gemma4_channel_config(),
            **kwargs,
        )
        vocab = self.vocab
        self._tool_call_token_id: int | None = vocab.get("<|tool_call>")
        self._new_turn_token_id: int | None = vocab.get("<|turn>")
        self._tool_response_token_id: int | None = vocab.get("<|tool_response>")
        self._reasoning_text: str = ""
        self._prefix_stripped: bool = False

    def _reset_streaming_state(self) -> None:
        super()._reset_streaming_state()
        self._reasoning_text = ""
        self._prefix_stripped = False

    def adjust_request(
        self, request: ChatCompletionRequest | ResponsesRequest
    ) -> ChatCompletionRequest | ResponsesRequest:
        request.skip_special_tokens = False
        return request

    def is_reasoning_end(self, input_ids: Sequence[int]) -> bool:
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

    def extract_reasoning_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
    ) -> DeltaMessage | None:
        result = super().extract_reasoning_streaming(
            previous_text,
            current_text,
            delta_text,
            previous_token_ids,
            current_token_ids,
            delta_token_ids,
        )
        if result is None:
            return None
        if result.reasoning is None:
            return result

        self._reasoning_text += result.reasoning

        if self._prefix_stripped:
            return result

        if self._reasoning_text.startswith(_GEMMA4_THOUGHT_PREFIX):
            prefix_len = len(_GEMMA4_THOUGHT_PREFIX)
            prev_reasoning_len = len(self._reasoning_text) - len(result.reasoning)
            if prev_reasoning_len >= prefix_len:
                self._prefix_stripped = True
                return result
            chars_of_prefix_in_delta = prefix_len - prev_reasoning_len
            stripped = result.reasoning[chars_of_prefix_in_delta:]
            if stripped:
                self._prefix_stripped = True
                result.reasoning = stripped
                return result
            if len(self._reasoning_text) >= prefix_len:
                self._prefix_stripped = True
                result.reasoning = ""
                return result
            return None

        if _GEMMA4_THOUGHT_PREFIX.startswith(self._reasoning_text):
            return None

        self._prefix_stripped = True
        result.reasoning = self._reasoning_text
        return result
