# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Registered unified grammar parser classes.

Each class passes a unified ``GrammarConfig`` (covering both reasoning
and tool calls) to :class:`GrammarParser`, and adds any model-specific
post-processing.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from vllm.entrypoints.openai.engine.protocol import DeltaMessage
from vllm.grammar_parser.events import SemanticEvent
from vllm.grammar_parser.grammars.deepseek_v4_unified import deepseek_v4_unified_config
from vllm.grammar_parser.grammars.gemma4_unified import gemma4_unified_config
from vllm.grammar_parser.grammars.hermes import hermes_config
from vllm.grammar_parser.grammars.qwen3_unified import qwen3_unified_config
from vllm.grammar_parser.grammars.qwen3xml import qwen3xml_config
from vllm.grammar_parser.grammars.think_tag import think_tag_config
from vllm.grammar_parser.unified_parser import GrammarParser

if TYPE_CHECKING:
    from vllm.entrypoints.openai.chat_completion.protocol import (
        ChatCompletionRequest,
    )
    from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
    from vllm.tokenizers import TokenizerLike
    from vllm.tool_parsers.abstract_tool_parser import Tool

_GEMMA4_THOUGHT_PREFIX = "thought\n"


class Gemma4GrammarParser(GrammarParser):
    """Unified Gemma4 parser: ``<|channel>`` reasoning + ``<|tool_call>``
    tool calls in a single engine.

    Absorbs model-specific logic from the separate
    ``GrammarGemma4ReasoningParser``:
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
            grammar_config=gemma4_unified_config(),
            **kwargs,
        )
        vocab = self.vocab
        self._tool_call_token_id: int | None = vocab.get("<|tool_call>")
        self._new_turn_token_id: int | None = vocab.get("<|turn>")
        self._tool_response_token_id: int | None = vocab.get("<|tool_response>")
        self._reasoning_text: str = ""
        self._prefix_stripped: bool = False

    def _reset(self) -> None:
        super()._reset()
        self._reasoning_text = ""
        self._prefix_stripped = False

    def adjust_request(
        self, request: ChatCompletionRequest | ResponsesRequest
    ) -> ChatCompletionRequest | ResponsesRequest:
        request.skip_special_tokens = False
        return request

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
    ) -> DeltaMessage | None:
        delta = super()._events_to_delta(events)
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
                delta.reasoning = ""
                return delta
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


class Qwen3GrammarParser(GrammarParser):
    """Unified Qwen3 parser: ``<think>``/``</think>`` reasoning +
    ``<tool_call>`` XML tool calls in a single engine.

    Absorbs model-specific logic from the separate
    ``GrammarQwen3ReasoningParser``:
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
        super().__init__(
            tokenizer,
            tools,
            grammar_config=qwen3_unified_config(),
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


class HermesGrammarParser(GrammarParser):
    """Unified Hermes parser: ``<tool_call>``/``</tool_call>`` JSON tool
    calls."""

    def __init__(
        self,
        tokenizer: TokenizerLike,
        tools: list[Tool] | None = None,
        **kwargs,
    ) -> None:
        super().__init__(tokenizer, tools, grammar_config=hermes_config(), **kwargs)


class Qwen3XMLGrammarParser(GrammarParser):
    """Unified Qwen3 XML parser: ``<tool_call><function=...>`` tool calls."""

    def __init__(
        self,
        tokenizer: TokenizerLike,
        tools: list[Tool] | None = None,
        **kwargs,
    ) -> None:
        super().__init__(tokenizer, tools, grammar_config=qwen3xml_config(), **kwargs)


class Qwen3CoderGrammarParser(GrammarParser):
    """Unified Qwen3 Coder parser: same XML format as Qwen3 XML."""

    def __init__(
        self,
        tokenizer: TokenizerLike,
        tools: list[Tool] | None = None,
        **kwargs,
    ) -> None:
        super().__init__(tokenizer, tools, grammar_config=qwen3xml_config(), **kwargs)


class ThinkTagGrammarParser(GrammarParser):
    """Unified think-tag reasoning parser: ``<think>``/``</think>``."""

    def __init__(
        self,
        tokenizer: TokenizerLike,
        tools: list[Tool] | None = None,
        **kwargs,
    ) -> None:
        super().__init__(tokenizer, tools, grammar_config=think_tag_config(), **kwargs)


class DeepSeekV4GrammarParser(GrammarParser):
    """Unified DeepSeek V4 parser: ``<think>``/``</think>`` reasoning +
    DSML tool calls (``<｜DSML｜tool_calls>``/``<｜DSML｜invoke>``) in a
    single state machine.

    Initial state is CONTENT — the model generates ``<think>`` itself in
    thinking mode; in chat mode the prompt pre-fills ``</think>`` so the
    model outputs content directly.

    ``skip_special_tokens=False`` is required so that DSML special tokens
    appear in ``delta_text`` for text-based lexing.
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
            grammar_config=deepseek_v4_unified_config(),
            **kwargs,
        )

    def adjust_request(
        self, request: ChatCompletionRequest | ResponsesRequest
    ) -> ChatCompletionRequest | ResponsesRequest:
        request.skip_special_tokens = False
        return request
