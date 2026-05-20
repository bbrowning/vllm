# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Adapters that expose :class:`GrammarParser` through the legacy
:class:`ReasoningParser` and :class:`ToolParser` interfaces.

This lets grammar parsers flow through the existing serving-layer code
paths that expect separate reasoning and tool parser instances, without
any changes to the serving layer itself.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from vllm.reasoning.abs_reasoning_parsers import ReasoningParser
from vllm.tool_parsers.abstract_tool_parser import ToolParser

if TYPE_CHECKING:
    from vllm.entrypoints.openai.chat_completion.protocol import (
        ChatCompletionRequest,
    )
    from vllm.entrypoints.openai.engine.protocol import (
        DeltaMessage,
        ExtractedToolCallInformation,
    )
    from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
    from vllm.grammar_parser.unified_parser import GrammarParser
    from vllm.tokenizers import TokenizerLike
    from vllm.tool_parsers.utils import Tool


class GrammarReasoningAdapter(ReasoningParser):
    """Adapts a :class:`GrammarParser` to the :class:`ReasoningParser`
    interface so grammar parsers can be used as reasoning parsers in the
    existing serving code.

    Subclasses set :attr:`_grammar_cls` to the concrete
    :class:`GrammarParser` class.
    """

    _grammar_cls: type[GrammarParser]

    def __init__(self, tokenizer: TokenizerLike, *args, **kwargs) -> None:
        super().__init__(tokenizer, *args, **kwargs)
        self._grammar = self._grammar_cls(tokenizer)  # type: ignore[call-arg]

    def is_reasoning_end(self, input_ids: Sequence[int]) -> bool:
        return self._grammar.is_reasoning_end(list(input_ids))

    def extract_content_ids(self, input_ids: list[int]) -> list[int]:
        return self._grammar.extract_content_ids(input_ids)

    def extract_reasoning(
        self,
        model_output: str,
        request: ChatCompletionRequest | ResponsesRequest,
    ) -> tuple[str | None, str | None]:
        # The grammar parser's extract_reasoning() consumes tool call
        # events, leaving only TEXT_CHUNK as content.  The serving layer
        # expects tool call text to remain in the content so it can be
        # passed to the tool parser.  Use text-based stripping instead
        # so the tool call text is preserved verbatim.
        cfg = self._grammar.grammar_config
        start = cfg.terminals.get("THINK_START") or ""
        end = cfg.terminals.get("THINK_END") or ""

        if not start or not end:
            return None, model_output

        start_idx = model_output.find(start)
        if start_idx == -1:
            return None, model_output

        end_idx = model_output.find(end, start_idx + len(start))
        if end_idx == -1:
            return None, model_output

        reasoning = model_output[start_idx + len(start) : end_idx]
        content = model_output[:start_idx] + model_output[end_idx + len(end) :]
        return reasoning or None, content or None

    def extract_reasoning_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
    ) -> DeltaMessage | None:
        return self._grammar.extract_reasoning_streaming(
            previous_text,
            current_text,
            delta_text,
            previous_token_ids,
            current_token_ids,
            delta_token_ids,
        )

    @property
    def reasoning_start_str(self) -> str | None:
        return self._grammar.reasoning_start_str

    @property
    def reasoning_end_str(self) -> str | None:
        return self._grammar.reasoning_end_str

    def adjust_request(
        self,
        request: ChatCompletionRequest | ResponsesRequest,
    ) -> ChatCompletionRequest | ResponsesRequest:
        return self._grammar.adjust_request(request)

    def has_reasoning_ended(self) -> bool | None:
        return self._grammar._reasoning_ended


class GrammarToolAdapter(ToolParser):
    """Adapts a :class:`GrammarParser` to the :class:`ToolParser` interface.

    :meth:`extract_tool_calls` starts the grammar engine in ``CONTENT``
    state so it can parse reasoning-stripped content (i.e. the output of
    :meth:`ReasoningParser.extract_reasoning`).

    Subclasses set :attr:`_grammar_cls` to the concrete
    :class:`GrammarParser` class.
    """

    _grammar_cls: type[GrammarParser]

    def __init__(
        self,
        tokenizer: TokenizerLike,
        tools: list[Tool] | None = None,
    ) -> None:
        super().__init__(tokenizer, tools)
        self._grammar = self._grammar_cls(tokenizer, tools)  # type: ignore[call-arg]

    def adjust_request(
        self,
        request: ChatCompletionRequest | ResponsesRequest,
    ) -> ChatCompletionRequest | ResponsesRequest:
        return self._grammar.adjust_request(request)

    def extract_tool_calls(
        self,
        model_output: str,
        request: ChatCompletionRequest,
    ) -> ExtractedToolCallInformation:
        return self._grammar.extract_tool_calls_from_content(model_output, request)

    def extract_tool_calls_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
        request: ChatCompletionRequest,
    ) -> DeltaMessage | None:
        return self._grammar.extract_tool_calls_streaming(
            previous_text,
            current_text,
            delta_text,
            previous_token_ids,
            current_token_ids,
            delta_token_ids,
            request,
        )


def make_adapters(
    grammar_cls: type[GrammarParser],
) -> tuple[type[GrammarReasoningAdapter], type[GrammarToolAdapter]]:
    """Create :class:`ReasoningParser` and :class:`ToolParser` adapter
    classes for a given :class:`GrammarParser` subclass."""
    reasoning_adapter = type(
        f"{grammar_cls.__name__}ReasoningAdapter",
        (GrammarReasoningAdapter,),
        {"_grammar_cls": grammar_cls},
    )
    tool_adapter = type(
        f"{grammar_cls.__name__}ToolAdapter",
        (GrammarToolAdapter,),
        {"_grammar_cls": grammar_cls},
    )
    return reasoning_adapter, tool_adapter
