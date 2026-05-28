# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Nemotron V3 grammar parser.

The Nemotron 3 Super model uses the same tool call and reasoning
format as Qwen3 (``<think>``/``</think>`` + ``<tool_call>`` XML).
This config reuses :func:`qwen3_config` with a distinct name.

When ``enable_thinking=False`` or ``force_nonempty_content=True`` and
content is empty, reasoning and content are swapped.
"""

from __future__ import annotations

import dataclasses
import functools
from typing import TYPE_CHECKING

from vllm.parser.grammar.parsers.qwen3 import Qwen3GrammarParser, qwen3_config

if TYPE_CHECKING:
    from vllm.entrypoints.openai.chat_completion.protocol import (
        ChatCompletionRequest,
    )
    from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
    from vllm.parser.grammar.grammar_config import GrammarConfig
    from vllm.tokenizers import TokenizerLike
    from vllm.tool_parsers.abstract_tool_parser import Tool


@functools.cache
def nemotron_v3_config() -> GrammarConfig:
    """Return the grammar config for Nemotron V3 reasoning + tool calls."""
    return dataclasses.replace(qwen3_config(), name="nemotron_v3")


class NemotronV3GrammarParser(Qwen3GrammarParser):
    """Nemotron V3 parser: same format as Qwen3, with Nemotron-specific
    behavior: when ``enable_thinking=False`` or
    ``force_nonempty_content=True`` and content is empty, swaps
    reasoning and content.
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
            grammar_config=nemotron_v3_config(),
            **kwargs,
        )

    def extract_reasoning(
        self,
        model_output: str,
        request: ChatCompletionRequest | ResponsesRequest,
    ) -> tuple[str | None, str | None]:
        reasoning, content = super().extract_reasoning(model_output, request)
        chat_template_kwargs = getattr(request, "chat_template_kwargs", None)

        if (
            chat_template_kwargs
            and (
                chat_template_kwargs.get("enable_thinking") is False
                or chat_template_kwargs.get("force_nonempty_content") is True
            )
            and (content is None or not content.strip())
        ):
            reasoning, content = content, reasoning

        return reasoning, content
