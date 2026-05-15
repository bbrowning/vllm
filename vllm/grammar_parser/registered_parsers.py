# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Registered grammar-based parser classes.

Each class is a thin wrapper that passes the appropriate
:class:`GrammarConfig` to the generic :class:`GrammarToolParser` or
:class:`GrammarReasoningParser` adapters, so they can be instantiated
with just ``(tokenizer)`` by the manager registries.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from vllm.grammar_parser.adapter import GrammarReasoningParser, GrammarToolParser
from vllm.grammar_parser.grammars.gemma4 import gemma4_config
from vllm.grammar_parser.grammars.hermes import hermes_config
from vllm.grammar_parser.grammars.qwen3xml import qwen3xml_config
from vllm.grammar_parser.grammars.think_tag import think_tag_config

if TYPE_CHECKING:
    from vllm.tokenizers import TokenizerLike
    from vllm.tool_parsers.abstract_tool_parser import Tool


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
