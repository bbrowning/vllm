# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Concrete adapter classes for each registered grammar parser.

These are created via :func:`make_adapters` and exposed as module-level
names so that :class:`ReasoningParserManager` and
:class:`ToolParserManager` can load them lazily.
"""

from vllm.grammar_parser.adapters import make_adapters
from vllm.grammar_parser.parsers import (
    DeepSeekV4GrammarParser,
    Gemma4GrammarParser,
    HermesGrammarParser,
    NemotronV3GrammarParser,
    Qwen3GrammarParser,
    Qwen3XMLGrammarParser,
)

(
    DeepSeekV4GrammarParserReasoningAdapter,
    DeepSeekV4GrammarParserToolAdapter,
) = make_adapters(DeepSeekV4GrammarParser)

(
    Gemma4GrammarParserReasoningAdapter,
    Gemma4GrammarParserToolAdapter,
) = make_adapters(Gemma4GrammarParser)

(
    Qwen3GrammarParserReasoningAdapter,
    Qwen3GrammarParserToolAdapter,
) = make_adapters(Qwen3GrammarParser)

(
    HermesGrammarParserReasoningAdapter,
    HermesGrammarParserToolAdapter,
) = make_adapters(HermesGrammarParser)

(
    NemotronV3GrammarParserReasoningAdapter,
    NemotronV3GrammarParserToolAdapter,
) = make_adapters(NemotronV3GrammarParser)

(
    Qwen3XMLGrammarParserReasoningAdapter,
    Qwen3XMLGrammarParserToolAdapter,
) = make_adapters(Qwen3XMLGrammarParser)
