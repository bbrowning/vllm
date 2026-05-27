# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Concrete adapter classes for each registered grammar parser.

These are created via :func:`make_adapters` and exposed as module-level
names so that :class:`ReasoningParserManager` and
:class:`ToolParserManager` can load them lazily.
"""

from vllm.parser.grammar.adapters import make_adapters
from vllm.parser.grammar.parsers.deepseek_v4 import DeepSeekV4GrammarParser
from vllm.parser.grammar.parsers.gemma4 import Gemma4GrammarParser
from vllm.parser.grammar.parsers.nemotron_v3 import NemotronV3GrammarParser
from vllm.parser.grammar.parsers.qwen3 import (
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
    NemotronV3GrammarParserReasoningAdapter,
    NemotronV3GrammarParserToolAdapter,
) = make_adapters(NemotronV3GrammarParser)

(
    Qwen3XMLGrammarParserReasoningAdapter,
    Qwen3XMLGrammarParserToolAdapter,
) = make_adapters(Qwen3XMLGrammarParser)
