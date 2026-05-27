# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Grammar-driven streaming parser framework for tool call and reasoning
extraction.

Instead of hand-rolling a parser for every model's tool-call / reasoning
format, each format is declared as a GrammarConfig (terminals, states,
and transitions) and a shared incremental engine handles streaming,
ambiguity buffering, token-ID mapping, and delta computation.
"""

from vllm.parser.grammar.events import EventType, SemanticEvent
from vllm.parser.grammar.parsers.deepseek_v4 import DeepSeekV4GrammarParser
from vllm.parser.grammar.parsers.gemma4 import Gemma4GrammarParser
from vllm.parser.grammar.parsers.nemotron_v3 import NemotronV3GrammarParser
from vllm.parser.grammar.parsers.qwen3 import (
    Qwen3GrammarParser,
    Qwen3XMLGrammarParser,
)

__all__ = [
    "DeepSeekV4GrammarParser",
    "EventType",
    "Gemma4GrammarParser",
    "NemotronV3GrammarParser",
    "Qwen3GrammarParser",
    "Qwen3XMLGrammarParser",
    "SemanticEvent",
]
