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

from vllm.grammar_parser.events import EventType, SemanticEvent
from vllm.grammar_parser.parsers import (
    DeepSeekV4GrammarParser,
    Gemma4GrammarParser,
    HermesGrammarParser,
    NemotronV3GrammarParser,
    Qwen3GrammarParser,
    Qwen3XMLGrammarParser,
)

__all__ = [
    "DeepSeekV4GrammarParser",
    "EventType",
    "Gemma4GrammarParser",
    "HermesGrammarParser",
    "NemotronV3GrammarParser",
    "Qwen3GrammarParser",
    "Qwen3XMLGrammarParser",
    "SemanticEvent",
]
