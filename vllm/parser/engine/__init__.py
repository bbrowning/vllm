# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Streaming parser engine framework for tool call and reasoning extraction.

Instead of hand-rolling a parser for every model's tool-call / reasoning
format, each format is declared as a ParserEngineConfig (terminals,
states, and transitions) and a shared incremental engine handles
streaming, ambiguity buffering, token-ID mapping, and delta computation.
"""

from vllm.parser.engine.events import EventType, SemanticEvent
from vllm.parser.engine.parsers.deepseek_v4 import DeepSeekV4Parser
from vllm.parser.engine.parsers.gemma4 import Gemma4Parser
from vllm.parser.engine.parsers.nemotron_v3 import NemotronV3Parser
from vllm.parser.engine.parsers.qwen3 import (
    Qwen3Parser,
    Qwen3XMLParser,
)

__all__ = [
    "DeepSeekV4Parser",
    "EventType",
    "Gemma4Parser",
    "NemotronV3Parser",
    "Qwen3Parser",
    "Qwen3XMLParser",
    "SemanticEvent",
]
