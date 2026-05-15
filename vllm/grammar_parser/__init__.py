# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Grammar-driven streaming parser framework for tool call and reasoning
extraction.

Instead of hand-rolling a state machine for every model's tool-call /
reasoning format, each format is declared as a Lark grammar and a shared
incremental engine handles streaming, ambiguity buffering, token-ID
mapping, and delta computation.
"""

from vllm.grammar_parser.events import EventType, SemanticEvent

__all__ = ["EventType", "SemanticEvent"]
