# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Nemotron V3 grammar configuration.

The Nemotron 3 Super model uses the same tool call and reasoning
format as Qwen3 (``<think>``/``</think>`` + ``<tool_call>`` XML).
This config reuses :func:`qwen3_config` with a distinct name.
"""

from __future__ import annotations

import dataclasses

from vllm.grammar_parser.grammar_config import GrammarConfig
from vllm.grammar_parser.grammars.qwen3 import qwen3_config


def nemotron_v3_config() -> GrammarConfig:
    """Return the grammar config for Nemotron V3 reasoning + tool calls."""
    return dataclasses.replace(qwen3_config(), name="nemotron_v3")
