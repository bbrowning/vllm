# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Parameterized grammar configuration for think-tag reasoning.

Covers 12+ models that use a start/end tag pair to delimit reasoning
content (e.g. ``<think>``/``</think>``, ``[THINK]``/``[/THINK]``,
``<seed:think>``/``</seed:think>``).

Usage::

    from vllm.grammar_parser.grammars.think_tag import think_tag_config

    # Default <think>/</think> tags:
    config = think_tag_config()

    # Custom tags:
    config = think_tag_config(
        start_tag="[THINK]",
        end_tag="[/THINK]",
        name="mistral",
    )
"""

from __future__ import annotations

from vllm.grammar_parser.events import EventType
from vllm.grammar_parser.grammar_config import (
    GrammarConfig,
    ParserState,
    Transition,
)


def think_tag_config(
    start_tag: str = "<think>",
    end_tag: str = "</think>",
    name: str = "think_tag",
) -> GrammarConfig:
    """Return a grammar config for think-tag reasoning.

    Args:
        start_tag: The tag that opens the reasoning block.
        end_tag: The tag that closes the reasoning block.
        name: Name for this grammar config instance.
    """
    return GrammarConfig(
        name=name,
        terminals={
            "THINK_START": start_tag,
            "THINK_END": end_tag,
        },
        token_id_terminals={
            "THINK_START": start_tag,
            "THINK_END": end_tag,
        },
        transitions={
            (ParserState.CONTENT, "THINK_START"): Transition(
                ParserState.REASONING,
                [EventType.REASONING_START],
            ),
            (ParserState.REASONING, "THINK_END"): Transition(
                ParserState.CONTENT,
                [EventType.REASONING_END],
            ),
        },
        content_events={
            ParserState.CONTENT: EventType.TEXT_CHUNK,
            ParserState.REASONING: EventType.REASONING_CHUNK,
        },
    )
