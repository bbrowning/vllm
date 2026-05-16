# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Grammar configuration for Gemma4 channel-based reasoning.

Gemma4 uses ``<|channel>``/``<channel|>`` tokens to delimit reasoning
content. When thinking is enabled, the model outputs::

    <|channel>thought
    ...chain of thought reasoning...<channel|>
    Final answer text here.

The ``thought\\n`` prefix is a structural label that must be stripped.
"""

from __future__ import annotations

from vllm.grammar_parser.grammars.gemma4_tokens import GEMMA4_DROP_TOKENS
from vllm.grammar_parser.grammars.think_tag import think_tag_config


def gemma4_channel_config():
    """Return a grammar config for Gemma4 channel-based reasoning."""
    config = think_tag_config(
        start_tag="<|channel>",
        end_tag="<channel|>",
        name="gemma4_channel",
    )
    config.drop_tokens = GEMMA4_DROP_TOKENS - {"<|channel>", "<channel|>"}
    return config
