# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Gemma4 special tokens that should be silently dropped by the scanner.

When ``skip_special_tokens=False`` (required so the grammar parser can
see boundary tokens like ``<|channel>`` and ``<|tool_call>``), *all*
special tokens become visible in the detokenized text.  Tokens that
are not grammar terminals would leak into client-facing content.

Each grammar config subtracts the terminals it actually uses before
passing the set to :pyattr:`GrammarConfig.drop_tokens`.
"""

GEMMA4_DROP_TOKENS: set[str] = {
    # Structural
    "<eos>",
    "<bos>",
    "<pad>",
    "<unk>",
    "<mask>",
    # Turn boundaries
    "<|turn>",
    "<turn|>",
    # Channel / reasoning
    "<|channel>",
    "<channel|>",
    # Tool calling
    "<|tool>",
    "<tool|>",
    "<|tool_call>",
    "<tool_call|>",
    "<|tool_response>",
    "<tool_response|>",
    '<|"|>',
    # Thinking
    "<|think|>",
    # Multi-modal
    "<|image>",
    "<|image|>",
    "<image|>",
    "<|audio>",
    "<|audio|>",
    "<audio|>",
    "<|video|>",
}
