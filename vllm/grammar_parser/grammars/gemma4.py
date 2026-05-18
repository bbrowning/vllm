# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Gemma4 tool call argument conversion.

The Gemma4 format uses a custom argument serialization (not JSON):
unquoted keys, ``<|"|>`` string delimiters, and bare values for
numbers/booleans.  The ``_gemma4_arg_converter`` function converts
this to JSON for the OpenAI protocol and is used by
``gemma4_unified_config()``.

Format::

    <|tool_call>call:func_name{key:<|"|>value<|"|>,num:42}<tool_call|>
"""

from __future__ import annotations

import json

from vllm.tool_parsers.gemma4_tool_parser import (
    _parse_gemma4_args,
)

TOOL_CALL_START = "<|tool_call>"
TOOL_CALL_END = "<tool_call|>"


def _gemma4_arg_converter(raw_args: str, partial: bool) -> str:
    """Convert Gemma4 custom arg format to JSON string.

    The raw text is everything between ``{`` and the closing ``}``
    (inclusive of any trailing ``}`` from the format).  We strip the
    trailing ``}`` before parsing.
    """
    text = raw_args.strip()
    if text.endswith("}"):
        text = text[:-1]

    parsed = _parse_gemma4_args(text, partial=partial)
    return json.dumps(parsed, ensure_ascii=False)
