# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Tool call parser for Google Gemma4 models.

Gemma4 uses a custom serialization format (not JSON) for tool calls::

    <|tool_call>call:func_name{key:<|"|>value<|"|>,num:42}<tool_call|>

Strings are delimited by ``<|"|>`` (token 52), keys are unquoted, and
multiple tool calls are concatenated without separators.

Used when ``--enable-auto-tool-choice --tool-call-parser gemma4`` are set.

For offline inference tool call parsing (direct ``tokenizer.decode()`` output),
see ``vllm.tool_parsers.gemma4_utils.parse_tool_calls``.
"""

from vllm.parser.engine.registered_adapters import Gemma4ParserToolAdapter
from vllm.parser.gemma4 import (
    STRING_DELIM,
    TOOL_CALL_END,
    TOOL_CALL_START,
    _parse_gemma4_args,
    _parse_gemma4_array,
)

__all__ = [
    "Gemma4ToolParser",
    "TOOL_CALL_START",
    "TOOL_CALL_END",
    "STRING_DELIM",
    "_parse_gemma4_args",
    "_parse_gemma4_array",
]

Gemma4ToolParser = Gemma4ParserToolAdapter
