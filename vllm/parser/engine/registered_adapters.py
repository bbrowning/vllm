# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Concrete adapter classes for each registered parser engine.

These are created via :func:`make_adapters` and exposed as module-level
names so that :class:`ReasoningParserManager` and
:class:`ToolParserManager` can load them lazily.
"""

from vllm.parser.engine.adapters import make_adapters
from vllm.parser.engine.parsers.deepseek_v4 import DeepSeekV4Parser
from vllm.parser.engine.parsers.gemma4 import Gemma4Parser
from vllm.parser.engine.parsers.nemotron_v3 import NemotronV3Parser
from vllm.parser.engine.parsers.qwen3 import (
    Qwen3Parser,
    Qwen3XMLParser,
)

(
    DeepSeekV4ParserReasoningAdapter,
    DeepSeekV4ParserToolAdapter,
) = make_adapters(DeepSeekV4Parser)

(
    Gemma4ParserReasoningAdapter,
    Gemma4ParserToolAdapter,
) = make_adapters(Gemma4Parser)

(
    Qwen3ParserReasoningAdapter,
    Qwen3ParserToolAdapter,
) = make_adapters(Qwen3Parser)

(
    NemotronV3ParserReasoningAdapter,
    NemotronV3ParserToolAdapter,
) = make_adapters(NemotronV3Parser)

(
    Qwen3XMLParserReasoningAdapter,
    Qwen3XMLParserToolAdapter,
) = make_adapters(Qwen3XMLParser)
