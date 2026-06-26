# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.parser.engine.registered_adapters import MinimaxM2ParserToolAdapter
from vllm.parser.minimax_m2 import MinimaxM2Parser


class MinimaxM2ToolParser(MinimaxM2ParserToolAdapter):  # type: ignore[valid-type, misc]
    structural_tag_model = MinimaxM2Parser.structural_tag_model
