# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from vllm.parser.engine.registered_adapters import Glm47MoeParserToolAdapter
from vllm.parser.glm47_moe import Glm47MoeParser


class Glm47MoeModelToolParser(Glm47MoeParserToolAdapter):  # type: ignore[valid-type, misc]
    supports_required_and_named = False
    structural_tag_model = Glm47MoeParser.structural_tag_model
