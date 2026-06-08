# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
)
from vllm.envs import VLLM_ENFORCE_STRICT_TOOL_CALLING
from vllm.parser.engine.registered_adapters import Qwen3XMLParserToolAdapter
from vllm.tool_parsers.structural_tag_registry import (
    get_enable_structured_outputs_in_reasoning,
    get_model_structural_tag,
)


class Qwen3CoderEngineToolParser(Qwen3XMLParserToolAdapter):  # type: ignore[valid-type, misc]
    supports_required_and_named: bool = not VLLM_ENFORCE_STRICT_TOOL_CALLING

    def get_structural_tag(self, request: ChatCompletionRequest):
        return get_model_structural_tag(
            model="qwen_3_5",
            tools=request.tools,
            tool_choice=request.tool_choice,
            reasoning=get_enable_structured_outputs_in_reasoning(),
        )
