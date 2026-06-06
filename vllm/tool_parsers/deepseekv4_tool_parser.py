# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
)
from vllm.parser.engine.registered_adapters import DeepSeekV4ParserToolAdapter
from vllm.tool_parsers.structural_tag_registry import (
    get_enable_structured_outputs_in_reasoning,
    get_model_structural_tag,
)


class DeepSeekV4ToolParser(DeepSeekV4ParserToolAdapter):  # type: ignore[valid-type, misc]
    """DeepSeek V4 DSML tool parser.

    V4 keeps the V3.2 DSML invoke/parameter grammar, but wraps tool calls in
    ``<｜DSML｜tool_calls>`` instead of ``<｜DSML｜function_calls>``.
    """

    def get_structural_tag(self, request: ChatCompletionRequest):
        return get_model_structural_tag(
            model="deepseek_v4",
            tools=request.tools,
            tool_choice=request.tool_choice,
            reasoning=get_enable_structured_outputs_in_reasoning(),
        )
