# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence

from openai.types.responses import ResponseFunctionToolCall

from vllm.entrypoints.chat_utils import ChatCompletionMessageParam
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.responses.protocol import (
    ResponseInputOutputItem,
    ResponsesRequest,
)


def apply_structural_tag(
    request: ChatCompletionRequest | ResponsesRequest,
    structural_tag_model: str | None,
) -> None:
    """Apply a structural tag to *request* for guided tool-call generation.

    This is the single implementation shared by both :class:`ParserEngine`
    (direct/paired path) and :class:`DelegatingParser` (adapter path).
    """
    import vllm.envs as envs

    if structural_tag_model is None or not request.tools:
        return

    if not envs.VLLM_ENFORCE_STRICT_TOOL_CALLING:
        return

    existing = getattr(request, "structured_outputs", None)
    if existing is not None and existing.structural_tag is not None:
        return

    from openai.types.responses import ToolChoiceFunction

    from vllm.entrypoints.openai.chat_completion.protocol import (
        ChatCompletionNamedToolChoiceParam,
    )

    need_tool_calling = (
        request.tool_choice == "auto"
        or request.tool_choice == "required"
        or isinstance(
            request.tool_choice,
            (ChatCompletionNamedToolChoiceParam, ToolChoiceFunction),
        )
    )
    if not need_tool_calling:
        return

    from vllm.tool_parsers.structural_tag_registry import get_model_structural_tag

    structure_tag = get_model_structural_tag(
        model=structural_tag_model,
        tools=request.tools,
        tool_choice=request.tool_choice,
        reasoning=False,
    )
    if structure_tag is None:
        return

    from vllm.sampling_params import StructuredOutputsParams

    structural_tag = json.dumps(structure_tag.model_dump())
    request.structured_outputs = StructuredOutputsParams(
        structural_tag=structural_tag,
    )
    if isinstance(request, ResponsesRequest):
        request.text = None
    else:
        request.response_format = None


def count_tool_calls(tool_calls: object) -> int:
    if tool_calls is None:
        return 0
    if isinstance(tool_calls, (str, bytes, dict)):
        return 1
    if isinstance(tool_calls, Iterable):
        return sum(1 for _ in tool_calls)
    return 1


def count_chat_history_tool_calls(
    messages: Sequence[ChatCompletionMessageParam],
) -> int:
    return sum(
        count_tool_calls(msg.get("tool_calls"))
        for msg in messages
        if isinstance(msg, dict) and msg.get("role") == "assistant"
    )


def count_response_history_tool_calls(
    response_items: Sequence[ResponseInputOutputItem],
) -> int:
    count = 0
    for item in response_items:
        if isinstance(item, ResponseFunctionToolCall):
            count += 1
            continue

        if isinstance(item, dict):
            item_type = item.get("type")
            if item_type == "function_call":
                count += 1
            elif item.get("role") == "assistant":
                count += count_tool_calls(item.get("tool_calls"))

    return count


def count_history_tool_calls(
    request: ChatCompletionRequest | ResponsesRequest,
) -> int:
    if isinstance(request, ChatCompletionRequest):
        return count_chat_history_tool_calls(request.messages)

    request_input = request.input
    if isinstance(request_input, str):
        return 0

    return count_response_history_tool_calls(request_input)
