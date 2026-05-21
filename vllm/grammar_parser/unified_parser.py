# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unified grammar parser that handles both reasoning and tool call
extraction with a single :class:`StreamingParserEngine`.

Unlike the adapter-based approach (``GrammarReasoningParser`` +
``GrammarToolParser`` composed via ``DelegatingParser``), this class
uses one ``GrammarConfig`` state machine that covers the complete model
output format.  This eliminates the delta_text / delta_token_ids
mismatches that arise when two engines process different phases of the
same token stream.
"""

from __future__ import annotations

import json
import os
from collections.abc import Sequence
from functools import cached_property
from typing import TYPE_CHECKING

import regex as re

from vllm.entrypoints.chat_utils import make_tool_call_id
from vllm.entrypoints.openai.engine.protocol import (
    DeltaFunctionCall,
    DeltaMessage,
    DeltaToolCall,
    ExtractedToolCallInformation,
    FunctionCall,
    ToolCall,
)
from vllm.grammar_parser.events import EventType, SemanticEvent
from vllm.grammar_parser.grammar_config import GrammarConfig, ParserState
from vllm.grammar_parser.parser_engine import StreamingParserEngine
from vllm.logger import init_logger
from vllm.parser.abstract_parser import Parser, StreamState
from vllm.tool_parsers.utils import find_tool_properties

if TYPE_CHECKING:
    from vllm.entrypoints.openai.chat_completion.protocol import (
        ChatCompletionRequest,
    )
    from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
    from vllm.tokenizers import TokenizerLike
    from vllm.tool_parsers.abstract_tool_parser import Tool

logger = init_logger(__name__)

_DUMP_PATH = os.environ.get("VLLM_DUMP_PARSER_TOKENS")


class GrammarParser(Parser):
    """A :class:`Parser` backed by a single declarative grammar config.

    Subclasses set the ``GrammarConfig`` in ``__init__`` to define the
    complete output format for a model (reasoning + tool calls).
    """

    def __init__(
        self,
        tokenizer: TokenizerLike,
        tools: list[Tool] | None = None,
        *,
        grammar_config: GrammarConfig,
        **kwargs,
    ) -> None:
        super().__init__(tokenizer)
        self._tools = tools
        self.grammar_config = grammar_config
        self._engine = StreamingParserEngine(grammar_config, tokenizer)

        self._reasoning_ended: bool = False

        self._tool_call_ids: list[str] = []
        self._tool_names: list[str] = []
        self._tool_args: list[str] = []
        self._name_sent: list[bool] = []
        self._streamed_json: list[str] = []

        self._capture_tokens: list[list] | None = [] if _DUMP_PATH else None
        if self._capture_tokens is not None:
            logger.info(
                "Token capture enabled for %s -> %s",
                grammar_config.name,
                _DUMP_PATH,
            )

        vocab = self.vocab
        self._reasoning_start_token_id: int | None = None
        self._reasoning_end_token_id: int | None = None

        start_text = grammar_config.token_id_terminals.get("THINK_START")
        end_text = grammar_config.token_id_terminals.get("THINK_END")
        if start_text:
            self._reasoning_start_token_id = vocab.get(start_text)
        if end_text:
            self._reasoning_end_token_id = vocab.get(end_text)

    @property
    def reasoning_start_str(self) -> str | None:
        return self.grammar_config.terminals.get("THINK_START")

    @property
    def reasoning_end_str(self) -> str | None:
        return self.grammar_config.terminals.get("THINK_END")

    @cached_property
    def vocab(self) -> dict[str, int]:
        return self.model_tokenizer.get_vocab()

    def __del__(self) -> None:
        self.flush_capture()

    # ── Engine lifecycle ──────────────────────────────────────────────

    def _reset(self, initial_state: ParserState | None = None) -> None:
        self.flush_capture()
        self._engine = StreamingParserEngine(
            self.grammar_config,
            self.model_tokenizer,
            initial_state=initial_state,
        )
        self._reasoning_ended = False
        self._tool_call_ids.clear()
        self._tool_names.clear()
        self._tool_args.clear()
        self._name_sent.clear()
        self._streamed_json.clear()
        self._stream_state = StreamState()

    def adjust_request(
        self, request: ChatCompletionRequest | ResponsesRequest
    ) -> ChatCompletionRequest | ResponsesRequest:
        request.skip_special_tokens = False
        logger.info("adjust_request: skip_special_tokens set to False")
        return request

    # ── Schema-aware type correction ─────────────────────────────────

    def _fix_arg_types(self, args_json: str, func_name: str) -> str:
        """Correct parameter types wrongly coerced by the arg_converter.

        The arg_converter may blindly coerce e.g. ``"1"`` to ``1``.  If the
        tool schema declares the parameter as ``"string"``, revert to string.
        """
        if not self._tools or not func_name:
            return args_json
        try:
            args = json.loads(args_json)
        except (json.JSONDecodeError, ValueError):
            return args_json
        if not isinstance(args, dict):
            return args_json

        properties = find_tool_properties(self._tools, func_name)
        if not properties:
            return args_json

        changed = False
        for key, value in args.items():
            if isinstance(value, str):
                continue
            prop = properties.get(key)
            if not isinstance(prop, dict) or prop.get("type") != "string":
                continue
            if isinstance(value, bool):
                args[key] = "true" if value else "false"
            elif value is None:
                args[key] = "null"
            else:
                args[key] = str(value)
            changed = True

        if changed:
            return json.dumps(args, ensure_ascii=False)
        return args_json

    # ── Streaming: parse_delta ────────────────────────────────────────

    def parse_delta(
        self,
        delta_text: str,
        delta_token_ids: list[int],
        request: ChatCompletionRequest | ResponsesRequest,
        prompt_token_ids: list[int] | None = None,
        finished: bool = False,
    ) -> DeltaMessage | None:
        if self._capture_tokens is not None:
            for tid in delta_token_ids:
                self._capture_tokens.append([tid, ""])
        events = self._engine.feed(delta_text, delta_token_ids)
        if finished:
            events.extend(self._engine.finish())
        return self._events_to_delta(events)

    def flush_capture(self) -> None:
        """Write captured token sequence to the dump file.

        Call at end of stream to write the accumulated tokens and
        parse result to the JSONL file specified by
        ``VLLM_DUMP_PARSER_TOKENS``.
        """
        if _DUMP_PATH is None or self._capture_tokens is None:
            return
        if not self._capture_tokens:
            return

        vocab_capture: dict[str, int] = {}
        full_vocab = self.vocab
        for text in self.grammar_config.token_id_terminals.values():
            tid = full_vocab.get(text)
            if tid is not None:
                vocab_capture[text] = tid

        token_decode_map: dict[int, str] = {}
        for tid_text in self._capture_tokens:
            tid = tid_text[0]
            if tid in token_decode_map:
                continue
            decoded = self.model_tokenizer.decode([tid])
            token_decode_map[tid] = decoded
            tid_text[1] = decoded

        for tid_text in self._capture_tokens:
            tid_text[1] = token_decode_map.get(tid_text[0], "")

        record = {
            "id": f"{self.grammar_config.name}-capture-auto",
            "description": "auto-captured from live model run",
            "source": "VLLM_DUMP_PARSER_TOKENS",
            "vocab": vocab_capture,
            "tokens": self._capture_tokens,
        }

        with open(_DUMP_PATH, "a") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        logger.info(
            "Flushed %d tokens to %s",
            len(self._capture_tokens),
            _DUMP_PATH,
        )
        self._capture_tokens = []

    # ── Non-streaming: extract_reasoning ──────────────────────────────

    def extract_reasoning(
        self,
        model_output: str,
        request: ChatCompletionRequest | ResponsesRequest,
    ) -> tuple[str | None, str | None]:
        self._reset()
        events = self._engine.feed(model_output, [])
        events.extend(self._engine.finish())

        reasoning_parts: list[str] = []
        content_parts: list[str] = []

        for event in events:
            if event.type == EventType.REASONING_CHUNK:
                reasoning_parts.append(event.value)
            elif event.type == EventType.TEXT_CHUNK:
                content_parts.append(event.value)
            elif event.type == EventType.REASONING_END:
                self._reasoning_ended = True

        reasoning = "".join(reasoning_parts) or None
        content = "".join(content_parts) or None
        return reasoning, content

    # ── Non-streaming: extract_reasoning_streaming ────────────────────

    def extract_reasoning_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
    ) -> DeltaMessage | None:
        events = self._engine.feed(delta_text, delta_token_ids)
        return self._events_to_delta(events)

    # ── Non-streaming: extract_tool_calls ─────────────────────────────

    def extract_tool_calls(
        self,
        model_output: str,
        request: ChatCompletionRequest,
    ) -> ExtractedToolCallInformation:
        self._reset()
        result = self.extract_tool_calls_streaming(
            previous_text="",
            current_text=model_output,
            delta_text=model_output,
            previous_token_ids=[],
            current_token_ids=[],
            delta_token_ids=[],
            request=request,
        )
        finish_events = self._engine.finish()
        finish_delta = self._events_to_delta(finish_events) if finish_events else None
        return self._build_extracted_result(result, finish_delta)

    def extract_tool_calls_from_content(
        self,
        content: str,
        request: ChatCompletionRequest,
    ) -> ExtractedToolCallInformation:
        """Extract tool calls from reasoning-stripped content.

        Unlike :meth:`extract_tool_calls` which re-parses the full model
        output, this method starts the grammar engine in ``CONTENT`` state
        so it can parse content that has already had reasoning stripped.
        """
        self._reset(initial_state=ParserState.CONTENT)
        events = self._engine.feed(content, [])
        result = self._events_to_delta(events)
        finish_events = self._engine.finish()
        finish_delta = self._events_to_delta(finish_events) if finish_events else None
        return self._build_extracted_result(result, finish_delta)

    def extract_tool_calls_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
        request: ChatCompletionRequest,
    ) -> DeltaMessage | None:
        if not previous_text:
            self._reset()
        events = self._engine.feed(delta_text, delta_token_ids)
        return self._events_to_delta(events)

    # ── Reasoning state queries ───────────────────────────────────────

    def is_reasoning_end(self, input_ids: list[int]) -> bool:
        end_id = self._reasoning_end_token_id
        start_id = self._reasoning_start_token_id
        if end_id is not None:
            for i in range(len(input_ids) - 1, -1, -1):
                if input_ids[i] == end_id:
                    return True
                if start_id is not None and input_ids[i] == start_id:
                    return False
            return False
        return self._reasoning_ended

    def extract_content_ids(self, input_ids: list[int]) -> list[int]:
        end_id = self._reasoning_end_token_id
        if end_id is not None:
            for i in range(len(input_ids) - 1, -1, -1):
                if input_ids[i] == end_id:
                    return input_ids[i + 1 :]
            return input_ids
        if self._reasoning_ended:
            return []
        return input_ids

    # ── Response outputs (Responses API) ──────────────────────────────

    def extract_response_outputs(
        self,
        *,
        model_output: str,
        model_output_token_ids: Sequence[int],
        request: ResponsesRequest,
        enable_auto_tools: bool = False,
        tool_call_id_type: str = "random",
        logprobs=None,
    ) -> list:
        from openai.types.responses import (
            ResponseFunctionToolCall,
            ResponseOutputMessage,
            ResponseOutputText,
        )
        from openai.types.responses.response_output_item import ResponseOutputItem
        from openai.types.responses.response_reasoning_item import (
            Content as ResponseReasoningTextContent,
        )
        from openai.types.responses.response_reasoning_item import (
            ResponseReasoningItem,
        )

        from vllm.utils import random_uuid

        # Single-pass parse with token IDs so the engine can
        # distinguish real special tokens from text that happens
        # to look like them (e.g. "<tool_call>" in source code).
        self._reset()
        events = self._engine.feed(model_output, model_output_token_ids)
        events.extend(self._engine.finish())

        delta = self._events_to_delta(events)
        tool_call_info = self._build_extracted_result()

        reasoning = delta.reasoning if delta else None
        content = delta.content if delta else None
        if tool_call_info.tools_called and content:
            content = content.strip() or None

        outputs: list[ResponseOutputItem] = []

        if reasoning:
            outputs.append(
                ResponseReasoningItem(
                    id=f"rs_{random_uuid()}",
                    summary=[],
                    type="reasoning",
                    content=[
                        ResponseReasoningTextContent(
                            text=reasoning, type="reasoning_text"
                        )
                    ],
                    status=None,
                )
            )

        if content:
            outputs.append(
                ResponseOutputMessage(
                    id=f"msg_{random_uuid()}",
                    content=[
                        ResponseOutputText(
                            text=content,
                            annotations=[],
                            type="output_text",
                            logprobs=logprobs,
                        )
                    ],
                    role="assistant",
                    status="completed",
                    type="message",
                )
            )

        if tool_call_info.tools_called:
            for i, tc in enumerate(tool_call_info.tool_calls):
                outputs.append(
                    ResponseFunctionToolCall(
                        id=f"fc_{random_uuid()}",
                        call_id=tc.id
                        if tc.id
                        else make_tool_call_id(
                            id_type=tool_call_id_type,
                            func_name=tc.function.name,
                            idx=i,
                        ),
                        type="function_call",
                        status="completed",
                        name=tc.function.name,
                        arguments=tc.function.arguments,
                    )
                )

        return outputs

    # ── Event-to-delta conversion ─────────────────────────────────────

    def _events_to_delta(
        self,
        events: list[SemanticEvent],
    ) -> DeltaMessage | None:
        tool_call_deltas: list[DeltaToolCall] = []
        content_parts: list[str] = []
        reasoning_parts: list[str] = []

        for event in events:
            match event.type:
                case EventType.TEXT_CHUNK:
                    content_parts.append(event.value)
                case EventType.REASONING_CHUNK:
                    reasoning_parts.append(event.value)
                case EventType.REASONING_END:
                    self._reasoning_ended = True
                case EventType.TOOL_CALL_START:
                    self._init_tool_slot(event)
                case EventType.TOOL_NAME:
                    self._handle_tool_name(event)
                case EventType.ARG_VALUE_CHUNK:
                    self._handle_arg_chunk(event, tool_call_deltas)
                case EventType.TOOL_CALL_END:
                    self._handle_tool_end(event, tool_call_deltas)

        content = "".join(content_parts) or None
        reasoning = "".join(reasoning_parts) or None

        if content or tool_call_deltas or reasoning:
            return DeltaMessage(
                content=content,
                reasoning=reasoning,
                tool_calls=tool_call_deltas,
            )
        return None

    def _ensure_slot(self, idx: int) -> None:
        """Ensure per-tool-call lists have room for *idx*."""
        while len(self._tool_call_ids) <= idx:
            self._tool_call_ids.append("")
            self._tool_names.append("")
            self._tool_args.append("")
            self._name_sent.append(False)
            self._streamed_json.append("")

    def _init_tool_slot(self, event: SemanticEvent) -> None:
        """Initialize a slot for a new tool call."""
        idx = event.tool_index
        self._ensure_slot(idx)
        self._tool_call_ids[idx] = make_tool_call_id()

    def _handle_tool_name(self, event: SemanticEvent) -> None:
        idx = event.tool_index
        self._tool_names[idx] += event.value

    def _emit_name_delta(
        self,
        idx: int,
        deltas: list[DeltaToolCall],
        name: str | None,
    ) -> None:
        """Emit a name delta and mark the slot as name-sent."""
        if not name:
            return
        self._tool_names[idx] = name
        self._name_sent[idx] = True
        deltas.append(
            DeltaToolCall(
                index=idx,
                id=self._tool_call_ids[idx],
                type="function",
                function=DeltaFunctionCall(name=name),
            )
        )

    def _handle_arg_chunk(
        self,
        event: SemanticEvent,
        deltas: list[DeltaToolCall],
    ) -> None:
        """Accumulate args and emit name/arg deltas as needed."""
        idx = event.tool_index
        if event.value:
            self._tool_args[idx] += event.value

        if not self._name_sent[idx]:
            if self._tool_names[idx]:
                self._emit_name_delta(idx, deltas, self._tool_names[idx])
            elif event.value:
                # Name not yet known — try to extract from accumulated args
                name = self._try_extract_name(idx)
                self._emit_name_delta(idx, deltas, name)
        elif event.value:
            # Name already sent — emit arg delta
            arg_delta = self._compute_arg_delta(idx, event.value)
            if arg_delta:
                deltas.append(
                    DeltaToolCall(
                        index=idx,
                        function=DeltaFunctionCall(arguments=arg_delta),
                    )
                )

    def _handle_tool_end(
        self,
        event: SemanticEvent,
        deltas: list[DeltaToolCall],
    ) -> None:
        """Flush arg converter and emit final delta for completed tool."""
        idx = event.tool_index
        if idx >= len(self._tool_args):
            return

        remaining = self._flush_arg_converter(idx)

        if not self._name_sent[idx]:
            name = self._tool_names[idx] or self._try_extract_name(idx)
            if name:
                self._tool_names[idx] = name
                self._name_sent[idx] = True
                deltas.append(
                    DeltaToolCall(
                        index=idx,
                        id=self._tool_call_ids[idx],
                        type="function",
                        function=DeltaFunctionCall(
                            name=name,
                            arguments=remaining or "",
                        ),
                    )
                )
                remaining = None

        if remaining and self._name_sent[idx]:
            deltas.append(
                DeltaToolCall(
                    index=idx,
                    function=DeltaFunctionCall(arguments=remaining),
                )
            )

    # ── Arg conversion helpers (from GrammarToolParser) ───────────────

    def _compute_arg_delta(self, idx: int, raw_delta: str) -> str | None:
        converter = self.grammar_config.arg_converter
        if converter is None:
            return raw_delta

        if not self.grammar_config.strip_trailing_quotes:
            return None

        structural = self.grammar_config.arg_structural_chars
        if structural is not None and structural.isdisjoint(raw_delta):
            return None

        accumulated = self._tool_args[idx]
        try:
            current_json = converter(accumulated, True)
        except Exception:
            return None

        if not current_json:
            return None

        prev = self._streamed_json[idx]
        safe_json = current_json
        while safe_json and safe_json[-1] in ("}", '"', "]"):
            safe_json = safe_json[:-1]

        if not safe_json or safe_json == prev:
            return None

        if prev:
            if not safe_json.startswith(prev):
                return None
            diff = safe_json[len(prev) :]
        else:
            diff = safe_json

        if diff:
            self._streamed_json[idx] = safe_json
            return diff
        return None

    def _flush_arg_converter(self, idx: int) -> str | None:
        converter = self.grammar_config.arg_converter
        if converter is None:
            return None

        accumulated = self._tool_args[idx]
        try:
            final_json = converter(accumulated, False)
        except Exception:
            return None

        if final_json and idx < len(self._tool_names):
            final_json = self._fix_arg_types(final_json, self._tool_names[idx])

        prev = self._streamed_json[idx]
        if final_json and len(final_json) > len(prev):
            diff = final_json[len(prev) :]
            self._streamed_json[idx] = final_json
            return diff
        return None

    _NAME_RE = re.compile(r'"name"\s*:\s*"([^"]*)"')

    def _try_extract_name(self, idx: int) -> str | None:
        accumulated = self._tool_args[idx]
        m = self._NAME_RE.search(accumulated)
        if m:
            name = m.group(1)
            if name:
                return name
        return None

    # ── Build ExtractedToolCallInformation ─────────────────────────────

    def _build_extracted_result(
        self,
        *deltas: DeltaMessage | None,
    ) -> ExtractedToolCallInformation:
        content_parts: list[str] = []
        for delta in deltas:
            if delta is not None and delta.content:
                content_parts.append(delta.content)

        tool_calls: list[ToolCall] = []
        for idx in range(len(self._tool_call_ids)):
            if not self._tool_call_ids[idx]:
                continue

            name = self._tool_names[idx]
            raw_body = self._tool_args[idx]

            if not name and raw_body.strip():
                name, args_json = self._extract_name_and_args(raw_body)
            elif raw_body.strip():
                converter = self.grammar_config.arg_converter
                if converter is not None:
                    try:
                        args_json = converter(raw_body, False)
                    except Exception:
                        args_json = self._extract_args_json(raw_body, name)
                else:
                    args_json = self._extract_args_json(raw_body, name)
            else:
                args_json = "{}"

            if name:
                args_json = self._fix_arg_types(args_json, name)
                tool_calls.append(
                    ToolCall(
                        id=self._tool_call_ids[idx],
                        function=FunctionCall(name=name, arguments=args_json),
                    )
                )

        content_str = "".join(content_parts)
        if tool_calls:
            content_str = content_str.strip()
        content = content_str or None

        return ExtractedToolCallInformation(
            tools_called=len(tool_calls) > 0,
            tool_calls=tool_calls,
            content=content,
        )

    @staticmethod
    def _extract_args_value(parsed: dict) -> str | None:
        for key in ("arguments", "parameters"):
            if key in parsed:
                val = parsed[key]
                if isinstance(val, str):
                    return val
                return json.dumps(val, ensure_ascii=False)
        return None

    def _extract_name_and_args(
        self,
        raw_body: str,
    ) -> tuple[str, str]:
        raw_body = raw_body.strip()
        try:
            parsed = json.loads(raw_body)
        except json.JSONDecodeError:
            return "", raw_body

        if not isinstance(parsed, dict):
            return "", raw_body

        name = parsed.get("name", "")
        args = self._extract_args_value(parsed)
        if args is not None:
            return name, args

        without_name = {k: v for k, v in parsed.items() if k != "name"}
        return name, json.dumps(without_name, ensure_ascii=False)

    def _extract_args_json(self, raw_args: str, func_name: str) -> str:
        raw_args = raw_args.strip()
        if not raw_args:
            return "{}"

        try:
            parsed = json.loads(raw_args)
        except json.JSONDecodeError:
            if self.grammar_config.value_postprocessor:
                return self.grammar_config.value_postprocessor(raw_args)
            return raw_args

        if isinstance(parsed, dict):
            args = self._extract_args_value(parsed)
            if args is not None:
                return args
            if "name" in parsed:
                without_name = {k: v for k, v in parsed.items() if k != "name"}
                return json.dumps(without_name, ensure_ascii=False)

        return raw_args
