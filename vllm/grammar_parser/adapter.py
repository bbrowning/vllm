# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Adapter classes that wrap :class:`StreamingParserEngine` behind the
existing :class:`ToolParser` and :class:`ReasoningParser` interfaces."""

from __future__ import annotations

import json
from collections.abc import Sequence
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
from vllm.grammar_parser.grammar_config import GrammarConfig
from vllm.grammar_parser.parser_engine import StreamingParserEngine
from vllm.reasoning.abs_reasoning_parsers import ReasoningParser
from vllm.tool_parsers.abstract_tool_parser import ToolParser

if TYPE_CHECKING:
    from vllm.entrypoints.openai.chat_completion.protocol import (
        ChatCompletionRequest,
    )
    from vllm.tokenizers import TokenizerLike
    from vllm.tool_parsers.abstract_tool_parser import Tool


class GrammarToolParser(ToolParser):
    """A :class:`ToolParser` backed by a declarative grammar config."""

    def __init__(
        self,
        tokenizer: TokenizerLike,
        tools: list[Tool] | None = None,
        *,
        grammar_config: GrammarConfig,
    ) -> None:
        super().__init__(tokenizer, tools)
        self.grammar_config = grammar_config
        self._engine = StreamingParserEngine(grammar_config, tokenizer)

        self._tool_call_ids: list[str] = []
        self._tool_names: list[str] = []
        self._tool_args: list[str] = []
        self._name_sent: list[bool] = []
        self._streamed_json: list[str] = []

    def extract_tool_calls(
        self,
        model_output: str,
        request: ChatCompletionRequest,
    ) -> ExtractedToolCallInformation:
        engine = StreamingParserEngine(
            self.grammar_config,
            self.model_tokenizer,
        )
        events = engine.parse_complete(model_output)
        return self._events_to_extracted(events)

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
            self._reset_streaming_state()
        events = self._engine.feed(delta_text, delta_token_ids)
        return self._events_to_delta(events)

    def _reset_streaming_state(self) -> None:
        """Reset all streaming state for a new request."""
        self._engine = StreamingParserEngine(
            self.grammar_config,
            self.model_tokenizer,
        )
        self._tool_call_ids.clear()
        self._tool_names.clear()
        self._tool_args.clear()
        self._name_sent.clear()
        self._streamed_json.clear()

    def _events_to_extracted(
        self,
        events: list[SemanticEvent],
    ) -> ExtractedToolCallInformation:
        names: dict[int, str] = {}
        args: dict[int, list[str]] = {}
        content_parts: list[str] = []
        tool_indices: set[int] = set()

        for event in events:
            if event.type == EventType.TOOL_CALL_START:
                tool_indices.add(event.tool_index)
            elif event.type == EventType.TOOL_NAME:
                names[event.tool_index] = event.value
            elif event.type == EventType.ARG_VALUE_CHUNK:
                args.setdefault(event.tool_index, []).append(event.value)
                tool_indices.add(event.tool_index)
            elif event.type == EventType.TEXT_CHUNK:
                content_parts.append(event.value)

        tool_calls: list[ToolCall] = []
        for idx in sorted(tool_indices):
            raw_body = "".join(args.get(idx, []))
            name = names.get(idx, "")
            args_json = "{}"

            if not name and raw_body.strip():
                name, args_json = self._extract_name_and_args(raw_body)
            elif raw_body.strip():
                converter = self.grammar_config.arg_converter
                if converter is not None:
                    try:
                        args_json = converter(raw_body, False)
                    except Exception:
                        args_json = self._extract_args_json(
                            raw_body,
                            name,
                        )
                else:
                    args_json = self._extract_args_json(raw_body, name)

            if name:
                tool_calls.append(
                    ToolCall(
                        id=make_tool_call_id(),
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

    def _extract_name_and_args(
        self,
        raw_body: str,
    ) -> tuple[str, str]:
        """Extract function name and arguments from a JSON body like
        ``{"name": "func", "arguments": {...}}``."""
        raw_body = raw_body.strip()
        try:
            parsed = json.loads(raw_body)
        except json.JSONDecodeError:
            return "", raw_body

        if not isinstance(parsed, dict):
            return "", raw_body

        name = parsed.get("name", "")

        for key in ("arguments", "parameters"):
            if key in parsed:
                val = parsed[key]
                if isinstance(val, str):
                    return name, val
                return name, json.dumps(val, ensure_ascii=False)

        without_name = {k: v for k, v in parsed.items() if k != "name"}
        return name, json.dumps(without_name, ensure_ascii=False)

    def _extract_args_json(self, raw_args: str, func_name: str) -> str:
        """Extract arguments JSON from the raw event text.

        For configs where the full JSON body includes ``"name"`` and
        ``"arguments"`` keys, this pulls out the arguments value.
        Otherwise returns the raw text as-is.
        """
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
            if "arguments" in parsed:
                args_val = parsed["arguments"]
                if isinstance(args_val, str):
                    return args_val
                return json.dumps(args_val, ensure_ascii=False)
            if "parameters" in parsed:
                params_val = parsed["parameters"]
                if isinstance(params_val, str):
                    return params_val
                return json.dumps(params_val, ensure_ascii=False)
            if "name" in parsed:
                without_name = {k: v for k, v in parsed.items() if k != "name"}
                return json.dumps(without_name, ensure_ascii=False)

        return raw_args

    def _events_to_delta(
        self,
        events: list[SemanticEvent],
    ) -> DeltaMessage | None:
        tool_call_deltas: list[DeltaToolCall] = []
        content_parts: list[str] = []
        reasoning_parts: list[str] = []

        for event in events:
            if event.type == EventType.TEXT_CHUNK:
                content_parts.append(event.value)

            elif event.type == EventType.TOOL_CALL_START:
                idx = event.tool_index
                call_id = make_tool_call_id()
                while len(self._tool_call_ids) <= idx:
                    self._tool_call_ids.append("")
                    self._tool_names.append("")
                    self._tool_args.append("")
                    self._name_sent.append(False)
                    self._streamed_json.append("")
                self._tool_call_ids[idx] = call_id

            elif event.type == EventType.TOOL_NAME:
                idx = event.tool_index
                self._tool_names[idx] += event.value

            elif event.type == EventType.ARG_VALUE_CHUNK:
                idx = event.tool_index
                if event.value:
                    self._tool_args[idx] += event.value

                if not self._name_sent[idx] and self._tool_names[idx]:
                    self._name_sent[idx] = True
                    tool_call_deltas.append(
                        DeltaToolCall(
                            index=idx,
                            id=self._tool_call_ids[idx],
                            type="function",
                            function=DeltaFunctionCall(
                                name=self._tool_names[idx],
                            ),
                        )
                    )
                elif not self._name_sent[idx] and event.value:
                    name = self._try_extract_name(idx)
                    if name:
                        self._tool_names[idx] = name
                        self._name_sent[idx] = True
                        tool_call_deltas.append(
                            DeltaToolCall(
                                index=idx,
                                id=self._tool_call_ids[idx],
                                type="function",
                                function=DeltaFunctionCall(name=name),
                            )
                        )
                elif self._name_sent[idx] and event.value:
                    arg_delta = self._compute_arg_delta(
                        idx,
                        event.value,
                    )
                    if arg_delta:
                        tool_call_deltas.append(
                            DeltaToolCall(
                                index=idx,
                                function=DeltaFunctionCall(
                                    arguments=arg_delta,
                                ),
                            )
                        )

            elif event.type == EventType.TOOL_CALL_END:
                idx = event.tool_index
                if idx < len(self._tool_args):
                    remaining = self._flush_arg_converter(idx)
                    if not self._name_sent[idx]:
                        name = self._tool_names[idx] or self._try_extract_name(idx)
                        if name:
                            self._tool_names[idx] = name
                            self._name_sent[idx] = True
                            tool_call_deltas.append(
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
                        tool_call_deltas.append(
                            DeltaToolCall(
                                index=idx,
                                function=DeltaFunctionCall(
                                    arguments=remaining,
                                ),
                            )
                        )

            elif event.type == EventType.REASONING_CHUNK:
                reasoning_parts.append(event.value)

        content = "".join(content_parts) or None
        reasoning = "".join(reasoning_parts) or None

        if content or tool_call_deltas or reasoning:
            return DeltaMessage(
                content=content,
                reasoning=reasoning,
                tool_calls=tool_call_deltas,
            )
        return None

    def _compute_arg_delta(
        self,
        idx: int,
        raw_delta: str,
    ) -> str | None:
        """Compute the argument delta to stream to the client.

        When ``arg_converter`` is set (custom format like Gemma4), we
        accumulate raw text, convert to JSON, and diff against previously
        streamed JSON.  Otherwise, pass through directly.
        """
        converter = self.grammar_config.arg_converter
        if converter is None:
            return raw_delta

        if not self.grammar_config.strip_trailing_quotes:
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
        """Final flush for arg_converter: emit any remaining JSON."""
        converter = self.grammar_config.arg_converter
        if converter is None:
            return None

        accumulated = self._tool_args[idx]
        try:
            final_json = converter(accumulated, False)
        except Exception:
            return None

        prev = self._streamed_json[idx]
        if final_json and len(final_json) > len(prev):
            diff = final_json[len(prev) :]
            self._streamed_json[idx] = final_json
            return diff
        return None

    _NAME_RE = re.compile(r'"name"\s*:\s*"([^"]*)"')

    def _try_extract_name(self, idx: int) -> str | None:
        """Try to extract function name from accumulated JSON args.

        Uses regex to only match when the complete quoted name value is
        present, avoiding partial names from incomplete streaming.
        """
        accumulated = self._tool_args[idx]
        m = self._NAME_RE.search(accumulated)
        if m:
            name = m.group(1)
            if name:
                return name
        return None


class GrammarReasoningParser(ReasoningParser):
    """A :class:`ReasoningParser` backed by a declarative grammar config."""

    def __init__(
        self,
        tokenizer: TokenizerLike,
        *,
        grammar_config: GrammarConfig,
        **kwargs,
    ) -> None:
        super().__init__(tokenizer, **kwargs)
        self.grammar_config = grammar_config
        self._engine = StreamingParserEngine(grammar_config, tokenizer)
        self._reasoning_ended: bool = False

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
        return self.grammar_config.token_id_terminals.get("THINK_START")

    @property
    def reasoning_end_str(self) -> str | None:
        return self.grammar_config.token_id_terminals.get("THINK_END")

    def is_reasoning_end(self, input_ids: Sequence[int]) -> bool:
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

    def extract_reasoning(
        self,
        model_output: str,
        request,
    ) -> tuple[str | None, str | None]:
        engine = StreamingParserEngine(
            self.grammar_config,
            self.model_tokenizer,
        )
        events = engine.parse_complete(model_output)

        reasoning_parts: list[str] = []
        content_parts: list[str] = []
        for event in events:
            if event.type == EventType.REASONING_CHUNK:
                reasoning_parts.append(event.value)
            elif event.type == EventType.TEXT_CHUNK:
                content_parts.append(event.value)

        reasoning = "".join(reasoning_parts) or None
        content = "".join(content_parts) or None
        return reasoning, content

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

        if reasoning or content:
            return DeltaMessage(reasoning=reasoning, content=content)
        return None
