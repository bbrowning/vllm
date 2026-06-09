# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import contextlib
import json
from collections.abc import Sequence
from typing import TYPE_CHECKING

from vllm.logger import init_logger
from vllm.parser.engine.parser_engine_config import ParserEngineConfig

if TYPE_CHECKING:
    from vllm.entrypoints.openai.engine.protocol import DeltaMessage

logger = init_logger(__name__)


def accumulate_deltas(
    deltas: Sequence[DeltaMessage | None],
) -> dict:
    reasoning_parts: list[str] = []
    content_parts: list[str] = []
    tool_calls_by_idx: dict[int, dict] = {}

    for delta in deltas:
        if delta is None:
            continue
        if delta.reasoning:
            reasoning_parts.append(delta.reasoning)
        if delta.content:
            content_parts.append(delta.content)
        if delta.tool_calls:
            for tc in delta.tool_calls:
                if tc.function and tc.function.name:
                    existing = tool_calls_by_idx.get(tc.index)
                    if existing is None:
                        tool_calls_by_idx[tc.index] = {
                            "name": tc.function.name,
                            "_args_parts": [tc.function.arguments or ""],
                        }
                    else:
                        existing["_args_parts"].append(tc.function.arguments or "")
                elif tc.function and tc.function.arguments:
                    existing = tool_calls_by_idx.get(tc.index)
                    if existing is not None:
                        existing["_args_parts"].append(tc.function.arguments)

    return {
        "reasoning": "".join(reasoning_parts),
        "content": "".join(content_parts),
        "tool_calls": [
            {"name": tc["name"], "arguments": "".join(tc["_args_parts"])}
            for tc in tool_calls_by_idx.values()
        ],
    }


class TokenCapture:
    """Records token IDs and deltas for ``VLLM_DUMP_PARSER_TOKENS``."""

    def __init__(
        self,
        config_name: str,
        dump_path: str,
        parser_engine_config: ParserEngineConfig,
        vocab: dict[str, int],
        tokenizer,
    ) -> None:
        self._config_name = config_name
        self._dump_path = dump_path
        self._parser_engine_config = parser_engine_config
        self._vocab = vocab
        self._tokenizer = tokenizer
        self.tokens: list[list] = []
        self.deltas: list[DeltaMessage] = []
        logger.info("Token capture enabled for %s -> %s", config_name, dump_path)

    def flush(self) -> None:
        if not self.tokens:
            return

        vocab_capture: dict[str, int] = {}
        for text in self._parser_engine_config.token_id_terminals.values():
            tid = self._vocab.get(text)
            if tid is not None:
                vocab_capture[text] = tid

        token_decode_map: dict[int, str] = {}
        for tid_text in self.tokens:
            tid = tid_text[0]
            if tid in token_decode_map:
                continue
            decoded = self._tokenizer.decode([tid])
            token_decode_map[tid] = decoded
            tid_text[1] = decoded

        text = "".join(t[1] for t in self.tokens)
        parsed = self._merge_deltas() if self.deltas else None

        record = {
            "id": f"{self._config_name}-capture-auto",
            "description": "auto-captured from live model run",
            "source": "VLLM_DUMP_PARSER_TOKENS",
            "vocab": vocab_capture,
            "tokens": self.tokens,
            "text": text,
            "parsed": parsed,
        }

        with open(self._dump_path, "a") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        logger.info("Flushed %d tokens to %s", len(self.tokens), self._dump_path)
        self.tokens = []
        self.deltas = []

    def _merge_deltas(self) -> dict:
        result = accumulate_deltas(self.deltas)
        for tc in result["tool_calls"]:
            args_str = tc.get("arguments", "")
            if args_str:
                with contextlib.suppress(json.JSONDecodeError, ValueError):
                    tc["arguments"] = json.loads(args_str)
        reasoning = result["reasoning"] or None
        content_str = result["content"]
        cfg = self._parser_engine_config
        if result["tool_calls"]:
            if cfg.strip_content_whitespace_with_tools:
                content_str = content_str.strip()
            elif (
                cfg.drop_whitespace_only_content_before_tools
                and not content_str.strip()
            ):
                content_str = ""
        content = content_str or None
        return {
            "reasoning": reasoning,
            "content": content,
            "tool_calls": result["tool_calls"],
        }
