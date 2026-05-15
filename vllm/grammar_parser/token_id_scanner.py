# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Scan delta token IDs for special tokens and split the stream into
pre-lexed grammar terminals and plain text chunks."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(slots=True)
class TextChunk:
    """A span of decoded text that needs further lexing."""

    text: str


@dataclass(slots=True)
class PreLexedTerminal:
    """A grammar terminal identified by token ID, already resolved."""

    terminal: str
    token_id: int
    text: str


LexerInput = TextChunk | PreLexedTerminal


class TokenIDScanner:
    """Maps special token IDs in the delta to grammar terminals.

    Before text-based lexing happens, the scanner checks each token ID
    in the delta against a mapping of ``{token_id: terminal_name}``.
    Matched tokens are emitted as :class:`PreLexedTerminal` items;
    everything else is grouped into :class:`TextChunk` items for the
    incremental lexer to process.

    This handles the case where ``skip_special_tokens=True`` strips the
    text representation from ``delta_text`` -- the token ID is still
    present in ``delta_token_ids``.
    """

    def __init__(
        self,
        token_id_to_terminal: dict[int, str],
        tokenizer,
    ) -> None:
        self.token_id_to_terminal = token_id_to_terminal
        self.tokenizer = tokenizer
        self._token_text_cache: dict[int, str] = {}

    def _decode_token(self, token_id: int) -> str:
        if token_id not in self._token_text_cache:
            self._token_text_cache[token_id] = self.tokenizer.decode([token_id])
        return self._token_text_cache[token_id]

    def scan(
        self,
        delta_text: str,
        delta_token_ids: Sequence[int],
    ) -> list[LexerInput]:
        if not self.token_id_to_terminal:
            return [TextChunk(delta_text)] if delta_text else []

        special_positions: list[tuple[int, int, str]] = []
        for idx, tid in enumerate(delta_token_ids):
            terminal = self.token_id_to_terminal.get(tid)
            if terminal is not None:
                special_positions.append((idx, tid, terminal))

        if not special_positions:
            return [TextChunk(delta_text)] if delta_text else []

        token_texts = [self._decode_token(tid) for tid in delta_token_ids]

        results: list[LexerInput] = []
        text_accum: list[str] = []

        for idx, tid in enumerate(delta_token_ids):
            terminal = self.token_id_to_terminal.get(tid)
            if terminal is not None:
                if text_accum:
                    joined = "".join(text_accum)
                    if joined:
                        results.append(TextChunk(joined))
                    text_accum.clear()
                results.append(PreLexedTerminal(terminal, tid, token_texts[idx]))
            else:
                text_accum.append(token_texts[idx])

        if text_accum:
            joined = "".join(text_accum)
            if joined:
                results.append(TextChunk(joined))

        return results
