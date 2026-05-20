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

    When a terminal's text is not yet in ``delta_text`` (held back by
    the detokenizer), the terminal is deferred until the text arrives
    in a subsequent delta.
    """

    def __init__(
        self,
        token_id_to_terminal: dict[int, str],
        tokenizer,
        drop_token_ids: set[int] | None = None,
    ) -> None:
        self.token_id_to_terminal = token_id_to_terminal
        self.tokenizer = tokenizer
        self._token_text_cache: dict[int, str] = {}
        self._drop_token_ids = drop_token_ids or set()
        self._deferred_terminals: list[PreLexedTerminal] = []
        self._deferred_post_text: str = ""

    def _decode_token(self, token_id: int) -> str:
        if token_id not in self._token_text_cache:
            self._token_text_cache[token_id] = self.tokenizer.decode([token_id])
        return self._token_text_cache[token_id]

    def scan(
        self,
        delta_text: str,
        delta_token_ids: Sequence[int],
    ) -> list[LexerInput]:
        prefix_items: list[LexerInput] = []
        effective_text = delta_text

        if self._deferred_terminals:
            prefix_items, effective_text = self._resolve_deferred(delta_text)

        if not self.token_id_to_terminal and not self._drop_token_ids:
            if effective_text:
                prefix_items.append(TextChunk(effective_text))
            return prefix_items

        has_special = False
        has_drop = False
        for tid in delta_token_ids:
            if tid in self.token_id_to_terminal:
                has_special = True
            if tid in self._drop_token_ids:
                has_drop = True

        if not has_special and not has_drop:
            if effective_text:
                prefix_items.append(TextChunk(effective_text))
            return prefix_items

        token_texts = [self._decode_token(tid) for tid in delta_token_ids]

        results: list[LexerInput] = []
        text_accum: list[str] = []

        for idx, tid in enumerate(delta_token_ids):
            if tid in self._drop_token_ids:
                continue
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

        if effective_text:
            if has_drop:
                clean_delta = effective_text
                for idx, tid in enumerate(delta_token_ids):
                    if tid in self._drop_token_ids:
                        dropped = token_texts[idx]
                        pos = clean_delta.find(dropped)
                        if pos >= 0:
                            clean_delta = (
                                clean_delta[:pos] + clean_delta[pos + len(dropped) :]
                            )
                if clean_delta:
                    if results:
                        results = self._recover_holdback_text(clean_delta, results)
                    else:
                        results = [TextChunk(clean_delta)]
            else:
                results = self._recover_holdback_text(effective_text, results)
        else:
            # No detokenizer text to validate against — individually-decoded
            # TextChunks are unreliable (context-dependent decoding).  Keep
            # only PreLexedTerminals; the text will arrive in a later delta.
            results = [r for r in results if isinstance(r, PreLexedTerminal)]

        return prefix_items + results

    def flush_pending(self) -> list[LexerInput]:
        """Emit any deferred terminals at end-of-stream."""
        if not self._deferred_terminals and not self._deferred_post_text:
            return []
        results: list[LexerInput] = []
        if self._deferred_post_text:
            results.append(TextChunk(self._deferred_post_text))
            self._deferred_post_text = ""
        results.extend(self._deferred_terminals)
        self._deferred_terminals.clear()
        return results

    def _resolve_deferred(
        self,
        delta_text: str,
    ) -> tuple[list[LexerInput], str]:
        """Resolve deferred terminals against new delta_text.

        When a previous ``scan()`` deferred a terminal (its text hadn't
        arrived yet), the next delta's text should contain that terminal's
        text.  Split delta_text at the terminal boundary: text before
        belongs to the previous parser state, the terminal triggers the
        state transition, and text after belongs to the new state.

        Returns ``(prefix_items, remaining_text)`` where prefix_items
        are the resolved deferred terminals (with any preceding text)
        and remaining_text is the unconsumed portion of delta_text that
        should be scanned with the current delta's token IDs.
        """
        deferred = self._deferred_terminals
        self._deferred_terminals = []

        results: list[LexerInput] = []
        remaining = delta_text

        if self._deferred_post_text:
            remaining = self._deferred_post_text + remaining
            self._deferred_post_text = ""

        for terminal in deferred:
            pos = remaining.find(terminal.text)
            if pos > 0:
                results.append(TextChunk(remaining[:pos]))
                results.append(terminal)
                remaining = remaining[pos + len(terminal.text) :]
            elif pos == 0:
                results.append(terminal)
                remaining = remaining[len(terminal.text) :]
            else:
                # Text still arriving via SentencePiece holdback — re-defer.
                if any(
                    remaining.endswith(terminal.text[:k])
                    for k in range(1, len(terminal.text))
                ):
                    self._deferred_post_text += remaining
                    remaining = ""
                    self._deferred_terminals.append(terminal)
                else:
                    results.append(terminal)

        return results, remaining

    def _recover_holdback_text(
        self,
        delta_text: str,
        results: list[LexerInput],
    ) -> list[LexerInput]:
        """Recover detokenizer hold-back text not in delta_token_ids.

        The detokenizer may flush previously held-back text in
        ``delta_text`` that has no corresponding token ID in
        ``delta_token_ids``.  This hold-back text always appears as a
        prefix of ``delta_text``.
        """
        if not results:
            return [TextChunk(delta_text)]

        reconstructed = self._join_decoded_text(results)
        if reconstructed is None:
            return results  # non-string text, return as-is

        if not reconstructed:
            return [TextChunk(delta_text)] + results

        pos = delta_text.find(reconstructed)
        if pos > 0:
            return [TextChunk(delta_text[:pos])] + results
        if pos == 0:
            return results

        # Fallback: SentencePiece context-dependent decoding mismatch.
        # Rebuild from delta_text using PreLexedTerminals as split anchors.
        return self._rebuild_from_anchors(delta_text, results)

    def _join_decoded_text(self, results: list[LexerInput]) -> str | None:
        """Join TextChunk and PreLexedTerminal text into one string.

        Returns ``None`` if any PreLexedTerminal has a non-string text
        field (indicating unreliable decode).
        """
        parts: list[str] = []
        for item in results:
            if isinstance(item, TextChunk):
                parts.append(item.text)
            elif isinstance(item, PreLexedTerminal):
                if not isinstance(item.text, str):
                    return None
                parts.append(item.text)
        return "".join(parts)

    def _rebuild_from_anchors(
        self,
        delta_text: str,
        results: list[LexerInput],
    ) -> list[LexerInput]:
        """Rebuild results from delta_text using terminals as anchors.

        When context-dependent decoding creates a mismatch between
        individually-decoded tokens and delta_text, use
        PreLexedTerminals as split points and reallocate text from
        delta_text.  If a terminal's text is not found in delta_text,
        it is deferred to the next scan() call.
        """
        new_results: list[LexerInput] = []
        remaining = delta_text
        for item in results:
            if not isinstance(item, PreLexedTerminal):
                continue
            pos = remaining.find(item.text)
            if pos > 0:
                new_results.append(TextChunk(remaining[:pos]))
                new_results.append(item)
                remaining = remaining[pos + len(item.text) :]
            elif pos == 0:
                new_results.append(item)
                remaining = remaining[len(item.text) :]
            else:
                if remaining:
                    self._deferred_post_text += remaining
                    remaining = ""
                self._deferred_terminals.append(item)
        if remaining:
            new_results.append(TextChunk(remaining))
        return new_results
