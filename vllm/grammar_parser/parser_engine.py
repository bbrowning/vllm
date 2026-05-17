# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Streaming parser engine that orchestrates token ID scanning,
incremental lexing, and state-machine-driven semantic event emission."""

from __future__ import annotations

from collections.abc import Sequence

from vllm.grammar_parser.events import EventType, SemanticEvent
from vllm.grammar_parser.grammar_config import GrammarConfig, ParserState, Transition
from vllm.grammar_parser.incremental_lexer import (
    IncrementalLexer,
    LexToken,
    terminals_from_literals,
)
from vllm.grammar_parser.token_id_scanner import (
    PreLexedTerminal,
    TextChunk,
    TokenIDScanner,
)


class StreamingParserEngine:
    """Consumes ``(delta_text, delta_token_ids)`` pairs and produces a
    stream of :class:`SemanticEvent` instances.

    This is the main entry point for grammar-driven streaming parsing.
    Create one per request (it is stateful).

    The pipeline is::

        delta_text + delta_token_ids
            → TokenIDScanner  (special token pre-lexing)
            → IncrementalLexer  (text → terminal tokens with prefix buffering)
            → State Machine  (terminal → semantic events)
            → list[SemanticEvent]

    Usage::

        engine = StreamingParserEngine(config, tokenizer)
        for each streaming delta:
            events = engine.feed(delta_text, delta_token_ids)
            # convert events to DeltaMessage
    """

    def __init__(
        self,
        config: GrammarConfig,
        tokenizer,
    ) -> None:
        self.config = config
        self.state = config.initial_state
        self.tool_index = -1

        resolved_token_ids: dict[int, str] = {}
        drop_token_ids: set[int] = set()
        if tokenizer is not None:
            vocab = tokenizer.get_vocab()
            if config.token_id_terminals:
                for terminal_name, token_text in config.token_id_terminals.items():
                    tid = vocab.get(token_text)
                    if tid is not None:
                        resolved_token_ids[tid] = terminal_name
            for token_text in config.drop_tokens:
                tid = vocab.get(token_text)
                if tid is not None:
                    drop_token_ids.add(tid)

        self._scanner = TokenIDScanner(
            resolved_token_ids,
            tokenizer,
            drop_token_ids,
            token_id_text_in_delta=config.token_id_text_in_delta,
        )

        terminal_defs = terminals_from_literals(config.terminals)
        self._lexer = IncrementalLexer(terminal_defs, content_terminal="__CONTENT__")

        self._args_buffer = ""
        self._args_brace_depth = 0
        self._args_in_string = False
        self._args_escape_next = False

    def feed(
        self,
        delta_text: str,
        delta_token_ids: Sequence[int],
    ) -> list[SemanticEvent]:
        """Feed one streaming delta and return produced events."""
        scanner_items = self._scanner.scan(delta_text, delta_token_ids)

        if len(scanner_items) == 1 and isinstance(scanner_items[0], TextChunk):
            lex_tokens = self._lexer.feed(scanner_items[0].text)
            if len(lex_tokens) == 1 and lex_tokens[0].terminal == "__CONTENT__":
                text = lex_tokens[0].value
                if self.state == ParserState.TOOL_ARGS:
                    if self.config.tool_args_json:
                        return self._feed_args_text(text)
                    return [
                        SemanticEvent(
                            EventType.ARG_VALUE_CHUNK,
                            value=text,
                            tool_index=self.tool_index,
                        )
                    ]
                content_type = self.config.content_events.get(self.state)
                if content_type is not None:
                    return [
                        SemanticEvent(
                            content_type,
                            value=text,
                            tool_index=self.tool_index,
                        )
                    ]
                return []
            return self._process_lex_tokens(lex_tokens)

        events: list[SemanticEvent] = []
        for item in scanner_items:
            if isinstance(item, PreLexedTerminal):
                events.extend(self._process_lex_tokens(self._lexer.flush()))
                events.extend(self._on_terminal(item.terminal, item.text))
            elif isinstance(item, TextChunk):
                events.extend(self._process_lex_tokens(self._lexer.feed(item.text)))

        return events

    def finish(self) -> list[SemanticEvent]:
        """Signal end-of-stream and flush any remaining buffered content."""
        events: list[SemanticEvent] = []

        for item in self._scanner.flush_pending():
            if isinstance(item, PreLexedTerminal):
                events.extend(self._process_lex_tokens(self._lexer.flush()))
                events.extend(self._on_terminal(item.terminal, item.text))
            elif isinstance(item, TextChunk):
                events.extend(self._process_lex_tokens(self._lexer.feed(item.text)))

        events.extend(self._process_lex_tokens(self._lexer.flush()))

        if self._args_buffer:
            events.append(
                SemanticEvent(
                    EventType.ARG_VALUE_CHUNK,
                    value=self._args_buffer,
                    tool_index=self.tool_index,
                )
            )
            self._args_buffer = ""

        return events

    def parse_complete(self, text: str) -> list[SemanticEvent]:
        """Non-streaming: parse a complete text and return all events."""
        token_ids: list[int] = []
        events = self.feed(text, token_ids)
        events.extend(self.finish())
        return events

    def _process_lex_tokens(self, tokens: list[LexToken]) -> list[SemanticEvent]:
        """Dispatch a list of lex tokens through _on_content / _on_terminal."""
        events: list[SemanticEvent] = []
        for tok in tokens:
            if tok.terminal == "__CONTENT__":
                events.extend(self._on_content(tok.value))
            else:
                events.extend(self._on_terminal(tok.terminal, tok.value))
        return events

    def _on_terminal(self, terminal: str, value: str) -> list[SemanticEvent]:
        """Process a terminal token through the state machine."""
        key = (self.state, terminal)
        transition = self.config.transitions.get(key)

        if transition is None:
            if self.state == ParserState.TOOL_ARGS:
                if self.config.tool_args_json:
                    return self._feed_args_char(value)
                return [
                    SemanticEvent(
                        EventType.ARG_VALUE_CHUNK,
                        value=value,
                        tool_index=self.tool_index,
                    )
                ]
            content_type = self.config.content_events.get(self.state)
            if content_type is not None:
                return [
                    SemanticEvent(content_type, value=value, tool_index=self.tool_index)
                ]
            return []

        return self._apply_transition(transition, value)

    def _on_content(self, text: str) -> list[SemanticEvent]:
        """Process plain content text based on current state."""
        if not text:
            return []

        if self.state == ParserState.TOOL_ARGS:
            if self.config.tool_args_json:
                return self._feed_args_text(text)
            return [
                SemanticEvent(
                    EventType.ARG_VALUE_CHUNK, value=text, tool_index=self.tool_index
                )
            ]

        content_type = self.config.content_events.get(self.state)
        if content_type is None:
            return []

        return [SemanticEvent(content_type, value=text, tool_index=self.tool_index)]

    def _apply_transition(
        self,
        transition: Transition,
        value: str,
    ) -> list[SemanticEvent]:
        """Apply a state transition and emit the associated events."""
        events: list[SemanticEvent] = []

        if (
            self.state == ParserState.TOOL_ARGS
            and transition.next_state != ParserState.TOOL_ARGS
        ):
            if self._args_buffer:
                events.append(
                    SemanticEvent(
                        EventType.ARG_VALUE_CHUNK,
                        value=self._args_buffer,
                        tool_index=self.tool_index,
                    )
                )
                self._args_buffer = ""
            self._args_brace_depth = 0
            self._args_in_string = False
            self._args_escape_next = False

        self.state = transition.next_state

        for event_type in transition.events:
            if event_type == EventType.TOOL_CALL_START:
                self.tool_index += 1
            events.append(
                SemanticEvent(
                    event_type,
                    value=value,
                    tool_index=self.tool_index,
                )
            )

        if self.state == ParserState.TOOL_ARGS:
            self._args_brace_depth = 0
            self._args_in_string = False
            self._args_escape_next = False

        return events

    def _feed_args_text(self, text: str) -> list[SemanticEvent]:
        """Feed text into the JSON argument streaming buffer.

        Streams argument characters incrementally while holding back
        closing braces/brackets that might change as more input arrives.
        """
        events: list[SemanticEvent] = []
        for ch in text:
            result = self._feed_args_char(ch)
            events.extend(result)
        return events

    def _feed_args_char(self, ch: str) -> list[SemanticEvent]:
        """Process one character of argument content."""
        self._args_buffer += ch

        if self._args_escape_next:
            self._args_escape_next = False
            return self._try_flush_args()

        if self._args_in_string:
            if ch == "\\":
                self._args_escape_next = True
            elif ch == '"':
                self._args_in_string = False
            return self._try_flush_args()

        if ch == '"':
            self._args_in_string = True
            return self._try_flush_args()

        if ch in ("{", "["):
            self._args_brace_depth += 1
            return self._try_flush_args()

        if ch in ("}", "]"):
            self._args_brace_depth -= 1
            if self._args_brace_depth <= 0:
                return []
            return self._try_flush_args()

        return self._try_flush_args()

    def _try_flush_args(self) -> list[SemanticEvent]:
        """Flush safe argument characters from the buffer.

        Characters inside strings and non-final structural characters
        are safe to emit.  Top-level closing braces are held back
        until confirmed by a subsequent terminal.
        """
        safe_end = 0
        depth = 0
        in_str = False
        escape = False

        for i, ch in enumerate(self._args_buffer):
            if escape:
                escape = False
                safe_end = i + 1
                continue
            if in_str:
                if ch == "\\":
                    escape = True
                elif ch == '"':
                    in_str = False
                safe_end = i + 1
                continue
            if ch == '"':
                in_str = True
                safe_end = i + 1
            elif ch in ("{", "["):
                depth += 1
                safe_end = i + 1
            elif ch in ("}", "]"):
                depth -= 1
                if depth > 0:
                    safe_end = i + 1
            else:
                safe_end = i + 1

        if safe_end == 0:
            return []

        to_emit = self._args_buffer[:safe_end]
        self._args_buffer = self._args_buffer[safe_end:]

        return [
            SemanticEvent(
                EventType.ARG_VALUE_CHUNK,
                value=to_emit,
                tool_index=self.tool_index,
            )
        ]
