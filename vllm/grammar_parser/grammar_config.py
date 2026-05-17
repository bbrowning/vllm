# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Declarative grammar configuration for model tool-call and reasoning
formats.

Each model format is described by a :class:`GrammarConfig` that specifies:

* **terminals** – literal strings or regex patterns that delimit the format
  (e.g. ``<tool_call>``, ``</think>``).
* **token_id_terminals** – terminals that should be matched by token ID
  rather than (or in addition to) text.
* **transitions** – a state machine mapping
  ``(state, terminal) → (new_state, events_to_emit)`` that drives semantic
  event generation during streaming.
* **content_events** – what :class:`EventType` to emit for plain content
  (non-terminal text) in each state.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum, auto

from vllm.grammar_parser.events import EventType


class ParserState(Enum):
    """States for the streaming parser state machine."""

    CONTENT = auto()
    REASONING = auto()
    TOOL_PREAMBLE = auto()
    TOOL_NAME = auto()
    TOOL_ARGS = auto()
    TOOL_BETWEEN = auto()
    DONE = auto()


@dataclass(slots=True)
class Transition:
    """A single state-machine transition."""

    next_state: ParserState
    events: list[EventType] = field(default_factory=list)


@dataclass
class GrammarConfig:
    """Declarative description of a model's tool-call / reasoning format.

    The engine feeds terminals from the incremental lexer into the
    transition table and emits the corresponding semantic events.
    Content tokens (text between terminals) are classified by the
    current state via ``content_events``.
    """

    name: str

    terminals: dict[str, str] = field(default_factory=dict)

    token_id_terminals: dict[str, str] = field(default_factory=dict)

    transitions: dict[tuple[ParserState, str], Transition] = field(
        default_factory=dict,
    )

    content_events: dict[ParserState, EventType] = field(
        default_factory=lambda: {
            ParserState.CONTENT: EventType.TEXT_CHUNK,
            ParserState.REASONING: EventType.REASONING_CHUNK,
            ParserState.TOOL_NAME: EventType.TOOL_NAME,
            ParserState.TOOL_ARGS: EventType.ARG_VALUE_CHUNK,
        },
    )

    initial_state: ParserState = ParserState.CONTENT

    value_postprocessor: Callable[[str], str] | None = None

    arg_converter: Callable[[str, bool], str] | None = None

    strip_trailing_quotes: bool = True

    tool_args_json: bool = True

    arg_structural_chars: frozenset[str] | None = None

    token_id_text_in_delta: bool = False

    drop_tokens: set[str] = field(default_factory=set)
