# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Pure-Python streaming parser for the Harmony message format.

Replaces the Rust ``StreamableParser`` that was previously imported from
``openai-harmony``.  The parser is a 3-state machine that consumes one
token at a time and produces structured ``Message`` objects.
"""

from __future__ import annotations

from vllm.entrypoints.openai.parser.harmony_encoding import (
    _CHANNEL_MARKER,
    _STOP_TOKENS,
    MESSAGE_TOKEN_ID,
    START_TOKEN_ID,
    HarmonyEncoding,
)
from vllm.entrypoints.openai.parser.harmony_types import (
    Author,
    Message,
    Role,
    StreamState,
    TextContent,
)

_REPLACEMENT = "\ufffd"


class StreamableParser:
    """Incremental parser that consumes tokens one-by-one.

    Mirrors the public API of ``openai_harmony.StreamableParser``.
    """

    def __init__(
        self,
        encoding: HarmonyEncoding,
        role: Role | None = None,
        *,
        strict: bool = True,
    ) -> None:
        self._encoding = encoding
        self._strict = strict

        # All tokens fed so far.
        self._tokens: list[int] = []
        # Fully-parsed messages.
        self._messages: list[Message] = []
        # Last successfully-decoded content delta (None while waiting for
        # a multi-byte UTF-8 sequence to complete).
        self._last_content_delta: str | None = None

        # Buffers used during incremental UTF-8 decoding inside the
        # Content state.
        self._undecoded_tokens: list[int] = []
        self._undecoded_bytes: bytearray = bytearray()

        # Header-state accumulator.
        self._header_tokens: list[int] = []

        # Content-state accumulators.
        self._content_tokens: list[int] = []

        # Parsed header fields for the current message.
        self._cur_author: Author | None = None
        self._cur_channel: str | None = None
        self._cur_recipient: str | None = None
        self._cur_content_type: str | None = None

        # The role passed at construction time; consumed after the first
        # header is parsed.
        self._next_role: Role | None = role

        # Initial state depends on whether a role was pre-supplied.
        if role is not None:
            self._state = StreamState.HEADER
        else:
            self._state = StreamState.EXPECT_START

    # -- Public properties --------------------------------------------------

    @property
    def state(self) -> StreamState:
        return self._state

    @property
    def messages(self) -> list[Message]:
        return self._messages

    @property
    def tokens(self) -> list[int]:
        return self._tokens

    @property
    def last_content_delta(self) -> str | None:
        return self._last_content_delta

    @property
    def current_content(self) -> str:
        if self._state != StreamState.CONTENT:
            return ""
        return self._encoding.decode(self._content_tokens)

    @property
    def current_channel(self) -> str | None:
        if self._state == StreamState.CONTENT:
            return self._cur_channel
        return None

    @property
    def current_recipient(self) -> str | None:
        if self._state == StreamState.CONTENT:
            return self._cur_recipient
        return None

    @property
    def current_role(self) -> Role | None:
        if self._state == StreamState.CONTENT and self._cur_author is not None:
            return self._cur_author.role
        return self._next_role

    @property
    def current_content_type(self) -> str | None:
        if self._state == StreamState.CONTENT:
            return self._cur_content_type
        return None

    # -- Token processing ---------------------------------------------------

    def process(self, token: int) -> StreamableParser:
        """Feed a single token and update internal state."""
        self._tokens.append(token)
        self._last_content_delta = None

        if self._state == StreamState.EXPECT_START:
            self._process_expect_start(token)
        elif self._state == StreamState.HEADER:
            self._process_header(token)
        elif self._state == StreamState.CONTENT:
            self._process_content(token)

        return self

    def process_eos(self) -> StreamableParser:
        """Signal end-of-sequence."""
        self._last_content_delta = None
        if self._state == StreamState.CONTENT:
            self._finalize_message()
        return self

    # -- State handlers -----------------------------------------------------

    def _process_expect_start(self, token: int) -> None:
        if token == START_TOKEN_ID:
            self._state = StreamState.HEADER
            self._header_tokens = []
        # EOS or other tokens while expecting start are ignored
        # (the Rust implementation also stays in ExpectStart on EOS)

    def _process_header(self, token: int) -> None:
        if token == MESSAGE_TOKEN_ID:
            # Header complete -> parse and transition to Content.
            self._parse_header()
            self._state = StreamState.CONTENT
            self._content_tokens = []
            self._undecoded_tokens = []
            self._undecoded_bytes = bytearray()
        elif not self._strict and token in _STOP_TOKENS:
            # Malformed: stop token while still in header.
            # If we have a pre-set role, try to salvage what we can.
            if self._next_role is not None and self._header_tokens:
                header_str = self._encoding.decode(self._header_tokens)
                parsed = self._parse_header_string(
                    header_str,
                    self._next_role,
                    parse_recipient_and_type=False,
                )
                if parsed is not None:
                    author, channel, recipient, content_type, remaining = parsed
                    text = remaining or ""
                    msg = Message(
                        author=author,
                        content=[TextContent(text=text)],
                        channel=channel,
                        recipient=recipient,
                        content_type=content_type,
                    )
                    self._messages.append(msg)
            self._state = StreamState.EXPECT_START
            self._next_role = None
        else:
            self._header_tokens.append(token)

    def _process_content(self, token: int) -> None:
        if token in _STOP_TOKENS:
            self._finalize_message()
            return

        # Accumulate and try to incrementally decode.
        self._undecoded_tokens.append(token)
        try:
            decoded_bytes = self._encoding.decode_bytes(self._undecoded_tokens)
        except Exception:
            # Bytes not yet valid — wait for next token.
            self._last_content_delta = None
            return

        self._undecoded_bytes.extend(decoded_bytes)
        self._undecoded_tokens = []

        try:
            decoded_str = self._undecoded_bytes.decode("utf-8")
            # Success — emit as delta and add to content tokens.
            self._content_tokens.extend(self._encoding._encode_ordinary(decoded_str))
            self._last_content_delta = decoded_str
            self._undecoded_bytes = bytearray()
        except UnicodeDecodeError as e:
            valid_len = e.start
            content_delta = ""

            if valid_len > 0:
                valid_str = bytes(self._undecoded_bytes[:valid_len]).decode("utf-8")
                self._content_tokens.extend(self._encoding._encode_ordinary(valid_str))
                content_delta += valid_str
                del self._undecoded_bytes[:valid_len]

            if e.reason != "unexpected end of data":
                # Definite error — emit replacement char.
                error_len = e.end - e.start
                self._content_tokens.extend(
                    self._encoding._encode_ordinary(_REPLACEMENT)
                )
                content_delta += _REPLACEMENT
                del self._undecoded_bytes[:error_len]

            if content_delta:
                self._last_content_delta = content_delta
            else:
                # Waiting for next byte in UTF-8 sequence.
                self._last_content_delta = None

    # -- Header parsing -----------------------------------------------------

    def _parse_header(self) -> None:
        """Parse accumulated header tokens into author/channel/recipient."""
        header_str = self._encoding.decode(self._header_tokens)
        role = self._next_role
        self._next_role = None

        parsed = self._parse_header_string(
            header_str, role, parse_recipient_and_type=True
        )
        if parsed is None:
            # Fallback
            self._cur_author = Author(role=role or Role.ASSISTANT)
            self._cur_channel = None
            self._cur_recipient = None
            self._cur_content_type = None
            return

        author, channel, recipient, content_type, remaining = parsed
        if remaining is not None:
            raise ValueError(
                f"Unexpected tokens remaining in message header: {remaining!r}"
            )
        self._cur_author = author
        self._cur_channel = channel
        self._cur_recipient = recipient
        self._cur_content_type = content_type

    def _parse_header_string(
        self,
        header_string: str,
        role: Role | None,
        parse_recipient_and_type: bool,
    ) -> tuple[Author, str | None, str | None, str | None, str | None] | None:
        """Parse a header string into (author, channel, recipient,
        content_type, remaining_content).

        Mirrors ``StreamableParser::parse_header_from_string`` in the
        Rust crate.
        """
        # 1. Extract channel
        channel: str | None = None
        idx = header_string.find(_CHANNEL_MARKER)
        if idx != -1:
            after = header_string[idx + len(_CHANNEL_MARKER) :]
            # Channel value extends to next whitespace or '<'
            end = len(after)
            for i, ch in enumerate(after):
                if ch.isspace() or ch == "<":
                    end = i
                    break
            channel_value = after[:end]
            if not channel_value:
                return None
            channel = channel_value
            # Remove channel section from header
            header_string = header_string[:idx] + after[end:]

        # 2. Trim whitespace
        header_string = header_string.strip()

        # 3. Handle <|constrain|> marker — insert space before it if not
        #    preceded by whitespace.
        constrain_marker = "<|constrain|>"
        if constrain_marker in header_string:
            header_string = header_string.replace(
                constrain_marker, f" {constrain_marker}"
            ).strip()

        # 4. Split on whitespace
        parts = header_string.split()

        # 5. Determine role
        role_str_opt: str | None = None
        if role is None:
            if not parts:
                return None
            role_str = parts[0]
            role_str_opt = role_str
            try:
                role = Role(role_str)
            except ValueError:
                # Unknown role — treat as Tool
                if len(parts) > 1 or (len(parts) == 1 and parts[0].startswith("to=")):
                    parts.pop(0)
                    role = Role.TOOL
                else:
                    return None

        # Remove role token if it matches
        if parts and parts[0] == role.value:
            parts.pop(0)

        # 6. Parse remaining parts
        recipient: str | None = None
        content_type: str | None = None
        remaining_content: str | None = None

        if parse_recipient_and_type and parts:
            num_parts = len(parts)
            last_part = parts.pop()

            if last_part.startswith("to="):
                recipient = last_part[3:]
            elif num_parts == 1:
                recipient = last_part
            else:
                content_type = last_part
                if parts:
                    raw_recipient = parts.pop()
                    if raw_recipient.startswith("to="):
                        recipient = raw_recipient[3:]
                    else:
                        recipient = raw_recipient

            if parts:
                remaining_content = " ".join(parts)
        else:
            if parts:
                remaining_content = " ".join(parts)

        # 7. Build author
        if role == Role.TOOL:
            author = Author(role=role, name=role_str_opt)
        else:
            author = Author(role=role)

        return author, channel, recipient, content_type, remaining_content

    # -- Message finalization -----------------------------------------------

    def _finalize_message(self) -> None:
        """Build a ``Message`` from the current content state and reset."""
        # Decode content tokens
        content_text = self._encoding.decode(self._content_tokens)

        # Decode any remaining undecoded tokens
        if self._undecoded_tokens:
            try:
                tokens_text = self._encoding.decode(self._undecoded_tokens)
            except Exception:
                tokens_text = _REPLACEMENT
            content_text += tokens_text

        # Decode any remaining undecoded bytes
        if self._undecoded_bytes:
            content_text += self._undecoded_bytes.decode("utf-8", errors="replace")

        msg = Message(
            author=self._cur_author or Author(role=Role.ASSISTANT),
            content=[TextContent(text=content_text)],
            channel=self._cur_channel,
            recipient=self._cur_recipient,
            content_type=self._cur_content_type,
        )
        self._messages.append(msg)

        # Reset for next message
        self._state = StreamState.EXPECT_START
        self._last_content_delta = None
        self._undecoded_tokens = []
        self._undecoded_bytes = bytearray()
        self._content_tokens = []
        self._cur_author = None
        self._cur_channel = None
        self._cur_recipient = None
        self._cur_content_type = None
