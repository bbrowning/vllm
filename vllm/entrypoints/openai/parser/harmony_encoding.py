# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Pure-Python Harmony encoding: renders ``Message`` objects to token-ID
sequences and provides tokenisation helpers backed by ``tiktoken``.

Replaces the Rust ``HarmonyEncoding`` + ``load_harmony_encoding`` that were
previously imported from ``openai-harmony``.
"""

from __future__ import annotations

import json
from typing import Any

import tiktoken

from vllm.entrypoints.openai.parser.harmony_types import (
    Conversation,
    DeveloperContent,
    HarmonyEncodingName,
    Message,
    ReasoningEffort,
    Role,
    SystemContent,
    TextContent,
    ToolNamespaceConfig,
)

# ---------------------------------------------------------------------------
# Special token string -> token-ID mapping  (from ``public_encodings.rs``)
# ---------------------------------------------------------------------------

_HARMONY_SPECIAL_TOKENS: dict[str, int] = {
    "<|startoftext|>": 199998,
    "<|endoftext|>": 199999,
    "<|reserved_200000|>": 200000,
    "<|reserved_200001|>": 200001,
    "<|return|>": 200002,
    "<|constrain|>": 200003,
    "<|reserved_200004|>": 200004,
    "<|channel|>": 200005,
    "<|start|>": 200006,
    "<|end|>": 200007,
    "<|message|>": 200008,
    "<|reserved_200009|>": 200009,
    "<|reserved_200010|>": 200010,
    "<|reserved_200011|>": 200011,
    "<|call|>": 200012,
    "<|reserved_200013|>": 200013,
}

# Add reserved token range 200014..=201088
for _id in range(200014, 201089):
    _HARMONY_SPECIAL_TOKENS[f"<|reserved_{_id}|>"] = _id

# Convenience constants for the token IDs we use most.
START_TOKEN_ID: int = _HARMONY_SPECIAL_TOKENS["<|start|>"]  # 200006
MESSAGE_TOKEN_ID: int = _HARMONY_SPECIAL_TOKENS["<|message|>"]  # 200008
END_TOKEN_ID: int = _HARMONY_SPECIAL_TOKENS["<|end|>"]  # 200007
CALL_TOKEN_ID: int = _HARMONY_SPECIAL_TOKENS["<|call|>"]  # 200012
CHANNEL_TOKEN_ID: int = _HARMONY_SPECIAL_TOKENS["<|channel|>"]  # 200005
RETURN_TOKEN_ID: int = _HARMONY_SPECIAL_TOKENS["<|return|>"]  # 200002
CONSTRAIN_TOKEN_ID: int = _HARMONY_SPECIAL_TOKENS["<|constrain|>"]  # 200003

# The set of all special-token *strings*.
_ALL_SPECIAL_TOKEN_STRINGS: frozenset[str] = frozenset(_HARMONY_SPECIAL_TOKENS.keys())

# Token-ID sets used as stop criteria.
_STOP_TOKENS: frozenset[int] = frozenset({END_TOKEN_ID, CALL_TOKEN_ID, RETURN_TOKEN_ID})
_STOP_TOKENS_FOR_ASSISTANT_ACTIONS: frozenset[int] = frozenset(
    {CALL_TOKEN_ID, RETURN_TOKEN_ID}
)
_SORTED_STOP_TOKENS: list[int] = sorted(_STOP_TOKENS)
_SORTED_STOP_TOKENS_FOR_ASSISTANT_ACTIONS: list[int] = sorted(
    _STOP_TOKENS_FOR_ASSISTANT_ACTIONS
)

# String representations of the formatting tokens used in header parsing.
_CHANNEL_MARKER = "<|channel|>"
_CONSTRAIN_MARKER = "<|constrain|>"

_REASONING_EFFORT_STRINGS: dict[ReasoningEffort, str] = {
    ReasoningEffort.LOW: "low",
    ReasoningEffort.MEDIUM: "medium",
    ReasoningEffort.HIGH: "high",
}


# ---------------------------------------------------------------------------
# tiktoken encoding construction
# ---------------------------------------------------------------------------

_cached_tiktoken_enc: tiktoken.Encoding | None = None


def _get_tiktoken_encoding() -> tiktoken.Encoding:
    """Build (and cache) a ``tiktoken.Encoding`` for *o200k_harmony*."""
    global _cached_tiktoken_enc
    if _cached_tiktoken_enc is not None:
        return _cached_tiktoken_enc

    base = tiktoken.get_encoding("o200k_base")
    enc = tiktoken.Encoding(
        name="o200k_harmony",
        pat_str=base._pat_str,
        mergeable_ranks=base._mergeable_ranks,
        special_tokens={**base._special_tokens, **_HARMONY_SPECIAL_TOKENS},
    )
    _cached_tiktoken_enc = enc
    return enc


# ---------------------------------------------------------------------------
# JSON Schema -> TypeScript conversion
# ---------------------------------------------------------------------------


def _is_enum(schema: dict) -> bool:
    enum_vals = schema.get("enum")
    return isinstance(enum_vals, list) and len(enum_vals) > 0


def json_schema_to_typescript(schema: dict | Any, indent: str = "") -> str:
    """Convert a JSON Schema object to a TypeScript-like type string.

    Faithfully mirrors ``HarmonyEncoding::json_schema_to_typescript`` in the
    Rust crate so that tool definitions render identically.
    """
    if not isinstance(schema, dict):
        return "any"

    # Handle top-level oneOf
    one_of = schema.get("oneOf")
    if isinstance(one_of, list) and one_of:
        parts: list[str] = []
        for i, variant in enumerate(one_of):
            prefix = f"\n{indent} | "
            type_str = json_schema_to_typescript(variant, f"{indent}   ")
            if (
                isinstance(variant, dict)
                and variant.get("nullable") is True
                and "null" not in type_str
            ):
                type_str = f"{type_str} | null"
            trailing = _trailing_comments(variant)
            parts.append(f"{prefix}{type_str}{trailing}")
        return "".join(parts)

    # Handle type-as-array (e.g. ["number", "string"])
    type_val = schema.get("type")
    if isinstance(type_val, list):
        mapped = [
            ("number" if t == "integer" else t) for t in type_val if isinstance(t, str)
        ]
        if mapped:
            return " | ".join(mapped)

    if isinstance(type_val, str):
        if type_val == "object":
            return _ts_object(schema, indent)
        if type_val == "string":
            enum_vals = schema.get("enum")
            if isinstance(enum_vals, list) and enum_vals:
                return " | ".join(f'"{v}"' for v in enum_vals if isinstance(v, str))
            return "string"
        if type_val in ("number", "integer"):
            return "number"
        if type_val == "boolean":
            return "boolean"
        if type_val == "array":
            items = schema.get("items")
            if items is not None:
                return f"{json_schema_to_typescript(items, indent)}[]"
            return "Array<any>"
        return "any"

    # Fallback: no type, maybe oneOf handled above already
    if isinstance(one_of, list) and one_of:
        # Already handled, but defensive
        parts = []
        first = True
        for variant in one_of:
            if not first:
                parts.append("\n | ")
            else:
                first = False
            parts.append(json_schema_to_typescript(variant, indent))
        return "".join(parts)

    return "any"


def _trailing_comments(variant: dict) -> str:
    """Build ``// description default: value`` trailing comment."""
    comments: list[str] = []
    desc = variant.get("description")
    if isinstance(desc, str):
        comments.append(desc)
    default = variant.get("default")
    if default is not None:
        if isinstance(default, str) and not _is_enum(variant):
            comments.append(f'default: "{default}"')
        elif isinstance(default, str):
            comments.append(f"default: {default}")
        else:
            comments.append(f"default: {json.dumps(default)}")
    if comments:
        return f" // {' '.join(comments)}"
    return ""


def _default_comment(schema: dict) -> str:
    """Build a ``// default: value`` trailing comment."""
    default = schema.get("default")
    if default is None:
        return ""
    if isinstance(default, str) and not _is_enum(schema):
        return f' // default: "{default}"'
    if isinstance(default, str):
        return f" // default: {default}"
    return f" // default: {json.dumps(default)}"


def _ts_object(schema: dict, indent: str) -> str:
    out: list[str] = []

    # Object-level description
    desc = schema.get("description")
    if isinstance(desc, str):
        out.append(f"{indent}// {desc}\n")

    out.append("{\n")

    props = schema.get("properties")
    if isinstance(props, dict):
        required_set: set[str] = set()
        req = schema.get("required")
        if isinstance(req, list):
            required_set = {r for r in req if isinstance(r, str)}

        for key, val in props.items():
            if not isinstance(val, dict):
                continue
            opt = "" if key in required_set else "?"

            # Title
            title = val.get("title")
            if isinstance(title, str):
                out.append(f"{indent}// {title}\n{indent}//\n")

            # oneOf at property level
            prop_one_of = val.get("oneOf")
            if isinstance(prop_one_of, list) and prop_one_of:
                _render_oneof_property(out, key, opt, val, prop_one_of, indent)
                continue

            # Description (only if not oneOf)
            prop_desc = val.get("description")
            if isinstance(prop_desc, str):
                out.append(f"{indent}// {prop_desc}\n")

            # Examples
            examples = val.get("examples")
            if isinstance(examples, list) and examples:
                out.append(f"{indent}// Examples:\n")
                for ex in examples:
                    if isinstance(ex, str):
                        out.append(f'{indent}// - "{ex}"\n')

            # Type
            type_str = json_schema_to_typescript(val, f"{indent}    ")
            if val.get("nullable") is True and "null" not in type_str:
                type_str = f"{type_str} | null"

            out.append(f"{indent}{key}{opt}: {type_str},")

            # Default as trailing comment
            out.append(_default_comment(val))
            out.append("\n")

    out.append(f"{indent}}}")
    return "".join(out)


def _render_oneof_property(
    out: list[str],
    key: str,
    opt: str,
    val: dict,
    one_of: list,
    indent: str,
) -> None:
    """Render a property whose type is ``oneOf``."""
    property_desc: str | None = None
    desc = val.get("description")
    if isinstance(desc, str):
        property_desc = desc

    # Check if first variant has same description -> skip property desc
    skip_property_desc = False
    if property_desc is not None and one_of:
        first_variant = one_of[0]
        if isinstance(first_variant, dict):
            variant_desc = first_variant.get("description")
            if isinstance(variant_desc, str) and variant_desc == property_desc:
                skip_property_desc = True

    rendered_desc_above = False
    if not skip_property_desc and property_desc is not None:
        out.append(f"{indent}// {property_desc}\n")
        rendered_desc_above = True

    # Property-level default
    default = val.get("default")
    if default is not None:
        if isinstance(default, str) and not _is_enum(val):
            out.append(f'{indent}// default: "{default}"\n')
        elif isinstance(default, str):
            out.append(f"{indent}// default: {default}\n")
        else:
            out.append(f"{indent}// default: {json.dumps(default)}\n")

    # Property name
    out.append(f"{indent}{key}{opt}:\n")

    # Variants
    for i, variant in enumerate(one_of):
        if not isinstance(variant, dict):
            continue
        out.append(f"{indent} | ")
        type_str = json_schema_to_typescript(variant, f"{indent}   ")
        if variant.get("nullable") is True and "null" not in type_str:
            type_str = f"{type_str} | null"
        out.append(type_str)

        # Trailing comments per variant
        trailing_comments: list[str] = []
        if i == 0 and rendered_desc_above:
            # Don't repeat the description for the first variant
            pass
        else:
            vdesc = variant.get("description")
            if isinstance(vdesc, str) and vdesc != property_desc:
                trailing_comments.append(vdesc)

        vdefault = variant.get("default")
        if vdefault is not None:
            if isinstance(vdefault, str) and not _is_enum(variant):
                trailing_comments.append(f'default: "{vdefault}"')
            elif isinstance(vdefault, str):
                trailing_comments.append(f"default: {vdefault}")
            else:
                trailing_comments.append(f"default: {json.dumps(vdefault)}")

        if trailing_comments:
            out.append(f" // {' '.join(trailing_comments)}")
        out.append("\n")

    out.append(f"{indent},\n")


# ---------------------------------------------------------------------------
# Tools section rendering
# ---------------------------------------------------------------------------


def _render_tools_section(
    tools: dict[str, ToolNamespaceConfig],
) -> str:
    """Render the ``# Tools`` section for system/developer content."""
    tool_sections: list[str] = ["# Tools"]

    # BTreeMap ordering in Rust = sorted by key
    for ns_name in sorted(tools.keys()):
        ns_config = tools[ns_name]
        section_lines: list[str] = [f"## {ns_config.name}\n"]

        if ns_config.description is not None:
            for line in ns_config.description.splitlines():
                if ns_config.tools:
                    section_lines.append(f"// {line}")
                else:
                    section_lines.append(line)

        if ns_config.tools:
            section_lines.append(f"namespace {ns_config.name} {{\n")
            for tool in ns_config.tools:
                for line in tool.description.splitlines():
                    section_lines.append(f"// {line}")
                if tool.parameters is not None:
                    param_type = json_schema_to_typescript(tool.parameters, "")
                    section_lines.append(
                        f"type {tool.name} = (_: {param_type}) => any;\n"
                    )
                else:
                    section_lines.append(f"type {tool.name} = () => any;\n")
            section_lines.append(f"}} // namespace {ns_config.name}")

        tool_sections.append("\n".join(section_lines))

    return "\n\n".join(tool_sections)


# ---------------------------------------------------------------------------
# Content rendering helpers
# ---------------------------------------------------------------------------


def _render_system_content_text(
    sys: SystemContent,
    conversation_has_function_tools: bool = False,
) -> str:
    """Produce the text body of a system message."""
    sections: list[str] = []

    # Section 1: identity block
    top: list[str] = []
    if sys.model_identity is not None:
        top.append(sys.model_identity)
    if sys.knowledge_cutoff is not None:
        top.append(f"Knowledge cutoff: {sys.knowledge_cutoff}")
    if sys.conversation_start_date is not None:
        top.append(f"Current date: {sys.conversation_start_date}")
    if top:
        sections.append("\n".join(top))

    # Section 2: reasoning effort
    if sys.reasoning_effort is not None:
        effort_str = _REASONING_EFFORT_STRINGS[sys.reasoning_effort]
        sections.append(f"Reasoning: {effort_str}")

    # Section 3: tools
    if sys.tools:
        sections.append(_render_tools_section(sys.tools))

    # Section 4: channel config
    if sys.channel_config is not None and sys.channel_config.valid_channels:
        channels_str = ", ".join(sys.channel_config.valid_channels)
        header = f"# Valid channels: {channels_str}."
        if sys.channel_config.channel_required:
            header += " Channel must be included for every message."
        if conversation_has_function_tools:
            header += (
                "\nCalls to these tools must go to the commentary channel: 'functions'."
            )
        sections.append(header)

    return "\n\n".join(sections)


def _render_developer_content_text(dev: DeveloperContent) -> str:
    """Produce the text body of a developer message."""
    sections: list[str] = []
    if dev.instructions is not None:
        sections.append("# Instructions")
        sections.append(dev.instructions)
    if dev.tools:
        sections.append(_render_tools_section(dev.tools))
    return "\n\n".join(sections)


# ---------------------------------------------------------------------------
# HarmonyEncoding
# ---------------------------------------------------------------------------


class HarmonyEncoding:
    """High-level encoding that renders ``Message`` objects to token IDs.

    Wraps a ``tiktoken.Encoding`` and mirrors the public API of the
    Rust ``HarmonyEncoding`` / Python ``openai_harmony.HarmonyEncoding``.
    """

    def __init__(self, enc: tiktoken.Encoding) -> None:
        self._enc = enc

    # -- Low-level tokenisation --------------------------------------------

    def encode(
        self,
        text: str,
        allowed_special: str | frozenset[str] | set[str] = frozenset(),
        **_kwargs: Any,
    ) -> list[int]:
        if allowed_special == "all":
            allowed_special = _ALL_SPECIAL_TOKEN_STRINGS
        return self._enc.encode(text, allowed_special=allowed_special)

    def decode(self, tokens: list[int] | tuple[int, ...]) -> str:
        return self._enc.decode(list(tokens))

    def decode_bytes(self, tokens: list[int] | tuple[int, ...]) -> bytes:
        return b"".join(self._enc.decode_single_token_bytes(t) for t in tokens)

    def decode_single_token_bytes(self, token: int) -> bytes:
        return self._enc.decode_single_token_bytes(token)

    # -- Stop tokens -------------------------------------------------------

    def stop_tokens(self) -> list[int]:
        return _SORTED_STOP_TOKENS

    def stop_tokens_for_assistant_actions(self) -> list[int]:
        return _SORTED_STOP_TOKENS_FOR_ASSISTANT_ACTIONS

    # -- Rendering ---------------------------------------------------------

    def render(self, message: Message) -> list[int]:
        """Render a single message to token IDs."""
        tokens: list[int] = []
        self._render_into(message, tokens)
        return tokens

    def render_conversation_for_completion(
        self,
        conversation: Conversation,
        next_turn_role: Role,
    ) -> list[int]:
        """Render all messages + ``<|start|>{role}`` completion prompt."""
        # Detect whether any message carries function tools
        has_function_tools = _conversation_has_function_tools(conversation.messages)

        tokens: list[int] = []
        for msg in conversation.messages:
            self._render_into(
                msg,
                tokens,
                conversation_has_function_tools=has_function_tools,
            )
        # Append completion prefix: <|start|>{role}
        tokens.append(START_TOKEN_ID)
        tokens.extend(
            self._encode_ordinary(
                next_turn_role.value
                if isinstance(next_turn_role, Role)
                else str(next_turn_role)
            )
        )
        return tokens

    # -- Internal rendering ------------------------------------------------

    def _encode_ordinary(self, text: str) -> list[int]:
        """Encode text without recognising special tokens."""
        return self._enc.encode(text, allowed_special=set())

    def _render_into(
        self,
        message: Message,
        tokens: list[int],
        conversation_has_function_tools: bool = False,
    ) -> None:
        """Append the token-ID representation of *message* to *tokens*."""

        # <|start|>
        tokens.append(START_TOKEN_ID)

        # Role / tool name
        if message.author.role == Role.TOOL:
            if message.author.name:
                tokens.extend(self._encode_ordinary(message.author.name))
            else:
                raise ValueError("Tool messages must have an author name")
        else:
            tokens.extend(self._encode_ordinary(message.author.role.value))
            if message.author.name:
                tokens.extend(self._encode_ordinary(f":{message.author.name}"))

        # Recipient
        if message.recipient is not None and message.recipient != "all":
            tokens.extend(self._encode_ordinary(f" to={message.recipient}"))

        # Channel
        if message.channel is not None:
            tokens.append(CHANNEL_TOKEN_ID)
            tokens.extend(self._encode_ordinary(message.channel))

        # Content type
        if message.content_type is not None:
            if message.content_type.startswith(_CONSTRAIN_MARKER):
                rest = message.content_type[len(_CONSTRAIN_MARKER) :]
                tokens.extend(self._encode_ordinary(" "))
                tokens.append(CONSTRAIN_TOKEN_ID)
                if rest:
                    tokens.extend(self._encode_ordinary(rest))
            else:
                tokens.extend(self._encode_ordinary(f" {message.content_type}"))

        # <|message|>
        tokens.append(MESSAGE_TOKEN_ID)

        # Content
        for content in message.content:
            if isinstance(content, SystemContent):
                if message.author.role != Role.SYSTEM:
                    raise ValueError("SystemContent may only appear in system messages")
                text = _render_system_content_text(
                    content, conversation_has_function_tools
                )
                tokens.extend(self._encode_ordinary(text))
            elif isinstance(content, DeveloperContent):
                if message.author.role != Role.DEVELOPER:
                    raise ValueError(
                        "DeveloperContent may only appear in developer messages"
                    )
                text = _render_developer_content_text(content)
                tokens.extend(self._encode_ordinary(text))
            elif isinstance(content, TextContent):
                tokens.extend(self._encode_ordinary(content.text))
            else:
                # Unknown content type — try to treat as text
                if hasattr(content, "text"):
                    tokens.extend(
                        self._encode_ordinary(content.text)  # type: ignore[union-attr]
                    )

        # End token
        if message.author.role == Role.ASSISTANT and message.recipient is not None:
            tokens.append(CALL_TOKEN_ID)
        else:
            tokens.append(END_TOKEN_ID)


def _conversation_has_function_tools(messages: list[Message]) -> bool:
    """Check if any message in the conversation carries function tools."""
    for msg in messages:
        for c in msg.content:
            if (
                isinstance(c, DeveloperContent)
                and c.tools
                and "functions" in c.tools
                and c.tools["functions"].tools
            ):
                return True
    return False


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------


def load_harmony_encoding(
    name: str | HarmonyEncodingName,
) -> HarmonyEncoding:
    """Load a ``HarmonyEncoding`` by name."""
    # Accept both strings and enum values.
    name_str = str(name) if not isinstance(name, str) else name
    if name_str != str(HarmonyEncodingName.HARMONY_GPT_OSS):
        raise ValueError(f"Unknown encoding: {name_str}")
    return HarmonyEncoding(_get_tiktoken_encoding())
