# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Pure-Python data classes for the Harmony message format.

Replaces the Pydantic models previously provided by the ``openai-harmony``
Rust/Python library with lightweight plain-Python equivalents that keep the
same public API surface.
"""

from __future__ import annotations

import enum
import json
from collections.abc import Sequence
from typing import Any

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class Role(str, enum.Enum):
    """Message author role (mirrors ``chat::Role`` in the Rust crate)."""

    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    DEVELOPER = "developer"
    TOOL = "tool"

    @classmethod
    def _missing_(cls, value: object) -> Role:
        raise ValueError(f"Unknown role: {value!r}")


class ReasoningEffort(str, enum.Enum):
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"


class HarmonyEncodingName(str, enum.Enum):
    HARMONY_GPT_OSS = "HarmonyGptOss"

    def __str__(self) -> str:
        return str(self.value)


class StreamState(enum.Enum):
    EXPECT_START = "ExpectStart"
    HEADER = "Header"
    CONTENT = "Content"


# ---------------------------------------------------------------------------
# Simple data holders
# ---------------------------------------------------------------------------


class Author:
    """Message author with role and optional name."""

    __slots__ = ("role", "name")

    def __init__(
        self,
        role: Role | str,
        name: str | None = None,
    ) -> None:
        self.role: Role = Role(role) if not isinstance(role, Role) else role
        self.name: str | None = name

    @classmethod
    def new(cls, role: Role | str, name: str | None = None) -> Author:
        return cls(role=role, name=name)

    def model_dump(self, **_kwargs: Any) -> dict[str, Any]:
        return {"role": self.role.value, "name": self.name}

    def __repr__(self) -> str:
        return f"Author(role={self.role!r}, name={self.name!r})"

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Author):
            return self.role == other.role and self.name == other.name
        return NotImplemented


class TextContent:
    """Plain text content block."""

    __slots__ = ("text",)

    def __init__(self, text: str = "") -> None:
        self.text = text

    def to_dict(self) -> dict[str, Any]:
        return {"type": "text", "text": self.text}

    def __repr__(self) -> str:
        return f"TextContent(text={self.text!r})"

    def __eq__(self, other: object) -> bool:
        if isinstance(other, TextContent):
            return self.text == other.text
        return NotImplemented


# ---------------------------------------------------------------------------
# Tool descriptions
# ---------------------------------------------------------------------------


class ToolDescription:
    """A single tool's metadata (name, description, JSON-Schema parameters)."""

    __slots__ = ("name", "description", "parameters")

    def __init__(
        self,
        name: str,
        description: str,
        parameters: dict | None = None,
    ) -> None:
        self.name = name
        self.description = description
        self.parameters = parameters

    @classmethod
    def new(
        cls,
        name: str,
        description: str | None,
        parameters: dict | None = None,
    ) -> ToolDescription:
        return cls(
            name=name,
            description=description or "",
            parameters=parameters,
        )

    def __repr__(self) -> str:
        return (
            f"ToolDescription(name={self.name!r}, "
            f"description={self.description!r}, "
            f"parameters={self.parameters!r})"
        )


class ToolNamespaceConfig:
    """A namespace of tools (e.g. ``functions``, ``browser``)."""

    __slots__ = ("name", "description", "tools")

    def __init__(
        self,
        name: str,
        description: str | None = None,
        tools: list[ToolDescription] | None = None,
    ) -> None:
        self.name = name
        self.description = description
        self.tools: list[ToolDescription] = tools if tools is not None else []

    @staticmethod
    def browser() -> ToolNamespaceConfig:
        return ToolNamespaceConfig(
            name="browser",
            description=(
                "Tool for browsing.\n"
                "The `cursor` appears in brackets before each browsing "
                "display: `[{cursor}]`.\n"
                "Cite information from the tool using the following format:\n"
                "`\u3010{cursor}\u2020L{line_start}(-L{line_end})?\u3011`, "
                "for example: `\u30106\u2020L9-L11\u3011` or "
                "`\u30108\u2020L3\u3011`.\n"
                "Do not quote more than 10 words directly from the tool "
                "output.\n"
                "sources=web (default: web)"
            ),
            tools=[
                ToolDescription.new(
                    name="search",
                    description=(
                        "Searches for information related to `query` and "
                        "displays `topn` results."
                    ),
                    parameters={
                        "type": "object",
                        "properties": {
                            "query": {"type": "string"},
                            "topn": {"type": "number", "default": 10},
                            "source": {"type": "string"},
                        },
                        "required": ["query"],
                    },
                ),
                ToolDescription.new(
                    name="open",
                    description=(
                        "Opens the link `id` from the page indicated by "
                        "`cursor` starting at line number `loc`, showing "
                        "`num_lines` lines.\n"
                        "Valid link ids are displayed with the formatting: "
                        "`\u3010{id}\u2020.*\u3011`.\n"
                        "If `cursor` is not provided, the most recent page "
                        "is implied.\n"
                        "If `id` is a string, it is treated as a fully "
                        "qualified URL associated with `source`.\n"
                        "If `loc` is not provided, the viewport will be "
                        "positioned at the beginning of the document or "
                        "centered on the most relevant passage, if "
                        "available.\n"
                        "Use this function without `id` to scroll to a new "
                        "location of an opened page."
                    ),
                    parameters={
                        "type": "object",
                        "properties": {
                            "id": {
                                "type": ["number", "string"],
                                "default": -1,
                            },
                            "cursor": {"type": "number", "default": -1},
                            "loc": {"type": "number", "default": -1},
                            "num_lines": {"type": "number", "default": -1},
                            "view_source": {
                                "type": "boolean",
                                "default": False,
                            },
                            "source": {"type": "string"},
                        },
                    },
                ),
                ToolDescription.new(
                    name="find",
                    description=(
                        "Finds exact matches of `pattern` in the current "
                        "page, or the page given by `cursor`."
                    ),
                    parameters={
                        "type": "object",
                        "properties": {
                            "pattern": {"type": "string"},
                            "cursor": {"type": "number", "default": -1},
                        },
                        "required": ["pattern"],
                    },
                ),
            ],
        )

    @staticmethod
    def python() -> ToolNamespaceConfig:
        return ToolNamespaceConfig(
            name="python",
            description=(
                "Use this tool to execute Python code in your chain of "
                "thought. The code will not be shown to the user. This tool "
                "should be used for internal reasoning, but not for code that "
                "is intended to be visible to the user (e.g. when creating "
                "plots, tables, or files).\n"
                "\n"
                "When you send a message containing Python code to python, it "
                "will be executed in a stateful Jupyter notebook environment. "
                "python will respond with the output of the execution or time "
                "out after 120.0 seconds. The drive at '/mnt/data' can be "
                "used to save and persist user files. Internet access for this "
                "session is UNKNOWN. Depends on the cluster."
            ),
            tools=[],
        )

    def __repr__(self) -> str:
        return (
            f"ToolNamespaceConfig(name={self.name!r}, "
            f"description={self.description!r}, "
            f"tools={self.tools!r})"
        )


# ---------------------------------------------------------------------------
# Channel config
# ---------------------------------------------------------------------------


class ChannelConfig:
    __slots__ = ("valid_channels", "channel_required")

    def __init__(
        self,
        valid_channels: list[str] | None = None,
        channel_required: bool = False,
    ) -> None:
        self.valid_channels: list[str] = (
            valid_channels if valid_channels is not None else []
        )
        self.channel_required = channel_required

    @classmethod
    def require_channels(cls, channels: list[str]) -> ChannelConfig:
        return cls(valid_channels=channels, channel_required=True)


# ---------------------------------------------------------------------------
# SystemContent & DeveloperContent
# ---------------------------------------------------------------------------


class SystemContent:
    """Content for system-role messages (model identity, tools, etc.)."""

    __slots__ = (
        "model_identity",
        "reasoning_effort",
        "conversation_start_date",
        "knowledge_cutoff",
        "channel_config",
        "tools",
    )

    def __init__(
        self,
        model_identity: str | None = (
            "You are ChatGPT, a large language model trained by OpenAI."
        ),
        reasoning_effort: ReasoningEffort | None = ReasoningEffort.MEDIUM,
        conversation_start_date: str | None = None,
        knowledge_cutoff: str | None = "2024-06",
        channel_config: ChannelConfig | None = None,
        tools: dict[str, ToolNamespaceConfig] | None = None,
    ) -> None:
        self.model_identity = model_identity
        self.reasoning_effort = reasoning_effort
        self.conversation_start_date = conversation_start_date
        self.knowledge_cutoff = knowledge_cutoff
        self.channel_config = (
            channel_config
            if channel_config is not None
            else ChannelConfig.require_channels(["analysis", "commentary", "final"])
        )
        self.tools = tools

    @classmethod
    def new(cls) -> SystemContent:
        return cls()

    # -- Fluent setters (mutate-in-place, return self) ----------------------

    def with_model_identity(self, model_identity: str) -> SystemContent:
        self.model_identity = model_identity
        return self

    def with_reasoning_effort(self, reasoning_effort: ReasoningEffort) -> SystemContent:
        self.reasoning_effort = reasoning_effort
        return self

    def with_conversation_start_date(
        self, conversation_start_date: str
    ) -> SystemContent:
        self.conversation_start_date = conversation_start_date
        return self

    def with_knowledge_cutoff(self, knowledge_cutoff: str) -> SystemContent:
        self.knowledge_cutoff = knowledge_cutoff
        return self

    def with_channel_config(self, channel_config: ChannelConfig) -> SystemContent:
        self.channel_config = channel_config
        return self

    def with_required_channels(self, channels: list[str]) -> SystemContent:
        self.channel_config = ChannelConfig.require_channels(channels)
        return self

    def with_tools(self, ns_config: ToolNamespaceConfig) -> SystemContent:
        if self.tools is None:
            self.tools = {}
        self.tools[ns_config.name] = ns_config
        return self

    def with_browser_tool(self) -> SystemContent:
        return self.with_tools(ToolNamespaceConfig.browser())

    def with_python_tool(self) -> SystemContent:
        return self.with_tools(ToolNamespaceConfig.python())

    @property
    def text(self) -> str:
        """Compatibility shim: content-accessing code expects ``.text``."""
        return ""

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"type": "system_content"}
        if self.model_identity is not None:
            out["model_identity"] = self.model_identity
        if self.reasoning_effort is not None:
            out["reasoning_effort"] = self.reasoning_effort.value
        if self.conversation_start_date is not None:
            out["conversation_start_date"] = self.conversation_start_date
        if self.knowledge_cutoff is not None:
            out["knowledge_cutoff"] = self.knowledge_cutoff
        return out


class DeveloperContent:
    """Content for developer-role messages (instructions, function tools)."""

    __slots__ = ("instructions", "tools")

    def __init__(
        self,
        instructions: str | None = None,
        tools: dict[str, ToolNamespaceConfig] | None = None,
    ) -> None:
        self.instructions = instructions
        self.tools = tools

    @classmethod
    def new(cls) -> DeveloperContent:
        return cls()

    def with_instructions(self, instructions: str) -> DeveloperContent:
        self.instructions = instructions
        return self

    def with_tools(self, ns_config: ToolNamespaceConfig) -> DeveloperContent:
        if self.tools is None:
            self.tools = {}
        self.tools[ns_config.name] = ns_config
        return self

    def with_function_tools(self, tools: Sequence[ToolDescription]) -> DeveloperContent:
        return self.with_tools(
            ToolNamespaceConfig(name="functions", description=None, tools=list(tools))
        )

    @property
    def text(self) -> str:
        """Compatibility shim: content-accessing code expects ``.text``."""
        return ""

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"type": "developer_content"}
        if self.instructions is not None:
            out["instructions"] = self.instructions
        return out


# ---------------------------------------------------------------------------
# Message
# ---------------------------------------------------------------------------

# Union of all content types that can appear in a message.
Content = TextContent | SystemContent | DeveloperContent


class Message:
    """A single Harmony protocol message."""

    __slots__ = ("author", "content", "channel", "recipient", "content_type")

    def __init__(
        self,
        author: Author,
        content: list[Content] | None = None,
        channel: str | None = None,
        recipient: str | None = None,
        content_type: str | None = None,
    ) -> None:
        self.author = author
        self.content: list[Content] = content if content is not None else []
        self.channel = channel
        self.recipient = recipient
        self.content_type = content_type

    # -- Factory constructors -----------------------------------------------

    @classmethod
    def from_author_and_content(
        cls,
        author: Author,
        content: str | Content | None,
    ) -> Message:
        if content is None:
            return cls(author=author, content=[])
        if isinstance(content, str):
            content = TextContent(text=content)
        return cls(author=author, content=[content])

    @classmethod
    def from_author_and_contents(
        cls,
        author: Author,
        contents: Sequence[Content],
    ) -> Message:
        return cls(author=author, content=list(contents))

    @classmethod
    def from_role_and_content(
        cls,
        role: Role | str,
        content: str | Content | None,
    ) -> Message:
        return cls.from_author_and_content(Author(role=role), content)

    @classmethod
    def from_role_and_contents(
        cls,
        role: Role | str,
        contents: Sequence[Content],
    ) -> Message:
        return cls(author=Author(role=role), content=list(contents))

    # -- Builder helpers (mutate in-place, return self) ----------------------

    def adding_content(self, content: str | Content) -> Message:
        if isinstance(content, str):
            content = TextContent(text=content)
        self.content.append(content)
        return self

    def with_channel(self, channel: str) -> Message:
        self.channel = channel
        return self

    def with_recipient(self, recipient: str) -> Message:
        self.recipient = recipient
        return self

    def with_content_type(self, content_type: str) -> Message:
        self.content_type = content_type
        return self

    # -- Dict-like access (backward compat with code expecting dicts) -------

    _ATTR_MAP = {
        "role": lambda self: self.author.role.value,
        "name": lambda self: self.author.name,
        "content": lambda self: self.content,
        "channel": lambda self: self.channel,
        "recipient": lambda self: self.recipient,
        "content_type": lambda self: self.content_type,
    }

    def __getitem__(self, key: str) -> Any:
        getter = self._ATTR_MAP.get(key)
        if getter is not None:
            return getter(self)
        raise KeyError(key)

    def __setitem__(self, key: str, value: Any) -> None:
        if key == "role":
            self.author = Author(role=value, name=self.author.name)
        elif key == "name":
            self.author = Author(role=self.author.role, name=value)
        elif key == "content":
            self.content = value
        elif key == "channel":
            self.channel = value
        elif key == "recipient":
            self.recipient = value
        elif key == "content_type":
            self.content_type = value
        else:
            raise KeyError(key)

    def __contains__(self, key: object) -> bool:
        if key in ("role", "content"):
            return True
        if key == "name":
            return self.author.name is not None
        if key == "channel":
            return self.channel is not None
        if key == "recipient":
            return self.recipient is not None
        if key == "content_type":
            return self.content_type is not None
        return False

    def get(self, key: str, default: Any = None) -> Any:
        try:
            return self[key]
        except KeyError:
            return default

    # -- Serialisation ------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            **self.author.model_dump(),
            "content": [c.to_dict() for c in self.content],
        }
        if self.channel is not None:
            out["channel"] = self.channel
        if self.recipient is not None:
            out["recipient"] = self.recipient
        if self.content_type is not None:
            out["content_type"] = self.content_type
        return out

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Message:
        role = Role(data["role"])
        author = Author(role=role, name=data.get("name"))

        contents: list[Content] = []
        raw_content = data["content"]

        # The Rust side serialises single text contents as a plain string.
        if isinstance(raw_content, str):
            raw_content = [{"type": "text", "text": raw_content}]

        for raw in raw_content:
            ctype = raw.get("type")
            if ctype == "text":
                contents.append(TextContent(text=raw.get("text", "")))
            elif ctype == "system_content":
                contents.append(
                    SystemContent(**{k: v for k, v in raw.items() if k != "type"})
                )
            elif ctype == "developer_content":
                contents.append(
                    DeveloperContent(**{k: v for k, v in raw.items() if k != "type"})
                )
            else:
                raise ValueError(f"Unknown content variant: {raw}")

        msg = cls(author=author, content=contents)
        msg.channel = data.get("channel")
        msg.recipient = data.get("recipient")
        msg.content_type = data.get("content_type")
        return msg

    # -- Pydantic v2 compatibility ------------------------------------------
    # Required because protocol.py uses ``list[Message | dict]`` in a
    # Pydantic model field.

    @classmethod
    def __get_pydantic_core_schema__(cls, source_type: Any, handler: Any) -> Any:
        from pydantic_core import core_schema

        def validate(value: Any) -> Message:
            if isinstance(value, Message):
                return value
            if isinstance(value, dict):
                # If it has "author" key it's already harmony-format
                if "author" in value and isinstance(value.get("author"), dict):
                    author_dict = value["author"]
                    role = author_dict.get("role")
                    name = author_dict.get("name")
                    author = Author(role=role, name=name)

                    raw_content = value.get("content", "")
                    if isinstance(raw_content, str):
                        content_list: list[Content] = [TextContent(text=raw_content)]
                    elif isinstance(raw_content, list):
                        content_list = [
                            TextContent(text=c.get("text", ""))
                            if isinstance(c, dict)
                            else c
                            for c in raw_content
                        ]
                    else:
                        content_list = [TextContent(text="")]

                    msg = cls(author=author, content=content_list)
                    msg.channel = value.get("channel")
                    msg.recipient = value.get("recipient")
                    msg.content_type = value.get("content_type")
                    return msg
                # Otherwise it has "role" key (chat-format)
                if "role" in value:
                    return cls.from_dict(value)
                raise ValueError(
                    f"Cannot validate Message from dict without "
                    f"'role' or 'author': {value}"
                )
            raise ValueError(f"Cannot validate Message from {type(value)}")

        def serialize(value: Message) -> dict[str, Any]:
            return value.to_dict()

        python_schema = core_schema.no_info_plain_validator_function(validate)
        return core_schema.json_or_python_schema(
            json_schema=python_schema,
            python_schema=python_schema,
            serialization=core_schema.plain_serializer_function_ser_schema(
                serialize,
                info_arg=False,
            ),
        )

    def __repr__(self) -> str:
        return (
            f"Message(author={self.author!r}, content={self.content!r}, "
            f"channel={self.channel!r}, recipient={self.recipient!r}, "
            f"content_type={self.content_type!r})"
        )

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Message):
            return (
                self.author == other.author
                and self.channel == other.channel
                and self.recipient == other.recipient
                and self.content_type == other.content_type
                and self.content == other.content
            )
        return NotImplemented


# ---------------------------------------------------------------------------
# Conversation
# ---------------------------------------------------------------------------


class Conversation:
    """An ordered list of messages."""

    __slots__ = ("messages",)

    def __init__(self, messages: list[Message] | None = None) -> None:
        self.messages: list[Message] = messages if messages is not None else []

    @classmethod
    def from_messages(cls, messages: Sequence[Message]) -> Conversation:
        return cls(messages=list(messages))

    def __iter__(self):
        return iter(self.messages)

    def to_dict(self) -> dict[str, Any]:
        return {"messages": [m.to_dict() for m in self.messages]}

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(cls, payload: str) -> Conversation:
        data = json.loads(payload)
        return cls(messages=[Message.from_dict(m) for m in data["messages"]])
