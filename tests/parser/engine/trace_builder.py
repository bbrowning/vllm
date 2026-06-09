# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""On-demand trace builder for parser engine testing and benchmarks.

Generates token sequences programmatically from model-agnostic scenario
definitions.  Each model format handler knows how to render scenarios
into the model's output format, tokenize them with correct special token
IDs, and compute expected parse outputs.

Every generated sample is self-validated by replaying it through the
real parser before being returned.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from tests.parser.engine.replay_harness import (
    MockTokenizer,
    Sample,
    assert_parse_output,
    collect_output,
    replay_streaming,
)
from vllm.parser.engine.registered_adapters import (
    DeepSeekV4Parser,
    Gemma4Parser,
    NemotronV3Parser,
    Qwen3Parser,
)

# ── Data structures ──────────────────────────────────────────────────


@dataclass
class ToolCallSpec:
    name: str
    arguments: dict[str, Any]


@dataclass
class Scenario:
    id: str
    description: str
    reasoning: str | None = None
    content: str | None = None
    tool_calls: list[ToolCallSpec] | None = None


# ── Scenarios ────────────────────────────────────────────────────────

_READ_TOOL = ToolCallSpec("read_file", {"path": "/tmp/test.txt"})
_BASH_TOOL = ToolCallSpec(
    "bash", {"command": "hostname", "description": "Get hostname"}
)
_WEATHER_TOOL = ToolCallSpec(
    "get_weather",
    {"city": "Dallas", "state": "TX", "unit": "fahrenheit"},
)
_COMPLEX_TOOL = ToolCallSpec(
    "search",
    {
        "query": "vllm parser",
        "filters": {"language": "python", "min_stars": 100},
        "tags": ["ml", "inference"],
        "limit": 10,
        "verbose": True,
    },
)

SCENARIOS: list[Scenario] = [
    Scenario(
        id="think-then-tool",
        description="Reasoning then single tool call",
        reasoning="Let me check the file.",
        tool_calls=[_READ_TOOL],
    ),
    Scenario(
        id="think-then-parallel-tools",
        description="Reasoning then two parallel tool calls",
        reasoning="I need to run both commands.",
        tool_calls=[_BASH_TOOL, _WEATHER_TOOL],
    ),
    Scenario(
        id="think-then-content",
        description="Reasoning then content response",
        reasoning="Let me think about this carefully.",
        content="The answer is 42.",
    ),
    Scenario(
        id="content-only",
        description="Plain content response without reasoning",
        content="Hello! How can I help you today?",
    ),
    Scenario(
        id="tool-only",
        description="Tool call without reasoning",
        tool_calls=[_READ_TOOL],
    ),
    Scenario(
        id="complex-json-args",
        description="Tool call with nested objects, arrays, numbers, booleans",
        reasoning="This needs a complex query.",
        tool_calls=[_COMPLEX_TOOL],
    ),
    Scenario(
        id="whitespace-before-tool",
        description="Whitespace-only content before tool call",
        content="\n\n",
        tool_calls=[_WEATHER_TOOL],
    ),
    Scenario(
        id="think-content-tool",
        description="Reasoning, content, then tool call",
        reasoning="Let me analyze and then fetch data.",
        content="Checking the weather now.",
        tool_calls=[_WEATHER_TOOL],
    ),
    Scenario(
        id="think-whitespace-tool",
        description="Reasoning, whitespace-only gap, then tool call",
        reasoning="Let me check the file contents.",
        content="\n\n",
        tool_calls=[_READ_TOOL],
    ),
    Scenario(
        id="empty-reasoning-content",
        description="Empty reasoning section followed by content",
        reasoning="",
        content="The epoch timestamp is 1779111346.",
    ),
]


# ── Tokenization ─────────────────────────────────────────────────────


def _word_split(text: str) -> list[str]:
    """Split text into word-like tokens, preserving all characters."""
    if not text:
        return []
    parts: list[str] = []
    current = ""
    for ch in text:
        if ch in " \t\n\r" and current and current[-1] not in " \t\n\r":
            parts.append(current)
            current = ch
        else:
            current += ch
    if current:
        parts.append(current)
    return parts


def _tokenize(
    segments: list[tuple[str, bool]],
    vocab: dict[str, int],
    start_id: int = 100,
) -> list[tuple[int, str]]:
    """Build token list from segments.

    Each segment is ``(text, is_special)``.  Special segments use vocab
    IDs; content segments are word-split with sequential IDs.
    """
    tokens: list[tuple[int, str]] = []
    next_id = start_id

    for text, is_special in segments:
        if not text:
            continue
        if is_special:
            tid = vocab.get(text)
            if tid is None:
                raise ValueError(f"Special token {text!r} not in vocab")
            tokens.append((tid, text))
        else:
            for word in _word_split(text):
                tokens.append((next_id, word))
                next_id += 1

    return tokens


# ── Tool definitions ─────────────────────────────────────────────────


def _tool_defs(tool_calls: list[ToolCallSpec]) -> list[dict]:
    """Generate OpenAI-style tool definitions from tool call specs."""
    seen: set[str] = set()
    tools: list[dict] = []
    for tc in tool_calls:
        if tc.name in seen:
            continue
        seen.add(tc.name)
        properties: dict[str, dict] = {}
        for key, value in tc.arguments.items():
            if isinstance(value, bool):
                properties[key] = {"type": "boolean"}
            elif isinstance(value, int):
                properties[key] = {"type": "integer"}
            elif isinstance(value, float):
                properties[key] = {"type": "number"}
            elif isinstance(value, str):
                properties[key] = {"type": "string"}
            elif isinstance(value, list):
                properties[key] = {"type": "array"}
            elif isinstance(value, dict):
                properties[key] = {"type": "object"}
        tools.append(
            {
                "type": "function",
                "function": {
                    "name": tc.name,
                    "parameters": {
                        "type": "object",
                        "properties": properties,
                    },
                },
            }
        )
    return tools


# ── Format handlers ──────────────────────────────────────────────────


def _validate_sample(sample: Sample, parser_cls: type, **kwargs) -> None:
    """Replay sample through the real parser and assert correctness."""
    tokenizer = MockTokenizer(vocab=dict(sample.vocab), tokens=sample.tokens)
    parser = parser_cls(tokenizer, sample.tools, **kwargs)
    deltas = replay_streaming(parser, sample.tokens, chunk_size=1, tools=sample.tools)
    output = collect_output(deltas)
    assert_parse_output(output, sample)


def _make_sample(
    sample_id: str,
    description: str,
    vocab: dict[str, int],
    segments: list[tuple[str, bool]],
    expected_reasoning: str | None,
    expected_content: str | None,
    expected_tool_calls: list[dict] | None,
    tools: list[dict] | None,
    chat_template_kwargs: dict | None = None,
) -> Sample:
    tokens = _tokenize(segments, vocab)
    return Sample(
        id=sample_id,
        description=description,
        source="trace-builder",
        vocab=dict(vocab),
        tokens=tokens,
        expected_reasoning=expected_reasoning,
        expected_content=expected_content,
        expected_tool_calls=expected_tool_calls,
        tools=tools,
        chat_template_kwargs=chat_template_kwargs,
    )


# ── Qwen3 / NemotronV3 (XML tool format, starts in REASONING) ───────

_QWEN3_VOCAB: dict[str, int] = {
    "<think>": 50,
    "</think>": 51,
    "<tool_call>": 60,
    "</tool_call>": 61,
}


def _qwen3_arg_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)


def _qwen3_tool_segments(tc: ToolCallSpec) -> list[tuple[str, bool]]:
    parts = [f"\n<function={tc.name}>"]
    for key, value in tc.arguments.items():
        parts.append(f"\n<parameter={key}>{_qwen3_arg_value(value)}</parameter>")
    parts.append("\n</function>\n")
    return [
        ("<tool_call>", True),
        ("".join(parts), False),
        ("</tool_call>", True),
    ]


def _qwen3_segments(scenario: Scenario) -> list[tuple[str, bool]]:
    segs: list[tuple[str, bool]] = []
    if scenario.reasoning is not None:
        segs.append((scenario.reasoning, False))
    if scenario.content is not None or scenario.tool_calls:
        segs.append(("</think>", True))
    if scenario.content is not None:
        segs.append((scenario.content, False))
    if scenario.tool_calls:
        for tc in scenario.tool_calls:
            segs.extend(_qwen3_tool_segments(tc))
    return segs


def _qwen3_expected_content(scenario: Scenario) -> str | None:
    if (
        scenario.content is not None
        and scenario.tool_calls
        and not scenario.content.strip()
    ):
        return ""
    return scenario.content


def _build_qwen3(
    scenario: Scenario,
    name: str = "qwen3",
    parser_cls: type = Qwen3Parser,
    strip_trailing_ws: bool = False,
    validate: bool = True,
) -> Sample:
    expected_reasoning: str | None
    if scenario.reasoning is not None:
        r = scenario.reasoning
        if strip_trailing_ws:
            r = r.rstrip()
        expected_reasoning = r
    else:
        expected_reasoning = ""

    expected_tc = (
        [{"name": tc.name, "arguments": tc.arguments} for tc in scenario.tool_calls]
        if scenario.tool_calls
        else None
    )

    sample = _make_sample(
        sample_id=f"{name}-{scenario.id}",
        description=scenario.description,
        vocab=_QWEN3_VOCAB,
        segments=_qwen3_segments(scenario),
        expected_reasoning=expected_reasoning,
        expected_content=_qwen3_expected_content(scenario),
        expected_tool_calls=expected_tc,
        tools=_tool_defs(scenario.tool_calls) if scenario.tool_calls else None,
    )
    if validate:
        _validate_sample(sample, parser_cls)
    return sample


def _build_nemotron_v3(scenario: Scenario, validate: bool = True) -> Sample:
    return _build_qwen3(
        scenario,
        name="nemotron_v3",
        parser_cls=NemotronV3Parser,
        strip_trailing_ws=True,
        validate=validate,
    )


# ── Gemma4 (channel reasoning, custom arg format) ────────────────────

_GEMMA4_VOCAB: dict[str, int] = {
    "<|channel>": 50,
    "<channel|>": 51,
    "<|tool_call>": 48,
    "<tool_call|>": 49,
    '<|"|>': 52,
}
_GEMMA4_THOUGHT_PREFIX = "thought\n"
_GEMMA4_QUOTE = '<|"|>'


def _gemma4_value_segments(value: Any) -> list[tuple[str, bool]]:
    """Render a value in Gemma4 arg format as segments."""
    if isinstance(value, str):
        return [(_GEMMA4_QUOTE, True), (value, False), (_GEMMA4_QUOTE, True)]
    if isinstance(value, bool):
        return [("true" if value else "false", False)]
    if isinstance(value, (int, float)):
        return [(str(value), False)]
    if isinstance(value, dict):
        segs: list[tuple[str, bool]] = [("{", False)]
        for i, (k, v) in enumerate(value.items()):
            if i > 0:
                segs.append((",", False))
            segs.append((f"{k}:", False))
            segs.extend(_gemma4_value_segments(v))
        segs.append(("}", False))
        return segs
    if isinstance(value, list):
        segs = [("[", False)]
        for i, item in enumerate(value):
            if i > 0:
                segs.append((",", False))
            segs.extend(_gemma4_value_segments(item))
        segs.append(("]", False))
        return segs
    return [(json.dumps(value, ensure_ascii=False), False)]


def _gemma4_tool_segments(tc: ToolCallSpec) -> list[tuple[str, bool]]:
    segs: list[tuple[str, bool]] = [
        ("<|tool_call>", True),
        (f"call:{tc.name}", False),
        ("{", False),
    ]
    for i, (key, value) in enumerate(tc.arguments.items()):
        if i > 0:
            segs.append((",", False))
        segs.append((f"{key}:", False))
        segs.extend(_gemma4_value_segments(value))
    segs.append(("}", False))
    segs.append(("<tool_call|>", True))
    return segs


def _gemma4_segments(scenario: Scenario) -> list[tuple[str, bool]]:
    segs: list[tuple[str, bool]] = []
    if scenario.reasoning is not None:
        segs.append(("<|channel>", True))
        segs.append((_GEMMA4_THOUGHT_PREFIX, False))
        segs.append((scenario.reasoning, False))
        segs.append(("<channel|>", True))
    if scenario.content is not None:
        segs.append((scenario.content, False))
    if scenario.tool_calls:
        for tc in scenario.tool_calls:
            segs.extend(_gemma4_tool_segments(tc))
    return segs


def _build_gemma4(scenario: Scenario, validate: bool = True) -> Sample:
    expected_tc = (
        [{"name": tc.name, "arguments": tc.arguments} for tc in scenario.tool_calls]
        if scenario.tool_calls
        else None
    )

    sample = _make_sample(
        sample_id=f"gemma4-{scenario.id}",
        description=scenario.description,
        vocab=_GEMMA4_VOCAB,
        segments=_gemma4_segments(scenario),
        expected_reasoning=scenario.reasoning,
        expected_content=_qwen3_expected_content(scenario),
        expected_tool_calls=expected_tc,
        tools=_tool_defs(scenario.tool_calls) if scenario.tool_calls else None,
    )
    if validate:
        _validate_sample(sample, Gemma4Parser)
    return sample


# ── DeepSeek V4 (DSML tool format) ──────────────────────────────────

_DSV4_VOCAB: dict[str, int] = {
    "<think>": 128821,
    "</think>": 128822,
}
_DSML = "｜DSML｜"  # ｜DSML｜


def _dsv4_param_text(key: str, value: Any) -> str:
    """Render one DSML parameter tag."""
    if isinstance(value, str):
        return (
            f'<{_DSML}parameter name="{key}" string="true">{value}</{_DSML}parameter>\n'
        )
    if isinstance(value, bool):
        val_str = "true" if value else "false"
    elif isinstance(value, (int, float)):
        val_str = str(value)
    else:
        val_str = json.dumps(value, ensure_ascii=False)
    return (
        f'<{_DSML}parameter name="{key}" string="false">{val_str}</{_DSML}parameter>\n'
    )


def _dsv4_tool_text(tc: ToolCallSpec) -> str:
    parts = [f'<{_DSML}invoke name="{tc.name}">\n']
    for key, value in tc.arguments.items():
        parts.append(_dsv4_param_text(key, value))
    parts.append(f"</{_DSML}invoke>\n")
    return "".join(parts)


def _dsv4_segments(scenario: Scenario, thinking: bool) -> list[tuple[str, bool]]:
    segs: list[tuple[str, bool]] = []

    if thinking:
        if scenario.reasoning is not None:
            segs.append((scenario.reasoning, False))
        if scenario.content is not None or scenario.tool_calls:
            segs.append(("</think>", True))
    else:
        if scenario.reasoning is not None:
            segs.append(("<think>", True))
            segs.append((scenario.reasoning, False))
            segs.append(("</think>", True))

    if scenario.content is not None:
        segs.append((scenario.content, False))

    if scenario.tool_calls:
        parts = [f"<{_DSML}tool_calls>\n"]
        for tc in scenario.tool_calls:
            parts.append(_dsv4_tool_text(tc))
        parts.append(f"</{_DSML}tool_calls>")
        segs.append(("".join(parts), False))

    return segs


def _build_deepseek_v4(scenario: Scenario, validate: bool = True) -> Sample:
    thinking = scenario.reasoning is not None
    chat_kwargs = {"thinking": True} if thinking else None

    if thinking:
        expected_reasoning: str | None = scenario.reasoning or ""
    else:
        expected_reasoning = None

    expected_tc = (
        [{"name": tc.name, "arguments": tc.arguments} for tc in scenario.tool_calls]
        if scenario.tool_calls
        else None
    )

    sample = _make_sample(
        sample_id=f"deepseek_v4-{scenario.id}",
        description=scenario.description,
        vocab=_DSV4_VOCAB,
        segments=_dsv4_segments(scenario, thinking),
        expected_reasoning=expected_reasoning,
        expected_content=_qwen3_expected_content(scenario),
        expected_tool_calls=expected_tc,
        tools=_tool_defs(scenario.tool_calls) if scenario.tool_calls else None,
        chat_template_kwargs=chat_kwargs,
    )
    if validate:
        kwargs = {}
        if chat_kwargs:
            kwargs["chat_template_kwargs"] = chat_kwargs
        _validate_sample(sample, DeepSeekV4Parser, **kwargs)
    return sample


# ── Registry and public API ──────────────────────────────────────────

_BUILDERS: dict[str, Any] = {
    "qwen3": _build_qwen3,
    "nemotron_v3": _build_nemotron_v3,
    "gemma4": _build_gemma4,
    "deepseek_v4": _build_deepseek_v4,
}


def build_samples(model: str) -> list[Sample]:
    """Build all scenario samples for a model, self-validated."""
    builder = _BUILDERS[model]
    return [builder(s) for s in SCENARIOS]


def build_sample(model: str, scenario: Scenario) -> Sample:
    """Build a single sample for one model + scenario."""
    return _BUILDERS[model](scenario)


_MODEL_TO_PARSER_NAME: dict[str, str] = {
    "qwen3": "qwen3_engine",
    "gemma4": "gemma4_engine",
    "deepseek_v4": "deepseek_v4_engine",
    "nemotron_v3": "nemotron_v3_engine",
}


def build_serving_samples(model: str) -> list:
    """Build serving-layer samples for a model.

    Returns a list of ``ServingSample`` objects (from the serving replay
    harness) with parser metadata derived from the model name and each
    sample's expected output.
    """
    from tests.entrypoints.serving_replay.replay_harness import (
        ServingSample,
    )

    parser_name = _MODEL_TO_PARSER_NAME[model]
    result = []
    for sample in build_samples(model):
        has_tools = sample.expected_tool_calls is not None
        result.append(
            ServingSample(
                id=sample.id,
                description=sample.description,
                source=sample.source,
                vocab=sample.vocab,
                tokens=sample.tokens,
                expected_reasoning=sample.expected_reasoning,
                expected_content=sample.expected_content,
                expected_tool_calls=sample.expected_tool_calls,
                parser_name=parser_name,
                tool_choice="auto" if has_tools else "none",
                tools=sample.tools,
                include_reasoning=sample.expected_reasoning is not None,
                expected_finish_reason=("tool_calls" if has_tools else "stop"),
                chat_template_kwargs=sample.chat_template_kwargs,
            )
        )
    return result


def build_scaling_sample(
    model: str, token_count: int, validate: bool = False
) -> Sample:
    """Build a sample with approximately *token_count* tokens."""
    sentence = "The quick brown fox jumps over the lazy dog. "
    text = sentence * (token_count // 10 + 1)
    scenario = Scenario(
        id=f"scaling-{token_count}",
        description=f"Scaling test with ~{token_count} tokens",
        reasoning=text,
        tool_calls=[_READ_TOOL],
    )
    return _BUILDERS[model](scenario, validate=validate)
