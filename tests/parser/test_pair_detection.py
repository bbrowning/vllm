# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Tests for ParserManager pair detection.

When both --reasoning-parser and --tool-call-parser resolve to engine
adapters backed by the same (or related) ParserEngine class,
ParserManager should return that engine directly instead of wrapping
them in a DelegatingParser.
"""

import os

import pytest

_STRICT_TOOL_CALLING_ENV = "VLLM_ENFORCE_STRICT_TOOL_CALLING"
_STRICT_TOOL_CALLING_ENV_VALUE = os.environ.get(_STRICT_TOOL_CALLING_ENV)
os.environ[_STRICT_TOOL_CALLING_ENV] = "0"

from vllm.parser.abstract_parser import DelegatingParser  # noqa: E402
from vllm.parser.engine.parser_engine import ParserEngine  # noqa: E402
from vllm.parser.parser_manager import ParserManager  # noqa: E402


@pytest.fixture(scope="module", autouse=True)
def restore_strict_tool_calling_env():
    yield
    if _STRICT_TOOL_CALLING_ENV_VALUE is None:
        os.environ.pop(_STRICT_TOOL_CALLING_ENV, None)
    else:
        os.environ[_STRICT_TOOL_CALLING_ENV] = _STRICT_TOOL_CALLING_ENV_VALUE


_SAME_ENGINE_PAIRS = [
    ("qwen3", "qwen3_coder"),
    ("qwen3", "qwen3_xml"),
    ("mimo", "mimo"),
    ("seed_oss", "seed_oss"),
    ("glm45", "glm45"),
    ("glm47", "glm47"),
    ("glm47", "glm45"),
    ("minimax_m2", "minimax_m2"),
    ("gemma4", "gemma4"),
]


@pytest.mark.parametrize("reasoning_name,tool_name", _SAME_ENGINE_PAIRS)
def test_same_engine_pair_returns_parser_engine(reasoning_name, tool_name):
    parser_cls = ParserManager.get_parser(
        reasoning_parser_name=reasoning_name,
        tool_parser_name=tool_name,
        enable_auto_tools=True,
    )
    assert parser_cls is not None
    assert issubclass(parser_cls, ParserEngine)
    assert not issubclass(parser_cls, DelegatingParser)


_CROSS_ENGINE_PAIRS = [
    ("nemotron_v3", "qwen3_coder"),
    ("nemotron_v3", "qwen3_xml"),
]


@pytest.mark.parametrize("reasoning_name,tool_name", _CROSS_ENGINE_PAIRS)
def test_cross_engine_pair_returns_parser_engine(reasoning_name, tool_name):
    from vllm.parser.nemotron_v3 import NemotronV3Parser

    parser_cls = ParserManager.get_parser(
        reasoning_parser_name=reasoning_name,
        tool_parser_name=tool_name,
        enable_auto_tools=True,
    )
    assert parser_cls is not None
    assert issubclass(parser_cls, NemotronV3Parser)
    assert not issubclass(parser_cls, DelegatingParser)


def test_non_matching_pair_falls_through_to_delegating():
    parser_cls = ParserManager.get_parser(
        reasoning_parser_name="qwen3",
        tool_parser_name="hermes",
        enable_auto_tools=True,
    )
    assert parser_cls is not None
    assert issubclass(parser_cls, DelegatingParser)


def test_reasoning_only_falls_through_to_delegating():
    parser_cls = ParserManager.get_parser(
        reasoning_parser_name="qwen3",
        tool_parser_name=None,
        enable_auto_tools=False,
    )
    assert parser_cls is not None
    assert issubclass(parser_cls, DelegatingParser)


def test_tool_only_falls_through_to_delegating():
    parser_cls = ParserManager.get_parser(
        reasoning_parser_name=None,
        tool_parser_name="qwen3_coder",
        enable_auto_tools=True,
    )
    assert parser_cls is not None
    assert issubclass(parser_cls, DelegatingParser)


def test_paired_parser_has_class_attrs():
    parser_cls = ParserManager.get_parser(
        reasoning_parser_name="qwen3",
        tool_parser_name="qwen3_coder",
        enable_auto_tools=True,
    )
    assert parser_cls is not None
    assert parser_cls.reasoning_parser_cls is not None
    assert parser_cls.tool_parser_cls is not None


@pytest.mark.parametrize("reasoning_name,tool_name", _SAME_ENGINE_PAIRS)
def test_paired_engine_inherits_structural_tag_model(reasoning_name, tool_name):
    parser_cls = ParserManager.get_parser(
        reasoning_parser_name=reasoning_name,
        tool_parser_name=tool_name,
        enable_auto_tools=True,
    )
    assert parser_cls is not None
    tool_adapter_cls = parser_cls.tool_parser_cls
    expected = getattr(tool_adapter_cls, "structural_tag_model", None)
    actual = getattr(parser_cls, "structural_tag_model", None)
    assert actual == expected
