# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Mock infrastructure for serving-layer replay tests.

Provides lightweight mock versions of the vLLM engine and config objects
needed to instantiate ``OpenAIServingChat``, ``AnthropicServingMessages``,
and ``OpenAIServingResponses`` without a GPU.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock

if TYPE_CHECKING:
    from vllm.entrypoints.anthropic.serving import (
        AnthropicServingMessages,
    )

from vllm.config import MultiModalConfig
from vllm.entrypoints.openai.chat_completion.serving import OpenAIServingChat
from vllm.entrypoints.openai.models.serving import (
    BaseModelPath,
    OpenAIServingModels,
)
from vllm.entrypoints.openai.responses.serving import OpenAIServingResponses
from vllm.entrypoints.serve.render.serving import OpenAIServingRender
from vllm.parser.abstract_parser import Parser

MODEL_NAME = "openai-community/gpt2"
CHAT_TEMPLATE = "Dummy chat template for testing {}"
BASE_MODEL_PATHS = [
    BaseModelPath(name=MODEL_NAME, model_path=MODEL_NAME),
]


@dataclass
class MockHFConfig:
    model_type: str = "any"


@dataclass
class MockModelConfig:
    task = "generate"
    runner_type = "generate"
    model = MODEL_NAME
    tokenizer = MODEL_NAME
    trust_remote_code = False
    tokenizer_mode = "auto"
    max_model_len = 100
    tokenizer_revision = None
    multimodal_config = MultiModalConfig()
    hf_config = MockHFConfig()
    hf_text_config = MockHFConfig()
    logits_processors: list[str] | None = None
    diff_sampling_param: dict | None = None
    allowed_local_media_path: str = ""
    allowed_media_domains: list[str] | None = None
    encoder_config = None
    generation_config: str = "auto"
    override_generation_config: dict[str, Any] = field(default_factory=dict)
    media_io_kwargs: dict[str, dict[str, Any]] = field(default_factory=dict)
    skip_tokenizer_init: bool = False
    is_encoder_decoder: bool = False
    is_multimodal_model: bool = False
    renderer_num_workers: int = 1
    enable_prompt_embeds: bool = False

    def get_diff_sampling_param(self):
        return self.diff_sampling_param or {}


@dataclass
class MockParallelConfig:
    _api_process_rank: int = 0


@dataclass
class MockVllmConfig:
    model_config: MockModelConfig
    parallel_config: MockParallelConfig


@dataclass
class MockEngine:
    model_config: MockModelConfig = field(default_factory=MockModelConfig)
    input_processor: MagicMock = field(default_factory=MagicMock)
    renderer: MagicMock = field(default_factory=MagicMock)
    vllm_config: MagicMock = field(default_factory=MagicMock)


def _build_serving_render(engine, model_registry) -> OpenAIServingRender:
    return OpenAIServingRender(
        model_config=engine.model_config,
        renderer=engine.renderer,
        model_registry=model_registry,
        request_logger=None,
        chat_template=CHAT_TEMPLATE,
        chat_template_content_format="auto",
    )


def _common_serving_args():
    engine = MockEngine()
    models = OpenAIServingModels(
        engine_client=engine, base_model_paths=BASE_MODEL_PATHS
    )
    render = _build_serving_render(engine, models.registry)
    return engine, models, render


def build_serving_chat(
    parser_cls: type[Parser] | None = None,
) -> OpenAIServingChat:
    """Build an ``OpenAIServingChat`` with the given parser class.

    The serving object is usable without a GPU — the engine is mocked and
    the parser class is injected directly.
    """
    engine, models, render = _common_serving_args()
    serving_chat = OpenAIServingChat(
        engine,
        models,
        response_role="assistant",
        openai_serving_render=render,
        chat_template=CHAT_TEMPLATE,
        chat_template_content_format="auto",
        request_logger=None,
    )
    serving_chat.parser_cls = parser_cls
    return serving_chat


def build_anthropic_serving(
    parser_cls: type[Parser] | None = None,
) -> AnthropicServingMessages:
    """Build an ``AnthropicServingMessages`` with the given parser class."""
    from vllm.entrypoints.anthropic.serving import AnthropicServingMessages

    engine, models, render = _common_serving_args()
    serving = AnthropicServingMessages(
        engine,
        models,
        response_role="assistant",
        openai_serving_render=render,
        chat_template=CHAT_TEMPLATE,
        chat_template_content_format="auto",
        request_logger=None,
    )
    serving.parser_cls = parser_cls
    return serving


def build_serving_responses(
    parser_cls: type[Parser] | None = None,
) -> OpenAIServingResponses:
    """Build an ``OpenAIServingResponses`` with the given parser class."""
    engine, models, render = _common_serving_args()
    serving = OpenAIServingResponses(
        engine,
        models,
        openai_serving_render=render,
        chat_template=CHAT_TEMPLATE,
        chat_template_content_format="auto",
        request_logger=None,
    )
    serving.parser = parser_cls
    return serving
