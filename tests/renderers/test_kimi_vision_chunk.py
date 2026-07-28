# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unified vision-chunk (Kimi-K2.5) video substitution survives forced tokenize.

Content protection and prompt_embeds force ``tokenize=True``, which used to leave
``replace_vision_chunk_video_placeholder`` a token-id list it silently ignored,
dropping per-video prompt expansion. The renderer now expands video placeholders
on the rendered *string* before tokenizing, so the expansion happens regardless
of the forced tokenize setting.
"""

from dataclasses import dataclass, field
from typing import Any

import pytest

from vllm.renderers.hf import HfRenderer
from vllm.tokenizers import get_tokenizer

pytestmark = pytest.mark.skip_global_cleanup

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
VIDEO_PLACEHOLDER = "<VIDEO_CHUNK>"
VIDEO_TEMPLATE = (
    "{% for m in messages %}{{ m.content }}" + VIDEO_PLACEHOLDER + "{% endfor %}"
)


@dataclass
class _MockHFConfig:
    model_type: str = "qwen2"
    use_unified_vision_chunk: bool = True
    video_placeholder: str = VIDEO_PLACEHOLDER


@dataclass
class _MockModelConfig:
    runner_type: str = "generate"
    task: str = "generate"
    model: str = MODEL_NAME
    tokenizer: str = MODEL_NAME
    trust_remote_code: bool = False
    tokenizer_revision: Any = None
    tokenizer_mode: str = "auto"
    hf_config: Any = field(default_factory=_MockHFConfig)
    encoder_config: Any = None
    allowed_local_media_path: str = ""
    allowed_media_domains: Any = None
    enable_prompt_embeds: bool = False
    skip_tokenizer_init: bool = False
    is_encoder_decoder: bool = False
    is_multimodal_model: bool = False
    renderer_num_workers: int = 1
    multimodal_config: Any = None


@dataclass
class _MockParallelConfig:
    _api_process_rank: int = 0


@dataclass
class _MockVllmConfig:
    model_config: _MockModelConfig
    parallel_config: _MockParallelConfig = field(default_factory=_MockParallelConfig)


@pytest.fixture(scope="module")
def tokenizer():
    return get_tokenizer(MODEL_NAME)


@pytest.fixture
def renderer(monkeypatch, tokenizer):
    monkeypatch.setenv("VLLM_CHAT_CONTENT_PROTECTION", "0")
    return HfRenderer(_MockVllmConfig(_MockModelConfig()), tokenizer)


def _mm_data():
    # One video (video_idx 0) whose per-chunk prompt must land in the output.
    return {
        "vision_chunk": [
            {
                "type": "video_chunk",
                "video_idx": 0,
                "prompt": "PER_VIDEO_PROMPT",
                "uuid": "u0",
            }
        ]
    }


def _kwargs(tokenize):
    return {"chat_template": VIDEO_TEMPLATE, "tokenize": tokenize, "return_dict": False}


def test_video_substitution_survives_forced_tokenize(renderer, tokenizer):
    # tokenize=True mirrors content-protection / prompt_embeds forcing. The
    # per-video prompt must still be expanded (and the raw placeholder gone),
    # which the pre-fix post-tokenize path silently skipped.
    conversation = [{"role": "user", "content": "watch this"}]
    out = renderer._render_with_video_chunk_substitution(
        tokenizer, conversation, _kwargs(tokenize=True), _mm_data(), VIDEO_PLACEHOLDER
    )
    assert isinstance(out, list)
    decoded = tokenizer.decode(out)
    assert "PER_VIDEO_PROMPT" in decoded
    assert VIDEO_PLACEHOLDER not in decoded


def test_video_substitution_returns_string_when_not_tokenized(renderer, tokenizer):
    conversation = [{"role": "user", "content": "watch this"}]
    out = renderer._render_with_video_chunk_substitution(
        tokenizer, conversation, _kwargs(tokenize=False), _mm_data(), VIDEO_PLACEHOLDER
    )
    assert isinstance(out, str)
    assert "PER_VIDEO_PROMPT" in out
    assert VIDEO_PLACEHOLDER not in out


@pytest.mark.asyncio
async def test_video_substitution_sync_async_parity(renderer, tokenizer):
    conversation = [{"role": "user", "content": "watch this"}]
    sync_out = renderer._render_with_video_chunk_substitution(
        tokenizer, conversation, _kwargs(tokenize=True), _mm_data(), VIDEO_PLACEHOLDER
    )
    async_out = await renderer._render_with_video_chunk_substitution_async(
        tokenizer, conversation, _kwargs(tokenize=True), _mm_data(), VIDEO_PLACEHOLDER
    )
    assert sync_out == async_out
