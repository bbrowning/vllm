# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fail-closed behavior of the shared BaseRenderer content-protection gate.

VLLM_CHAT_CONTENT_PROTECTION is only implemented by the HF renderer. Every
other renderer must refuse to start when it is opted in, rather than silently
serving unprotected. The gate lives in ``BaseRenderer.__init__``, so it covers
current and future non-HF renderers uniformly (including ones like
``TerratorchRenderer`` that don't override ``__init__``).
"""

from dataclasses import dataclass, field
from typing import Any

import pytest

from vllm.renderers.deepseek_v4 import DeepseekV4Renderer
from vllm.renderers.mistral import MistralRenderer
from vllm.renderers.terratorch import TerratorchRenderer

# CPU-only construction tests; skip the GPU/dist cleanup teardown.
pytestmark = pytest.mark.skip_global_cleanup


@dataclass
class _MockModelConfig:
    renderer_num_workers: int = 1
    is_multimodal_model: bool = False


@dataclass
class _MockParallelConfig:
    _api_process_rank: int = 0


@dataclass
class _MockVllmConfig:
    model_config: Any = field(default_factory=_MockModelConfig)
    parallel_config: Any = field(default_factory=_MockParallelConfig)


# All fail before the config is ever dereferenced (the gate is the first thing
# BaseRenderer.__init__ does), so a minimal config and no tokenizer suffice.
_NON_HF_RENDERERS = [MistralRenderer, TerratorchRenderer, DeepseekV4Renderer]


@pytest.mark.parametrize("renderer_cls", _NON_HF_RENDERERS)
def test_non_hf_renderer_rejects_content_protection(monkeypatch, renderer_cls):
    monkeypatch.setenv("VLLM_CHAT_CONTENT_PROTECTION", "1")
    with pytest.raises(ValueError, match="not supported for the"):
        renderer_cls(_MockVllmConfig(), tokenizer=None)


@pytest.mark.parametrize("renderer_cls", _NON_HF_RENDERERS)
@pytest.mark.parametrize("flag", ["0", None])
def test_non_hf_renderer_allows_disabled_content_protection(
    monkeypatch, renderer_cls, flag
):
    if flag is None:
        monkeypatch.delenv("VLLM_CHAT_CONTENT_PROTECTION", raising=False)
    else:
        monkeypatch.setenv("VLLM_CHAT_CONTENT_PROTECTION", flag)
    # The gate must not fire; construction proceeds past it (and may then fail
    # later for unrelated mock reasons, which is not what we assert here).
    try:
        renderer_cls(_MockVllmConfig(), tokenizer=None)
    except ValueError as exc:
        assert "content protection" not in str(exc).lower()
