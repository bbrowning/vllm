# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

import copy
import inspect
import itertools
import json
import weakref
from collections import defaultdict, deque
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    Final,
    Literal,
    NamedTuple,
    TypeAlias,
    cast,
    overload,
)

import jinja2
import jinja2.ext
import jinja2.meta
import jinja2.nodes
import jinja2.parser
import jinja2.sandbox
import torch
from typing_extensions import override

from vllm import envs
from vllm.entrypoints.chat_utils import (
    PROMPT_EMBEDS_PLACEHOLDER_TOKEN,
    ChatTemplateResolutionError,
    load_chat_template,
    parse_chat_messages,
    parse_chat_messages_async,
)
from vllm.inputs import EmbedsPrompt
from vllm.inputs.engine import MultiModalInput
from vllm.logger import init_logger
from vllm.multimodal.hasher import MultiModalHasher
from vllm.multimodal.inputs import (
    MultiModalFieldElem,
    MultiModalKwargsItem,
    MultiModalKwargsItems,
    MultiModalSharedField,
    PlaceholderRange,
)
from vllm.multimodal.processing.processor import (
    PromptReplacement,
    apply_token_matches,
    find_mm_placeholders,
)
from vllm.tokenizers.hf import HfTokenizer, maybe_make_thread_pool
from vllm.transformers_utils.chat_templates import get_chat_template_fallback_path
from vllm.transformers_utils.processor import cached_get_processor
from vllm.utils.async_utils import make_async
from vllm.utils.func_utils import supports_kw

from .base import BaseRenderer
from .inputs.preprocess import parse_dec_only_prompt

if TYPE_CHECKING:
    from collections.abc import Set

    from vllm.config import ModelConfig, VllmConfig
    from vllm.entrypoints.chat_utils import (
        ChatCompletionMessageParam,
        ChatTemplateContentFormat,
        ChatTemplateContentFormatOption,
        ConversationMessage,
    )
    from vllm.inputs import MultiModalDataDict, MultiModalUUIDDict, TokensPrompt
    from vllm.inputs.engine import TokensInput
    from vllm.multimodal.processing.processor import (
        MultiModalPromptUpdates,
        ResolvedPromptUpdate,
    )

    from .inputs import DictPrompt
    from .params import ChatParams

logger = init_logger(__name__)


# Cache of `tokenizer -> prompt_embeds placeholder token ID`. Keyed by the
# tokenizer object (not `id(tokenizer)`) so a fresh tokenizer landing at a
# recycled memory address can't pick up a stale tid. Entries evict atomically
# with the tokenizer's garbage-collection.
_PROMPT_EMBEDS_PLACEHOLDER_TOKEN_ID_CACHE: Final[
    weakref.WeakKeyDictionary[HfTokenizer, int]
] = weakref.WeakKeyDictionary()
_SINGLE_TOKEN_ID_ERROR: Final[str] = (
    "Expected {token!r} to tokenize to exactly 1 token, got {num_ids} ({ids!r})."
)
_PROMPT_EMBEDS_PLACEHOLDER_SPAN_MISMATCH_ERROR: Final[str] = (
    "Expected {expected} prompt_embeds placeholder spans in the "
    "tokenized prompt, found {actual}."
)
_MISSING_PROMPT_TOKEN_IDS_ERROR: Final[str] = (
    "Expected prompt_token_ids in rendered prompt when prompt_embeds "
    "are present. This indicates the chat template was invoked with "
    "tokenize=False."
)
_TOKENIZE_OVERRIDE_WARNING: Final[str] = (
    "Overriding `tokenize=False` to `True` because `prompt_embeds` "
    "post-processing requires tokenized IDs."
)


def _verify_single_token_id(tokenizer: HfTokenizer, token: str) -> int:
    """Return the single token ID for an already-registered special `token`,
    raising if it does not encode to exactly one ID."""
    ids = tokenizer.encode(token, add_special_tokens=False)
    if len(ids) != 1:
        raise RuntimeError(
            _SINGLE_TOKEN_ID_ERROR.format(token=token, num_ids=len(ids), ids=ids)
        )
    return ids[0]


def _ensure_prompt_embeds_placeholder_token(tokenizer: HfTokenizer) -> int:
    """Register `PROMPT_EMBEDS_PLACEHOLDER_TOKEN` as a special token and return
    its token ID."""
    cached = _PROMPT_EMBEDS_PLACEHOLDER_TOKEN_ID_CACHE.get(tokenizer)
    if cached is not None:
        return cached

    tokenizer.add_special_tokens(
        {"additional_special_tokens": [PROMPT_EMBEDS_PLACEHOLDER_TOKEN]}
    )
    token_id = _verify_single_token_id(tokenizer, PROMPT_EMBEDS_PLACEHOLDER_TOKEN)
    _PROMPT_EMBEDS_PLACEHOLDER_TOKEN_ID_CACHE[tokenizer] = token_id
    return token_id


def _build_prompt_embeds_updates(
    prompt_embeds_tensors: Sequence[torch.Tensor],
    placeholder_token_id: int,
) -> MultiModalPromptUpdates:
    """Build `MultiModalPromptUpdates` for `prompt_embeds` expansion.

    Each tensor produces a `PromptReplacement` that maps
    `[placeholder_token_id]` -> `[placeholder_token_id] x N`
    (where `N = tensor.shape[0]`).
    """
    updates: list[Sequence[ResolvedPromptUpdate]] = []
    for i, tensor in enumerate(prompt_embeds_tensors):
        update = PromptReplacement(
            modality="prompt_embeds",
            target=[placeholder_token_id],
            replacement=[placeholder_token_id] * tensor.shape[0],
        )
        updates.append([update.resolve(item_idx=i)])
    return {"prompt_embeds": updates}


def _expand_prompt_embeds_placeholders(
    token_ids: list[int],
    mm_prompt_updates: MultiModalPromptUpdates,
) -> list[int]:
    """Expand each 1-token `prompt_embeds` sentinel into an N-token span.

    Uses `apply_token_matches`.  Each single placeholder token in
    `token_ids` is replaced with a consecutive span of
    `tensor.shape[0]` copies, following tensors in order.
    """
    expanded, _ = apply_token_matches(token_ids, mm_prompt_updates, tokenizer=None)
    return expanded


def _build_prompt_embeds_positions(
    token_ids: list[int],
    num_tensors: int,
    mm_prompt_updates: MultiModalPromptUpdates,
) -> list[tuple[int, int]]:
    """Locate each prompt_embeds placeholder span in `token_ids`.

    Expects `token_ids` to already contain expanded N-token spans.
    Returns `[(start_idx, length), ...]` aligned with the tensors.
    """
    placeholders = find_mm_placeholders(
        prompt=token_ids,
        mm_prompt_updates=mm_prompt_updates,
        tokenizer=None,
    )
    features = placeholders.get("prompt_embeds", [])

    if len(features) != num_tensors:
        raise ValueError(
            _PROMPT_EMBEDS_PLACEHOLDER_SPAN_MISMATCH_ERROR.format(
                expected=num_tensors,
                actual=len(features),
            )
        )

    return [(f.start_idx, f.length) for f in features]


def _build_mixed_prompt_embeds(
    token_ids: list[int],
    prompt_embeds_tensors: Sequence[torch.Tensor],
    positions: list[tuple[int, int]],
) -> tuple[torch.Tensor, list[bool]]:
    """Build the full-length `prompt_embeds` tensor and the `is_token_ids`
    mask aligned to `token_ids`."""
    total_len = len(token_ids)
    hidden_size = prompt_embeds_tensors[0].shape[1]
    dtype = prompt_embeds_tensors[0].dtype

    full_embeds = torch.zeros(total_len, hidden_size, dtype=dtype)
    is_token_ids = torch.ones(total_len, dtype=torch.bool)

    for (start, length), tensor in zip(positions, prompt_embeds_tensors, strict=True):
        full_embeds[start : start + length] = tensor
        is_token_ids[start : start + length] = False

    return full_embeds, is_token_ids.tolist()


# --------------------------------------------------------------------------- #
# Token-space chat content protection (opt-in: VLLM_CHAT_CONTENT_PROTECTION).
#
# Protection is *selective*: an untrusted string is neutralized only when it
# actually contains an added/special token (i.e. the real tokenizer would encode
# it differently than a "safe backend" that has all added tokens stripped --
# see `_neutralization_safe_ids`). Such a string is replaced by a distinct
# reserved placeholder token before rendering; the structural skeleton is
# tokenized in one pass; then the string's safe-backend encoding is spliced back
# at the placeholder positions. Literal control-token strings therefore tokenize
# to subwords, never real special-token IDs, so they cannot forge
# system/assistant/tool turns. Clean strings (identical encodings) are left in
# place untouched, so the template renders them normally -- benign requests are a
# no-op and per-content transforms (`| trim`, `| tojson`, ...) apply correctly.
#
# Protected strings are every message's content (str, or each openai text /
# tool_reference part) and `name`, plus assistant `tool_calls` `function.name`
# and the string leaves and keys of `function.arguments` (a dict, list, or a raw
# JSON string -- all of which render raw/adjacent to control tokens). Coverage
# is role-agnostic: a stateless chat request lets a caller put anything under
# any role, so restricting protection to "untrusted" roles would not stop a
# client from forging turns anyway (they could just send role="system"
# directly) -- the point is that literal text never silently turns into a real
# special-token ID, no matter which role carries it. This includes assistant
# content: a model emitting a structural marker like `<think>`/`</think>` as
# plain decoded text (e.g. no reasoning parser configured) and having it
# resubmitted is the same shape of problem as a client typing it directly, so
# it is neutralized the same way -- callers that want faithful reasoning
# round-tripping should resubmit reasoning via the `reasoning`/
# `reasoning_content` message field with a reasoning parser configured, not
# rely on think-tokens surviving in plain `content`. Assistant `tool_calls`
# `id` and tool-message `tool_call_id` are validated and rejected rather than
# spliced (see `_validate_tool_call_ids`). Media parts and non-string scalars
# pass through.
#
# A neutralized string never passes through the Jinja template, so per-content
# transforms are not reproduced for it: it is spliced verbatim as isolated
# safe-backend tokens. For content that goes through `| tojson` this can yield
# malformed JSON (quotes/backslashes/newlines in the value are not re-escaped),
# but this only ever happens for a string that genuinely carries a control token
# -- i.e. while neutralizing an injection -- never for clean function-call args.
# When a template drops (0x) or duplicates (>1x) a slot -- or a transform mangles
# the placeholder so it no longer appears as its own token -- splicing degrades
# gracefully (omit / splice-into-all) rather than failing closed; content is
# omitted, never forged.
#
# Safety of leaving clean strings in the skeleton: a string clean in isolation
# could in theory merge with adjacent template text to form a token, but only if
# a template emitted a bare control-token-completing fragment right next to
# content. Real templates emit special tokens whole (`<|im_end|>`, `[/INST]`), so
# this is non-exploitable in practice. The safe backend is shared across worker
# threads (see `_make_backend_from_json` for the thread-safety rationale).
# --------------------------------------------------------------------------- #

# A pool of distinct reserved placeholder tokens, one per untrusted text slot.
# Distinct (not shared) tokens let us map each rendered placeholder back to its
# source slot by *token identity*, which is required for templates that re-order
# content relative to conversation order (e.g. Gemma-4 forward-scans `tool`
# messages onto the preceding assistant turn).
_UNTRUSTED_CONTENT_PLACEHOLDER_TEMPLATE: Final[str] = "<|vllm_untrusted_content_{}|>"
UNTRUSTED_CONTENT_PLACEHOLDER_PREFIX: Final[str] = "<|vllm_untrusted_content"
# The whole pool is pre-registered at init so per-request rendering never mutates
# the thread-shared tokenizer. Placeholders never reach the model -- they are
# spliced out before token IDs go to the engine -- so a large cap costs only a
# one-time registration, not vocab/accuracy. Full tool-call/argument protection
# creates a slot per string value and key, so keep this comfortably above
# realistic agentic requests; oversized requests fail closed.
_UNTRUSTED_PLACEHOLDER_POOL_CAP: Final[int] = 512

# Keyed by tokenizer identity: these caches use the default object hash/eq
# (identity). Neither the tokenizer class nor the `TokenizerPool` subclass that
# `maybe_make_thread_pool` swaps in overrides `__eq__`/`__hash__`, so identity
# hashing survives the in-place class swap -- a cached entry keeps hitting for the
# same object after the swap.
_UNTRUSTED_PLACEHOLDER_POOL_CACHE: Final[
    weakref.WeakKeyDictionary[HfTokenizer, list[int]]
] = weakref.WeakKeyDictionary()
_UNTRUSTED_SAFE_BACKEND_CACHE: Final[
    weakref.WeakKeyDictionary[HfTokenizer, _SafeBackends]
] = weakref.WeakKeyDictionary()

_CONTENT_PROTECTION_SPAN_COUNT_MSG: Final[str] = (
    "Content protection: an untrusted-content placeholder appeared %d time(s) in "
    "the rendered token skeleton (expected once). The template dropped or "
    "duplicated this content; splicing accordingly. Content is still emitted only "
    "as isolated safe-backend tokens, so no injection is possible."
)
_CONTENT_PROTECTION_POOL_EXHAUSTED_ERROR: Final[str] = (
    "Content protection needs {count} untrusted-content placeholders but the "
    "pool cap is {cap}. Refusing to render (fail closed)."
)
_CONTENT_PROTECTION_RESERVED_IN_TEXT_ERROR: Final[str] = (
    f"Message content may not contain the reserved content-protection "
    f"placeholder prefix {UNTRUSTED_CONTENT_PLACEHOLDER_PREFIX!r}."
)
_CONTENT_PROTECTION_NOT_FAST_ERROR: Final[str] = (
    "VLLM_CHAT_CONTENT_PROTECTION requires a fast HF tokenizer; this "
    "tokenizer is not fast."
)
_CONTENT_PROTECTION_UNSUPPORTED_TOKENIZER_ERROR: Final[str] = (
    "VLLM_CHAT_CONTENT_PROTECTION is only supported for HF tokenizers."
)
_CONTENT_PROTECTION_MISTRAL_ERROR: Final[str] = (
    "VLLM_CHAT_CONTENT_PROTECTION is not supported for Mistral tokenizers."
)
_CONTENT_PROTECTION_PROMPT_EMBEDS_ERROR: Final[str] = (
    "VLLM_CHAT_CONTENT_PROTECTION cannot be combined with enable_prompt_embeds."
)
_CONTENT_PROTECTION_ASSISTANT_MASK_ERROR: Final[str] = (
    "VLLM_CHAT_CONTENT_PROTECTION cannot be combined with return_assistant_tokens_mask."
)
_CONTENT_PROTECTION_MULTIMODAL_STRING_ERROR: Final[str] = (
    "VLLM_CHAT_CONTENT_PROTECTION cannot protect multimodal requests rendered "
    "with the 'string' chat-template content format: media placeholders are "
    "flattened into message content as special tokens and would be neutralized "
    "along with injected ones. Use the 'openai' content format "
    "(--chat-template-content-format openai), which keeps media as structured "
    "parts and lets the template emit their tokens directly."
)
_CONTENT_PROTECTION_SAFE_BACKEND_LEAK_ERROR: Final[str] = (
    "VLLM_CHAT_CONTENT_PROTECTION safe backend failed to neutralize "
    "reserved token(s) that still re-encode to their own single ID: "
    "{tokens}. Refusing to enable content protection (fail closed)."
)
_CONTENT_PROTECTION_TOOL_CALL_ID_ERROR: Final[str] = (
    "Tool-call id/tool_call_id may not contain control tokens that encode to "
    "reserved special-token IDs. Refusing to render (fail closed)."
)
_CONTENT_PROTECTION_TOKENIZE_OVERRIDE_WARNING: Final[str] = (
    "Overriding `tokenize=False` to `True` because "
    "VLLM_CHAT_CONTENT_PROTECTION splices token IDs."
)
_VIDEO_CHUNK_ASSISTANT_MASK_ERROR: Final[str] = (
    "return_assistant_tokens_mask cannot be combined with unified vision-chunk "
    "video substitution: the mask forces tokenized output, leaving no rendered "
    "string to expand video placeholders in. Refusing to render (fail closed)."
)


@dataclass
class _UntrustedSlot:
    """One untrusted string region that needs neutralization.

    A slot is created only for strings that actually contain an added/special
    token (see `_neutralization_safe_ids`); clean strings are left verbatim.
    `placeholder_id` is the distinct reserved token standing in for this string
    in the rendered skeleton; `safe_ids` is the string's safe-backend encoding,
    cached here at swap time so splicing needs no re-encode. Splicing matches by
    `placeholder_id` (token identity), so slot order does not affect the result.
    """

    placeholder_id: int
    safe_ids: list[int]


# Callback for `_transform_message`: given an untrusted string, return its
# replacement (the same object, by identity, when unchanged -- this drives the
# copy-on-write throughout the walk).
_SlotFn: TypeAlias = Callable[[str], str]


class _ContentProtection(NamedTuple):
    skeleton_conversation: list[ConversationMessage]
    slots: list[_UntrustedSlot]


class _SafeBackends(NamedTuple):
    """The raw backend tokenizers (and their reserved-id set) used for content
    protection.

    Both clear truncation/padding/post_processor so their encodings differ only
    by added-token matching. `safe` additionally strips all added tokens and is
    used to encode neutralized content into subwords. `reference` keeps the added
    tokens and is used to decide whether a string needs neutralization: a string
    is dirty iff its `reference` encoding contains an added-token id, which
    happens exactly when it carries an added/special token -- never merely
    because the real tokenizer would truncate/pad it.

    `reserved_ids` is the set of added-token ids `safe` strips (equivalently, the
    ids on which `reference` and `safe` diverge). A single `reference` encode plus
    a membership test against it is equivalent to comparing both encodings, so the
    common (benign) path tokenizes once instead of twice; the `safe` encode is
    computed only for strings that actually need neutralization. This equivalence
    holds because `_verify_safe_backend_neutralizes` fails closed at setup for any
    reserved token that `safe` could still reproduce as its own single id.
    """

    safe: Any
    reference: Any
    reserved_ids: frozenset[int]


def _ensure_untrusted_placeholder_pool(tokenizer: HfTokenizer, count: int) -> list[int]:
    """Register at least `count` distinct placeholder tokens and return their
    IDs (each enforced to be a single token). Fails closed past the pool cap.

    Invariant: index `i` in the returned list is the ID of the token string
    `_UNTRUSTED_CONTENT_PLACEHOLDER_TEMPLATE.format(i)`. `_swap_untrusted_content`
    relies on this to pair the placeholder string it renders (`format(i)`) with
    the ID it stores on the slot (`placeholder_ids[i]`)."""
    if count > _UNTRUSTED_PLACEHOLDER_POOL_CAP:
        raise ValueError(
            _CONTENT_PROTECTION_POOL_EXHAUSTED_ERROR.format(
                count=count, cap=_UNTRUSTED_PLACEHOLDER_POOL_CAP
            )
        )

    cache = _UNTRUSTED_PLACEHOLDER_POOL_CACHE.get(tokenizer)
    if cache is None:
        cache = []
        _UNTRUSTED_PLACEHOLDER_POOL_CACHE[tokenizer] = cache
    if len(cache) >= count:
        return cache[:count]

    new_tokens = [
        _UNTRUSTED_CONTENT_PLACEHOLDER_TEMPLATE.format(i)
        for i in range(len(cache), count)
    ]
    tokenizer.add_special_tokens({"additional_special_tokens": new_tokens})
    for token in new_tokens:
        cache.append(_verify_single_token_id(tokenizer, token))
    return cache[:count]


def _make_backend_from_json(
    tok_json: dict[str, Any], *, strip_added_tokens: bool
) -> Any:
    """Build a raw ``tokenizers.Tokenizer`` from parsed backend JSON.

    ``split_special_tokens=True`` only prevents matching of tokens registered
    as ``AddedToken(special=True)``. Non-special added tokens (e.g. Qwen's
    ``<tool_call>``) are still matched. When ``strip_added_tokens`` is set, *all*
    added tokens are removed so every added-vocab string in untrusted text is
    split into subwords (the "safe" backend). Otherwise the added tokens are kept
    (the "reference" backend used only for dirty-detection).

    Both variants clear truncation/padding/post_processor so that -- apart from
    added-token matching -- they encode identically to each other. This is what
    lets `_neutralization_safe_ids` flag a string as dirty *only* when it carries
    an added/special token, never merely because the source tokenizer.json
    enabled truncation/padding (which would otherwise make long benign content
    encode differently and be needlessly neutralized).

    Only top-level keys are reassigned, so a shallow copy isolates each variant
    from the shared parsed dict (the nested structures are read-only).

    The result is a raw ``tokenizers.Tokenizer``. We only ever call
    ``encode(..., add_special_tokens=False)`` on it and never mutate its state
    afterwards, so it is safe to share across the sync executor's worker threads:
    the Rust tokenizer keeps its state behind an ``RwLock`` and ``encode`` takes
    a read guard. (The ``maybe_make_thread_pool`` machinery exists for the
    *Python* fast-tokenizer wrapper, whose mutable state and non-atomic compound
    mutations we bypass.)
    """
    from tokenizers import Tokenizer as RustTokenizer

    variant = dict(tok_json)
    if strip_added_tokens:
        variant["added_tokens"] = []
    variant["post_processor"] = None
    variant["truncation"] = None
    variant["padding"] = None
    return RustTokenizer.from_str(json.dumps(variant))


def _ensure_untrusted_safe_backend(tokenizer: HfTokenizer) -> _SafeBackends:
    cached = _UNTRUSTED_SAFE_BACKEND_CACHE.get(tokenizer)
    if cached is not None:
        return cached
    # Parse the (multi-MB) backend JSON once; both variants and `reserved_ids`
    # derive from it.
    tok_json = json.loads(tokenizer.backend_tokenizer.to_str())
    safe = _make_backend_from_json(tok_json, strip_added_tokens=True)
    reference = _make_backend_from_json(tok_json, strip_added_tokens=False)
    _verify_safe_backend_neutralizes(tokenizer, safe)
    # The ids `safe` strips (`added_tokens` in the backend json): a `reference`
    # encode carrying one of these is exactly a string `safe` would encode
    # differently, i.e. one that needs neutralization. Assumed to be a subset of
    # the tokens `_verify_safe_backend_neutralizes` checks (`get_added_vocab()`
    # | `all_special_tokens`), which holds for how real models register control
    # tokens.
    reserved_ids = frozenset(entry["id"] for entry in tok_json["added_tokens"])
    backends = _SafeBackends(safe=safe, reference=reference, reserved_ids=reserved_ids)
    _UNTRUSTED_SAFE_BACKEND_CACHE[tokenizer] = backends
    return backends


def _verify_safe_backend_neutralizes(tokenizer: HfTokenizer, safe_backend: Any) -> None:
    """Fail closed if the safe backend does not neutralize every added/special
    token.

    Stripping the added-token layer normally forces each reserved token string to
    split into subwords. A token reachable as a single ID through the base vocab's
    merge path would still re-encode to its own reserved ID, re-opening the
    injection hole. This is not hypothetical for base-vocab specials (e.g.
    SentencePiece ``<s>``/``</s>`` that are real pieces rather than added tokens):
    they never appear in ``get_added_vocab()`` yet still re-encode to their own
    ID. Enumerate the union of the added vocab and ``all_special_tokens`` and flag
    any token whose safe-backend encoding still reproduces its real single ID;
    such a model cannot be protected, so refuse to enable the defense.

    Residual gap (accepted, not covered): a control token that is a single-ID
    base-vocab piece registered as *neither* an added token nor a special token is
    invisible here -- it encodes identically in both backends and is
    indistinguishable from an ordinary base-vocab piece by encoding behavior
    alone, so there is no systematic vocab-wide test for it (every benign
    single-ID piece "survives" the safe backend too). This defense relies on the
    added/special-token registration that real models use for control tokens.
    """
    candidates = set(tokenizer.get_added_vocab())
    candidates.update(t for t in tokenizer.all_special_tokens if t)

    leaking: list[str] = []
    for content in candidates:
        if UNTRUSTED_CONTENT_PLACEHOLDER_PREFIX in content:
            continue
        real_ids = tokenizer.encode(content, add_special_tokens=False)
        if len(real_ids) != 1:
            continue
        safe_ids = safe_backend.encode(content, add_special_tokens=False).ids
        if safe_ids == real_ids:
            leaking.append(content)
    if leaking:
        raise ValueError(
            _CONTENT_PROTECTION_SAFE_BACKEND_LEAK_ERROR.format(
                tokens=", ".join(repr(t) for t in sorted(leaking))
            )
        )


def _neutralization_safe_ids(text: str, backends: _SafeBackends) -> list[int] | None:
    """Return the safe-backend token ids for `text` iff it must be neutralized,
    else `None`.

    `text` needs neutralization exactly when it contains an added/special token
    (including `special:false` added tokens like Nemotron's `<think>`) that would
    otherwise become a real reserved id. The reference backend keeps the added
    tokens, so `text` is dirty iff its reference encoding contains an added-token
    id (`backends.reserved_ids`). This is equivalent to comparing the reference
    and safe encodings -- the two differ *only* on added-token matching (see
    `_make_backend_from_json`), and `_verify_safe_backend_neutralizes` guarantees no
    reserved id survives the safe backend -- but lets the common (benign) path
    tokenize once. Only a dirty string pays the second (safe) encode, whose ids
    the caller caches for splicing; clean text is left verbatim in the skeleton.
    """
    reference_ids = backends.reference.encode(text, add_special_tokens=False).ids
    if backends.reserved_ids.isdisjoint(reference_ids):
        return None
    return backends.safe.encode(text, add_special_tokens=False).ids


def _validate_content_protection_supported(
    model_config: ModelConfig, tokenizer: HfTokenizer
) -> None:
    """Fail closed when content protection is opted-in but unsupported."""
    from vllm.utils.mistral import is_mistral_tokenizer

    if getattr(model_config, "enable_prompt_embeds", False):
        raise ValueError(_CONTENT_PROTECTION_PROMPT_EMBEDS_ERROR)
    if is_mistral_tokenizer(tokenizer):
        raise ValueError(_CONTENT_PROTECTION_MISTRAL_ERROR)
    if not isinstance(tokenizer, HfTokenizer):
        raise ValueError(_CONTENT_PROTECTION_UNSUPPORTED_TOKENIZER_ERROR)
    if not getattr(tokenizer, "is_fast", False):
        raise ValueError(_CONTENT_PROTECTION_NOT_FAST_ERROR)


# Content-part types that carry one untrusted string, keyed by the field holding
# it. `tool_reference["name"]` is client-controlled (e.g. via the Anthropic
# tool_result adapter) and rendered raw by templates, so it needs the same
# neutralization as a `text` part's `text` field.
_UNTRUSTED_TEXT_PART_FIELDS: Final[dict[str, str]] = {
    "text": "text",
    "tool_reference": "name",
}


def _untrusted_text_part_field(part: Any) -> str | None:
    """Return the field name holding `part`'s untrusted string, or None if `part`
    is not a recognized untrusted-string-carrying content part."""
    if not isinstance(part, dict):
        return None
    part_type = part.get("type")
    if not isinstance(part_type, str):
        return None
    field = _UNTRUSTED_TEXT_PART_FIELDS.get(part_type)
    if field is None or not isinstance(part.get(field), str):
        return None
    return field


def _transform_arguments(value: Any, fn: _SlotFn) -> Any:
    """Recursively rebuild a `tool_calls` `arguments` value, replacing every
    string dict key and string leaf via `fn`. Non-string scalars pass through.

    `fn` is called for *every* untrusted string in a fixed order (dict entries
    in insertion order, key before value), so the swap pass assigns slot ordinals
    in a stable order. Copy-on-write: allocation is deferred until `fn` actually
    changes a string (by identity); a fully-unchanged value is returned as-is, so
    read-only passes allocate nothing.

    Deliberately not built on `vllm.utils.jsontree`'s `json_map_leaves` /
    `json_iter_leaves`: those walkers only transform dict *values* (never keys)
    and always rebuild every container. Both matter here -- attacker-controlled
    JSON object *keys* need protection too, and the reject/swap passes rely on
    `is not` identity to decide whether a message needs rewriting.
    """
    if isinstance(value, dict):
        new_dict: dict[Any, Any] | None = None
        for i, (k, v) in enumerate(value.items()):
            new_key = fn(k) if isinstance(k, str) else k
            new_val = _transform_arguments(v, fn)
            if new_dict is None:
                if new_key is k and new_val is v:
                    continue
                new_dict = dict(itertools.islice(value.items(), i))
            new_dict[new_key] = new_val
        return new_dict if new_dict is not None else value
    if isinstance(value, list):
        new_list: list[Any] | None = None
        for i, item in enumerate(value):
            new_item = _transform_arguments(item, fn)
            if new_list is None:
                if new_item is item:
                    continue
                new_list = list(value[:i])
            new_list.append(new_item)
        return new_list if new_list is not None else value
    if isinstance(value, str):
        return fn(value)
    return value


def _transform_tool_call(tool_call: Any, fn: _SlotFn) -> Any:
    """Rebuild one `tool_calls` entry, protecting `function.name` (a string
    emitted raw next to control tokens) and the string leaves/keys of
    `function.arguments` (a dict, list, or raw JSON string). Copy-on-write:
    returns the original when unchanged."""
    if not isinstance(tool_call, dict):
        return tool_call
    func = tool_call.get("function")
    if not isinstance(func, dict):
        return tool_call

    new_func: dict[Any, Any] | None = None
    name = func.get("name")
    if isinstance(name, str):
        new_name = fn(name)
        if new_name is not name:
            new_func = copy.copy(func)
            new_func["name"] = new_name

    args = func.get("arguments")
    if isinstance(args, (dict, list, str)):
        # `arguments` may be a raw JSON string (e.g. Kimi-style templates emit it
        # verbatim); treat the whole string as one untrusted slot.
        new_args = _transform_arguments(args, fn)
        if new_args is not args:
            if new_func is None:
                new_func = copy.copy(func)
            new_func["arguments"] = new_args

    if new_func is None:
        return tool_call
    new_tc = copy.copy(tool_call)
    new_tc["function"] = new_func
    return new_tc


def _transform_message(msg: ConversationMessage, fn: _SlotFn) -> ConversationMessage:
    """Walk a message's protected strings in a deterministic order, calling `fn`
    on each and returning a new message with the replacements. Copy-on-write:
    `fn` is always called for every protected string, but the message is copied
    only when `fn` actually changes one, so the read-only reject pass and
    unchanged slots allocate nothing.

    The order is deterministic so each neutralized string maps to a distinct
    placeholder in a predictable position, but it is not a correctness contract:
    splicing matches placeholders by token identity (see
    `_splice_untrusted_content`), so re-ordering the walk would still splice
    correctly. Order: content (str, or each
    `{"type":"text"}`/`{"type":"tool_reference"}` part), then the message
    `name`; for `assistant`, each `tool_calls` entry's `function.name` then its
    `arguments`. Coverage is role-agnostic -- see the module banner above.
    """
    updates: dict[str, Any] = {}

    content = msg.get("content")
    if isinstance(content, str):
        new_content = fn(content)
        if new_content is not content:
            updates["content"] = new_content
    elif isinstance(content, list):
        new_parts: list[Any] | None = None
        for j, part in enumerate(content):
            field = _untrusted_text_part_field(part)
            if field is None:
                continue
            old_text = part[field]
            new_text = fn(old_text)
            if new_text is not old_text:
                if new_parts is None:
                    new_parts = list(content)
                new_part = copy.copy(part)
                new_part[field] = new_text
                new_parts[j] = new_part
        if new_parts is not None:
            updates["content"] = new_parts
    name = msg.get("name")
    if isinstance(name, str):
        new_name = fn(name)
        if new_name is not name:
            updates["name"] = new_name

    if msg.get("role") == "assistant":
        tool_calls = msg.get("tool_calls") or []
        new_tool_calls = [_transform_tool_call(tc, fn) for tc in tool_calls]
        if any(new is not old for new, old in zip(new_tool_calls, tool_calls)):
            updates["tool_calls"] = new_tool_calls

    if not updates:
        return msg
    new_msg = copy.copy(msg)
    new_msg.update(updates)  # type: ignore[typeddict-item]
    return new_msg


def _validate_tool_call_ids(
    conversation: list[ConversationMessage],
    backends: _SafeBackends,
) -> None:
    """Reject (fail closed) when an attacker-controllable tool-call id carries a
    control token that would encode to a reserved special-token ID.

    `tool_call_id` (tool messages) and `tool_calls[].id` (assistant messages) are
    client-supplied on resubmitted history and rendered raw by some templates, so
    a literal control-token string there would forge a real special token. Unlike
    free-text content, ids never legitimately contain control tokens, so we reject
    rather than splice -- keeping legitimate ids intact through every template
    (including id-transforming ones like Mistral `id[-9:]`).
    """

    def _check(text: str) -> None:
        if _neutralization_safe_ids(text, backends) is not None:
            raise ValueError(_CONTENT_PROTECTION_TOOL_CALL_ID_ERROR)

    for msg in conversation:
        role = msg.get("role")
        if role == "tool":
            tool_call_id = msg.get("tool_call_id")
            if isinstance(tool_call_id, str):
                _check(tool_call_id)
        elif role == "assistant":
            for tool_call in msg.get("tool_calls") or []:
                if isinstance(tool_call, dict) and isinstance(tool_call.get("id"), str):
                    _check(tool_call["id"])


def _swap_untrusted_content(
    conversation: list[ConversationMessage],
    placeholder_ids: list[int],
    backends: _SafeBackends,
) -> tuple[list[ConversationMessage], list[_UntrustedSlot]]:
    """Return a skeleton copy of `conversation` with each untrusted string that
    needs neutralization replaced by its distinct placeholder token, plus the
    slots. The k-th neutralized string takes pool index k
    (`placeholder_ids[k]`); splicing later matches by token identity, so this
    index assignment is for allocation, not a splice-ordering requirement.

    Selective: a string is swapped only when it actually carries an added/special
    token (`_neutralization_safe_ids`); clean strings are returned unchanged so
    the template renders them normally. Covers string / list ("openai") content
    (`text` and `tool_reference` parts), message `name`, and assistant
    `tool_calls` (`function.name` + argument string leaves and keys). Media /
    other non-string parts and non-string scalars pass through untouched. Fails
    closed when the pool is exhausted.
    """
    slots: list[_UntrustedSlot] = []

    def _swap(text: str) -> str:
        # Reject the reserved placeholder prefix in any untrusted string so a
        # caller cannot plant a spurious splice point (folded into the swap walk
        # so the conversation is traversed once, not twice).
        if UNTRUSTED_CONTENT_PLACEHOLDER_PREFIX in text:
            raise ValueError(_CONTENT_PROTECTION_RESERVED_IN_TEXT_ERROR)
        safe_ids = _neutralization_safe_ids(text, backends)
        if safe_ids is None:
            return text
        i = len(slots)
        if i >= len(placeholder_ids):
            raise ValueError(
                _CONTENT_PROTECTION_POOL_EXHAUSTED_ERROR.format(
                    count=i + 1, cap=len(placeholder_ids)
                )
            )
        slots.append(_UntrustedSlot(placeholder_ids[i], safe_ids))
        # Slot i uses pool index i (see `_ensure_untrusted_placeholder_pool`);
        # return its placeholder string as the skeleton stand-in.
        return _UNTRUSTED_CONTENT_PLACEHOLDER_TEMPLATE.format(i)

    skeleton = [_transform_message(msg, _swap) for msg in conversation]
    return skeleton, slots


def _splice_untrusted_content(
    skeleton_ids: list[int],
    slots: list[_UntrustedSlot],
) -> list[int]:
    """Replace each slot's placeholder token in `skeleton_ids` with the slot's
    cached safe-backend encoding (`safe_ids`, no special tokens).

    Placeholders are matched by token identity, so slots are spliced correctly
    even when the template re-orders content. Degrades gracefully rather than
    failing closed when a template drops or duplicates content:

    - 0 occurrences: the template omitted this slot (or a transform mangled the
      placeholder so it no longer appears as its own token). Splice nothing --
      the content is simply absent; no injection is possible.
    - >1 occurrences: the template rendered this slot in multiple places. Splice
      the same content into every occurrence.
    """
    # Each slot has a distinct placeholder id, so a single dict suffices. The
    # per-placeholder counts are only for the diagnostic warning below, so we
    # tally them in the same pass that builds the result.
    encoded = {slot.placeholder_id: slot.safe_ids for slot in slots}
    counts: dict[int, int] = defaultdict(int)
    result: list[int] = []
    for token_id in skeleton_ids:
        replacement = encoded.get(token_id)
        if replacement is None:
            result.append(token_id)
        else:
            counts[token_id] += 1
            result.extend(replacement)

    for slot in slots:
        count = counts[slot.placeholder_id]
        if count != 1:
            logger.warning_once(_CONTENT_PROTECTION_SPAN_COUNT_MSG, count)
    return result


_PROCESSOR_CHAT_TEMPLATES = dict[tuple[str, bool], str | None]()
"""
Used in `_try_get_processor_chat_template` to avoid calling
`cached_get_processor` again if the processor fails to be loaded.

This is needed because `lru_cache` does not cache when an exception happens.
"""


def _try_get_processor_chat_template(
    tokenizer: HfTokenizer,
    *,
    trust_remote_code: bool,
) -> str | None:
    cache_key = (tokenizer.name_or_path, trust_remote_code)
    if cache_key in _PROCESSOR_CHAT_TEMPLATES:
        return _PROCESSOR_CHAT_TEMPLATES[cache_key]

    from transformers import ProcessorMixin, PythonBackend, TokenizersBackend

    try:
        processor = cached_get_processor(
            tokenizer.name_or_path,
            processor_cls=(PythonBackend, TokenizersBackend, ProcessorMixin),
            trust_remote_code=trust_remote_code,
        )
        if (
            isinstance(processor, ProcessorMixin)
            and hasattr(processor, "chat_template")
            and (chat_template := processor.chat_template) is not None
        ):
            _PROCESSOR_CHAT_TEMPLATES[cache_key] = chat_template
            return chat_template
    except Exception:
        logger.debug(
            "Failed to load AutoProcessor chat template for %s",
            tokenizer.name_or_path,
            exc_info=True,
        )

    _PROCESSOR_CHAT_TEMPLATES[cache_key] = None
    return None


def resolve_chat_template(
    tokenizer: HfTokenizer,
    chat_template: str | None,
    tools: list[dict[str, Any]] | None,
    *,
    model_config: ModelConfig,
) -> str | None:
    # 1st priority: The given chat template
    if chat_template is not None:
        # Resolve template names (e.g. "tool_use") to actual Jinja content
        # so that downstream kwargs detection can parse template variables.
        return tokenizer.get_chat_template(chat_template, tools=tools)

    # 2nd priority: AutoProcessor chat template, unless tool calling is enabled
    if tools is None:
        chat_template = _try_get_processor_chat_template(
            tokenizer,
            trust_remote_code=model_config.trust_remote_code,
        )
        if chat_template is not None:
            return chat_template

    # 3rd priority: AutoTokenizer chat template
    try:
        return tokenizer.get_chat_template(chat_template, tools=tools)
    except Exception:
        logger.debug(
            "Failed to load AutoTokenizer chat template for %s",
            tokenizer.name_or_path,
            exc_info=True,
        )

    # 4th priority: Predefined fallbacks
    path = get_chat_template_fallback_path(
        model_type=model_config.hf_config.model_type,
        tokenizer_name_or_path=tokenizer.name_or_path,
    )
    if path is not None:
        logger.info_once(
            "Loading chat template fallback for %s as there isn't one "
            "defined on HF Hub.",
            tokenizer.name_or_path,
        )
        chat_template = load_chat_template(path)
    else:
        logger.debug_once(
            "There is no chat template fallback for %s", tokenizer.name_or_path
        )

    return chat_template


def _is_var_access(node: jinja2.nodes.Node, varname: str) -> bool:
    if isinstance(node, jinja2.nodes.Name):
        return node.ctx == "load" and node.name == varname

    return False


def _is_attr_access(node: jinja2.nodes.Node, varname: str, key: str) -> bool:
    if isinstance(node, jinja2.nodes.Getitem):
        return (
            _is_var_access(node.node, varname)
            and isinstance(node.arg, jinja2.nodes.Const)
            and node.arg.value == key
        )

    if isinstance(node, jinja2.nodes.Getattr):
        return _is_var_access(node.node, varname) and node.attr == key

    return False


def _is_var_or_elems_access(
    node: jinja2.nodes.Node,
    varname: str,
    key: str | None = None,
) -> bool:
    if isinstance(node, jinja2.nodes.Filter):
        return node.node is not None and _is_var_or_elems_access(
            node.node, varname, key
        )
    if isinstance(node, jinja2.nodes.Test):
        return _is_var_or_elems_access(node.node, varname, key)

    if isinstance(node, jinja2.nodes.Getitem) and isinstance(
        node.arg, jinja2.nodes.Slice
    ):
        return _is_var_or_elems_access(node.node, varname, key)

    return _is_attr_access(node, varname, key) if key else _is_var_access(node, varname)


def _iter_nodes_assign_var_or_elems(root: jinja2.nodes.Node, varname: str):
    # Global variable that is implicitly defined at the root
    yield root, varname

    # Iterative BFS
    related_varnames = deque([varname])
    while related_varnames:
        related_varname = related_varnames.popleft()

        for assign_ast in root.find_all(jinja2.nodes.Assign):
            lhs = assign_ast.target
            rhs = assign_ast.node

            if _is_var_or_elems_access(rhs, related_varname):
                assert isinstance(lhs, jinja2.nodes.Name)
                yield assign_ast, lhs.name

                # Avoid infinite looping for self-assignment
                if lhs.name != related_varname:
                    related_varnames.append(lhs.name)


# NOTE: The proper way to handle this is to build a CFG so that we can handle
# the scope in which each variable is defined, but that is too complicated
def _iter_nodes_assign_messages_item(root: jinja2.nodes.Node):
    messages_varnames = [
        varname for _, varname in _iter_nodes_assign_var_or_elems(root, "messages")
    ]

    # Search for {%- for message in messages -%} loops
    for loop_ast in root.find_all(jinja2.nodes.For):
        loop_iter = loop_ast.iter
        loop_target = loop_ast.target

        for varname in messages_varnames:
            if _is_var_or_elems_access(loop_iter, varname):
                assert isinstance(loop_target, jinja2.nodes.Name)
                yield loop_ast, loop_target.name
                break


def _iter_nodes_assign_content_item(root: jinja2.nodes.Node):
    message_varnames = [
        varname for _, varname in _iter_nodes_assign_messages_item(root)
    ]

    # Search for {%- for content in message['content'] -%} loops
    # or {%- for item in content -%} loops
    for loop_ast in root.find_all(jinja2.nodes.For):
        loop_iter = loop_ast.iter
        loop_target = loop_ast.target

        for varname in message_varnames:
            if _is_var_or_elems_access(loop_iter, varname, "content"):
                assert isinstance(loop_target, jinja2.nodes.Name)
                yield loop_ast, loop_target.name
                break

        if isinstance(loop_iter, jinja2.nodes.Name) and loop_iter.name == "content":
            assert isinstance(loop_target, jinja2.nodes.Name)
            yield loop_ast, loop_target.name


def _try_extract_ast(chat_template: str) -> jinja2.nodes.Template | None:
    import transformers.utils.chat_template_utils as hf_chat_utils

    try:
        jinja_compiled = hf_chat_utils._compile_jinja_template(chat_template)
        return jinja_compiled.environment.parse(chat_template)
    except Exception:
        logger.exception("Error when compiling Jinja template")
        return None


@lru_cache(maxsize=32)
def _detect_content_format(
    chat_template: str,
    *,
    default: ChatTemplateContentFormat,
) -> ChatTemplateContentFormat:
    jinja_ast = _try_extract_ast(chat_template)
    if jinja_ast is None:
        return default

    try:
        next(_iter_nodes_assign_content_item(jinja_ast))
    except StopIteration:
        return "string"
    except Exception:
        logger.exception("Error when parsing AST of Jinja template")
        return default
    else:
        return "openai"


@lru_cache(maxsize=32)
def _detect_developer_role_support(chat_template: str) -> bool:
    return '"developer"' in chat_template or "'developer'" in chat_template


def _convert_developer_to_system(
    conversation: list[ConversationMessage],
) -> list[ConversationMessage]:
    converted: list[ConversationMessage] = []
    for msg in conversation:
        if msg["role"] == "developer":
            new_msg = dict(msg)
            new_msg["role"] = "system"
            new_msg.pop("tools", None)
            converted.append(new_msg)  # type: ignore[arg-type]
        else:
            converted.append(msg)
    return converted


def _consolidate_system_messages(
    conversation: list[ConversationMessage],
) -> list[ConversationMessage]:
    """Merge all system messages into one at position 0.

    Some chat templates (e.g. Qwen 3.6) require the system message to be the
    very first message.  After developer-to-system conversion, system messages
    may appear at non-first positions; this merges them into a single message.
    """
    system_contents: list[str] = []
    non_system: list[ConversationMessage] = []
    needs_consolidation = False
    for i, msg in enumerate(conversation):
        if msg["role"] == "system":
            if i > 0 or system_contents:
                needs_consolidation = True
            content = msg.get("content", "")
            if isinstance(content, list):
                parts = []
                for part in content:
                    if isinstance(part, dict) and "text" in part:
                        parts.append(part["text"])
                    elif isinstance(part, str):
                        parts.append(part)
                content = "\n".join(parts)
            if content:
                system_contents.append(content)
        else:
            non_system.append(msg)

    if not needs_consolidation:
        return conversation

    merged: ConversationMessage = {
        "role": "system",
        "content": "\n\n".join(system_contents),
    }
    return [merged, *non_system]


def _resolve_chat_template_content_format(
    chat_template: str | None,
    tools: list[dict[str, Any]] | None,
    tokenizer: HfTokenizer,
    *,
    model_config: ModelConfig,
) -> ChatTemplateContentFormat:
    resolved_chat_template = resolve_chat_template(
        tokenizer,
        chat_template=chat_template,
        tools=tools,
        model_config=model_config,
    )

    jinja_text = (
        resolved_chat_template
        if isinstance(resolved_chat_template, str)
        else load_chat_template(chat_template, is_literal=True)
    )

    detected_format = (
        "string"
        if jinja_text is None
        else _detect_content_format(jinja_text, default="string")
    )

    return detected_format


@lru_cache
def _log_chat_template_content_format(
    chat_template: str | None,  # For caching purposes
    given_format: ChatTemplateContentFormatOption,
    detected_format: ChatTemplateContentFormatOption,
):
    logger.info(
        "Detected the chat template content format to be '%s'. "
        "You can set `--chat-template-content-format` to override this.",
        detected_format,
    )

    if given_format != "auto" and given_format != detected_format:
        logger.warning(
            "You specified `--chat-template-content-format %s` "
            "which is different from the detected format '%s'. "
            "If our automatic detection is incorrect, please consider "
            "opening a GitHub issue so that we can improve it: "
            "https://github.com/vllm-project/vllm/issues/new/choose",
            given_format,
            detected_format,
        )


def resolve_chat_template_content_format(
    chat_template: str | None,
    tools: list[dict[str, Any]] | None,
    given_format: ChatTemplateContentFormatOption,
    tokenizer: HfTokenizer,
    *,
    model_config: ModelConfig,
) -> ChatTemplateContentFormat:
    if given_format != "auto":
        return given_format

    detected_format = _resolve_chat_template_content_format(
        chat_template,
        tools,
        tokenizer,
        model_config=model_config,
    )

    _log_chat_template_content_format(
        chat_template,
        given_format=given_format,
        detected_format=detected_format,
    )

    return detected_format


# adapted from https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/utils/chat_template_utils.py#L398-L412
# only preserve the parse function used to resolve chat template kwargs
class AssistantTracker(jinja2.ext.Extension):
    tags = {"generation"}

    def parse(self, parser: jinja2.parser.Parser) -> jinja2.nodes.Node:
        lineno = next(parser.stream).lineno
        body = parser.parse_statements(("name:endgeneration",), drop_needle=True)
        call = self.call_method("_generation_support")
        call_block = jinja2.nodes.CallBlock(call, [], [], body)
        return call_block.set_lineno(lineno)


def _resolve_chat_template_kwargs(chat_template: str) -> Set[str]:
    env = jinja2.sandbox.ImmutableSandboxedEnvironment(
        trim_blocks=True,
        lstrip_blocks=True,
        extensions=[AssistantTracker, jinja2.ext.loopcontrols],
    )
    parsed_content = env.parse(chat_template)
    template_vars = jinja2.meta.find_undeclared_variables(parsed_content)
    return template_vars


_cached_resolve_chat_template_kwargs = lru_cache(_resolve_chat_template_kwargs)


@lru_cache
def _get_hf_base_chat_template_params() -> frozenset[str]:
    from transformers import PythonBackend

    # Get standard parameters from HuggingFace's base tokenizer class.
    # This dynamically extracts parameters from PythonBackend's
    # apply_chat_template method, ensuring compatibility with tokenizers
    # that use **kwargs to receive standard parameters.

    # Read signature from HF's base class - the single source of truth
    base_sig = inspect.signature(PythonBackend.apply_chat_template)

    # Exclude VAR_KEYWORD (**kwargs) and VAR_POSITIONAL (*args) placeholders
    return frozenset(
        p.name
        for p in base_sig.parameters.values()
        if p.kind
        not in (inspect.Parameter.VAR_KEYWORD, inspect.Parameter.VAR_POSITIONAL)
    )


def resolve_chat_template_kwargs(
    tokenizer: HfTokenizer,
    chat_template: str,
    chat_template_kwargs: dict[str, Any],
    raise_on_unexpected: bool = True,
) -> dict[str, Any]:
    # We exclude chat_template from kwargs here, because
    # chat template has been already resolved at this stage
    unexpected_vars = {"chat_template", "tokenize"}
    if raise_on_unexpected and (
        unexpected_in_kwargs := unexpected_vars & chat_template_kwargs.keys()
    ):
        raise ValueError(
            "Found unexpected chat template kwargs from request: "
            f"{unexpected_in_kwargs}"
        )

    fn_kw = {
        k
        for k in chat_template_kwargs
        if supports_kw(tokenizer.apply_chat_template, k, allow_var_kwargs=False)
    }
    template_vars = _cached_resolve_chat_template_kwargs(chat_template)

    # Allow standard HF parameters even if tokenizer uses **kwargs to receive them
    hf_base_params = _get_hf_base_chat_template_params()

    accept_vars = (fn_kw | template_vars | hf_base_params) - unexpected_vars
    return {k: v for k, v in chat_template_kwargs.items() if k in accept_vars}


@overload
def safe_apply_chat_template(
    model_config: ModelConfig,
    tokenizer: HfTokenizer,
    conversation: list[ConversationMessage],
    *,
    tools: list[dict[str, Any]] | None = ...,
    chat_template: str | None = ...,
    tokenize: Literal[True] = ...,
    return_assistant_tokens_mask: Literal[False] = ...,
    **kwargs,
) -> list[int]: ...
@overload
def safe_apply_chat_template(
    model_config: ModelConfig,
    tokenizer: HfTokenizer,
    conversation: list[ConversationMessage],
    *,
    tools: list[dict[str, Any]] | None = ...,
    chat_template: str | None = ...,
    tokenize: Literal[False] = ...,
    return_assistant_tokens_mask: Literal[False] = ...,
    **kwargs,
) -> str: ...
@overload
def safe_apply_chat_template(
    model_config: ModelConfig,
    tokenizer: HfTokenizer,
    conversation: list[ConversationMessage],
    *,
    tools: list[dict[str, Any]] | None = ...,
    chat_template: str | None = ...,
    return_assistant_tokens_mask: Literal[True],
    **kwargs,
) -> tuple[list[int], list[int] | None]: ...
def safe_apply_chat_template(
    model_config: ModelConfig,
    tokenizer: HfTokenizer,
    conversation: list[ConversationMessage],
    *,
    tools: list[dict[str, Any]] | None = None,
    chat_template: str | None = None,
    tokenize: bool = True,
    return_assistant_tokens_mask: bool = False,
    **kwargs,
) -> str | list[int] | tuple[list[int], list[int] | None]:
    chat_template = resolve_chat_template(
        tokenizer,
        chat_template=chat_template,
        tools=tools,
        model_config=model_config,
    )
    if chat_template is None:
        raise ChatTemplateResolutionError(
            "As of transformers v4.44, default chat template is no longer "
            "allowed, so you must provide a chat template if the tokenizer "
            "does not define one."
        )
    if any(
        msg["role"] == "developer" for msg in conversation
    ) and not _detect_developer_role_support(chat_template):
        conversation = _convert_developer_to_system(conversation)
        conversation = _consolidate_system_messages(conversation)
        logger.info_once(
            "Chat template does not support the 'developer' message role. "
            "Converting developer messages to 'system' role.",
        )
    resolved_kwargs = resolve_chat_template_kwargs(
        tokenizer=tokenizer,
        chat_template=chat_template,
        chat_template_kwargs=kwargs,
    )

    # assistant_tokens_mask requires tokenized output — force tokenize=True.
    if return_assistant_tokens_mask:
        tokenize = True

    # When return_assistant_tokens_mask is requested and the template supports it,
    # request assistant_tokens_mask via return_dict.
    # Check for the actual Jinja tag, not just the word "generation"
    # (which also appears in add_generation_prompt).
    if return_assistant_tokens_mask and "{% generation %}" in chat_template:
        resolved_kwargs["return_assistant_tokens_mask"] = True
        resolved_kwargs["return_dict"] = True
        resolved_kwargs.pop("tokenize", None)
        try:
            result = tokenizer.apply_chat_template(
                conversation=conversation,  # type: ignore[arg-type]
                tools=tools,  # type: ignore[arg-type]
                chat_template=chat_template,
                tokenize=True,
                **resolved_kwargs,
            )
        except (TypeError, ValueError) as exc:
            logger.warning(
                "apply_chat_template failed for assistant_tokens_mask: %s", exc
            )
        else:
            if isinstance(result, Mapping):
                token_ids = list(result.get("input_ids", []))
                mask_raw = result.get("assistant_masks")
                mask = list(mask_raw) if mask_raw is not None else None
                return token_ids, mask
            return list(result), None

    # transformers v5 changed the default of `return_dict` to True, which
    # makes `apply_chat_template(tokenize=True)` return a `BatchEncoding`
    # instead of `list[int]`. Force `return_dict=False` so downstream code
    # that expects a flat token list (e.g. `parse_dec_only_prompt`) works
    # consistently across v4 and v5.
    if tokenize and "return_dict" not in resolved_kwargs:
        resolved_kwargs["return_dict"] = False

    try:
        plain = tokenizer.apply_chat_template(
            conversation=conversation,  # type: ignore[arg-type]
            tools=tools,  # type: ignore[arg-type]
            chat_template=chat_template,
            tokenize=tokenize,
            **resolved_kwargs,
        )
    except Exception as e:
        logger.exception(
            "An error occurred in `transformers` while applying chat template"
        )
        raise ValueError(str(e)) from e

    if return_assistant_tokens_mask:
        assert isinstance(plain, list), f"Expected list[int], got {type(plain)}"
        return plain, None
    return plain


def rebuild_mm_uuids_from_mm_data(
    mm_uuids: MultiModalUUIDDict,
    mm_data: MultiModalDataDict,
) -> MultiModalUUIDDict:
    """Rebuild mm_uuids after vision_chunk processing.

    When videos are split into chunks, the original UUIDs need to be updated
    to reflect the new UUIDs generated for each chunk.

    Args:
        mm_uuids: Original UUIDs dictionary
        mm_data: Processed multimodal data with vision_chunk items

    Returns:
        Updated UUIDs dictionary with chunk UUIDs
    """
    vision_chunks = mm_data.get("vision_chunk")
    if vision_chunks is None:
        return mm_uuids

    assert all(isinstance(item, dict) for item in vision_chunks), (
        "Expected all vision_chunk items to be dicts"
    )
    vision_chunks = cast(list[dict[str, Any]], vision_chunks)
    vision_chunk_uuids = [
        uuid_val for item in vision_chunks if (uuid_val := item.get("uuid")) is not None
    ]

    if vision_chunk_uuids:
        mm_uuids = dict(mm_uuids)
        mm_uuids["vision_chunk"] = vision_chunk_uuids

    return mm_uuids


def build_video_prompts_from_mm_data(
    mm_data: MultiModalDataDict,
) -> list[str]:
    """Build video prompts from vision_chunk data.

    Collects prompts from video chunks and groups them by video_idx.

    Args:
        mm_data: Processed multimodal data with vision_chunk items

    Returns:
        List of video prompts, one per video.
    """
    vision_chunks = mm_data.get("vision_chunk")
    if vision_chunks is None:
        return []

    # Group chunks by video_idx
    video_prompts_dict: dict[int, list[str]] = defaultdict(list)

    for item in vision_chunks:
        # vision_chunk items are always dicts (VisionChunkImage/VisionChunkVideo)
        assert isinstance(item, dict)
        if item.get("type") == "video_chunk":
            video_idx = item.get("video_idx", 0)
            prompt = item.get("prompt", "")
            video_prompts_dict[video_idx].append(prompt)

    # Build prompts in video order
    video_prompts = [
        "".join(video_prompts_dict[video_idx])
        for video_idx in sorted(video_prompts_dict.keys())
    ]

    return video_prompts


def replace_vision_chunk_video_placeholder(
    prompt_raw: str | list[int],
    mm_data: MultiModalDataDict,
    video_placeholder: str | None,
) -> str | list[int]:
    # get video placeholder, replace it with runtime video-chunk prompts
    if video_placeholder and isinstance(prompt_raw, str):
        video_prompts = build_video_prompts_from_mm_data(mm_data)

        # replace in order
        prompt_raw_parts = prompt_raw.split(video_placeholder)
        if len(prompt_raw_parts) == len(video_prompts) + 1:
            prompt_raw = "".join(
                itertools.chain.from_iterable(zip(prompt_raw_parts, video_prompts))
            )
            prompt_raw += prompt_raw_parts[-1]
        else:
            logger.warning(
                "Number of video placeholders (%d) does not match "
                "number of videos (%d) in the request.",
                len(prompt_raw_parts) - 1,
                len(video_prompts),
            )
    return prompt_raw


class HfRenderer(BaseRenderer[HfTokenizer]):
    _SUPPORTS_CONTENT_PROTECTION: ClassVar[bool] = True

    def __init__(
        self,
        config: VllmConfig,
        tokenizer: HfTokenizer | None,
    ) -> None:
        # maybe_make_thread_pool swaps `self.tokenizer.__class__` in place, so we
        # must never hand it the process-shared singleton -- a shallow copy is
        # enough to isolate that class swap. When we additionally register tokens
        # (prompt_embeds placeholder / content-protection pool), those
        # `add_special_tokens` calls mutate the Rust backend, which a shallow copy
        # shares; deep-copy in that case so the mutation (and the inflated vocab)
        # stays local to this renderer. Deep-copy only when a registration path
        # actually runs -- the common case pays only the cheap shallow copy.
        content_protection = bool(envs.VLLM_CHAT_CONTENT_PROTECTION)
        enable_prompt_embeds = getattr(
            config.model_config, "enable_prompt_embeds", False
        )
        if tokenizer is not None and (content_protection or enable_prompt_embeds):
            tokenizer = copy.deepcopy(tokenizer)
        else:
            tokenizer = copy.copy(tokenizer)

        # Skip for mock configs and tokenizers.
        if enable_prompt_embeds and isinstance(tokenizer, HfTokenizer):
            _ensure_prompt_embeds_placeholder_token(tokenizer)

        self._content_protection = content_protection
        self._safe_backends: _SafeBackends | None = None
        self._placeholder_pool: list[int] = []
        if self._content_protection and tokenizer is not None:
            _validate_content_protection_supported(config.model_config, tokenizer)
            # Register the whole placeholder pool and build the safe backends once,
            # here: per-request rendering then never mutates the (thread-shared)
            # tokenizer nor re-derives these, it just reads the cached instance
            # attributes.
            self._placeholder_pool = _ensure_untrusted_placeholder_pool(
                tokenizer, _UNTRUSTED_PLACEHOLDER_POOL_CAP
            )
            self._safe_backends = _ensure_untrusted_safe_backend(tokenizer)

        super().__init__(config, tokenizer)

        self.use_unified_vision_chunk = getattr(
            config.model_config.hf_config, "use_unified_vision_chunk", False
        )

        self._apply_chat_template_async = make_async(
            safe_apply_chat_template, executor=self._executor
        )
        # Content-protection prep re-tokenizes untrusted content; offload it to
        # the executor so it never blocks the event loop in the async path.
        self._prepare_content_protection_async = make_async(
            self._prepare_content_protection, executor=self._executor
        )
        # Vision-chunk substitution re-tokenizes the rendered prompt; offload it
        # too so the extra encode never blocks the event loop.
        self._render_with_video_chunk_substitution_async = make_async(
            self._render_with_video_chunk_substitution, executor=self._executor
        )

        if self.tokenizer is not None:
            maybe_make_thread_pool(
                self.tokenizer, config.model_config.renderer_num_workers + 1
            )

    def _can_produce_offsets(self) -> bool:
        # HF tokenizers may be slow (use_fast=False); only fast tokenizers
        # expose offset_mapping.
        return self.tokenizer is not None and self.tokenizer.is_fast

    def _prepare_content_protection(
        self,
        conversation: list[ConversationMessage],
        params: ChatParams,
        prompt_embeds_tensors: list[torch.Tensor] | None,
        content_format: ChatTemplateContentFormat,
        has_multimodal: bool,
    ) -> _ContentProtection | None:
        """Return the content-protection context when token-space content
        protection applies to this request, else None. Fails closed on
        per-request incompatibilities rather than silently disabling."""
        if not self._content_protection or self.tokenizer is None:
            return None
        if prompt_embeds_tensors:
            raise ValueError(_CONTENT_PROTECTION_PROMPT_EMBEDS_ERROR)
        # With the "string" content format, media placeholders are flattened into
        # message content as model special tokens; neutralization would strip
        # them and break MM processing. The "openai" format keeps media as
        # structured parts, so those tokens come from the (trusted) template.
        if has_multimodal and content_format == "string":
            raise ValueError(_CONTENT_PROTECTION_MULTIMODAL_STRING_ERROR)

        # Pool + backends are registered/derived once at init (see `__init__`);
        # the guard above guarantees they are set here.
        assert self._safe_backends is not None
        backends = self._safe_backends
        # Reject (not splice) control tokens in tool-call ids; see the function.
        _validate_tool_call_ids(conversation, backends)
        skeleton, slots = _swap_untrusted_content(
            conversation, self._placeholder_pool, backends
        )
        if not slots:
            # Nothing was neutralized: benign request, no splicing. Leave the
            # request untouched so features like the assistant-token mask still
            # work -- only requests we actually rewrite are incompatible with it.
            return None
        # Splicing token ids invalidates the assistant-token mask alignment, so we
        # cannot honor the mask once any content is neutralized.
        if params.return_assistant_tokens_mask:
            raise ValueError(_CONTENT_PROTECTION_ASSISTANT_MASK_ERROR)
        return _ContentProtection(skeleton, slots)

    def _apply_content_protection(
        self,
        protection: _ContentProtection,
        skeleton_ids: list[int],
    ) -> list[int]:
        return _splice_untrusted_content(skeleton_ids, protection.slots)

    def _render_with_video_chunk_substitution(
        self,
        tokenizer: HfTokenizer,
        render_conversation: list[ConversationMessage],
        chat_template_kwargs: dict[str, Any],
        mm_data: MultiModalDataDict,
        video_placeholder: str,
    ) -> str | list[int]:
        """Render, expand unified vision-chunk video placeholders, then tokenize.

        Video-chunk substitution needs the rendered *string*, but content
        protection / prompt_embeds may have forced `tokenize=True`. Render at
        `tokenize=False` regardless, substitute on the string, then re-tokenize
        to token IDs iff the caller wanted them (so the content-protection splice
        and prompt_embeds post-processing still operate on IDs). When the caller
        wanted a string, return the substituted string (matching the string-output
        shape of the non-Kimi path, without re-tokenizing).
        """
        wants_tokens = chat_template_kwargs.get("tokenize", True)
        string_kwargs = {**chat_template_kwargs, "tokenize": False}
        prompt_str = cast(
            str,
            safe_apply_chat_template(
                self.model_config,
                tokenizer,
                render_conversation,
                **string_kwargs,
            ),
        )
        prompt_str = cast(
            str,
            replace_vision_chunk_video_placeholder(
                prompt_str, mm_data, video_placeholder
            ),
        )
        if wants_tokens:
            return tokenizer.encode(prompt_str, add_special_tokens=False)
        return prompt_str

    def _resolve_video_chunk_placeholder(
        self,
        mm_uuids: MultiModalUUIDDict | None,
        mm_data: MultiModalDataDict | None,
        params: ChatParams,
    ) -> tuple[bool, str | None]:
        """Resolve unified vision-chunk (Kimi-K2.5) rendering state.

        Returns `(kimi_vision_chunk, video_placeholder)`: whether the request
        needs unified vision-chunk handling (a video mm_uuids rebuild), and the
        video-placeholder string to expand (None when there is nothing to
        expand). Fails closed on the mask combination that cannot produce a
        rendered string to expand into.
        """
        kimi_vision_chunk = (
            self.use_unified_vision_chunk
            and mm_uuids is not None
            and mm_data is not None
        )
        video_placeholder = (
            getattr(self.model_config.hf_config, "video_placeholder", None)
            if kimi_vision_chunk
            else None
        )
        if video_placeholder is not None and params.return_assistant_tokens_mask:
            raise ValueError(_VIDEO_CHUNK_ASSISTANT_MASK_ERROR)
        return kimi_vision_chunk, video_placeholder

    def _finalize_rendered_prompt(
        self,
        prompt_raw: str | list[int],
        *,
        protection: _ContentProtection | None,
        kimi_vision_chunk: bool,
        mm_data: MultiModalDataDict | None,
        mm_uuids: MultiModalUUIDDict | None,
        assistant_tokens_mask: list[int] | None,
        prompt_embeds_tensors: list[torch.Tensor] | None,
        prompt_embeds_placeholder_token_id: int | None,
        params: ChatParams,
    ) -> DictPrompt:
        """Assemble the final `DictPrompt` shared by the sync and async render
        paths: rebuild vision-chunk uuids, splice neutralized content, parse the
        raw prompt, then attach the assistant-mask / prompt_embeds / multimodal
        payloads. Does no tokenization or executor work (so neither render path
        needs to offload it); both call it after producing `prompt_raw`.
        """
        if kimi_vision_chunk and mm_uuids is not None and mm_data is not None:
            mm_uuids = rebuild_mm_uuids_from_mm_data(mm_uuids, mm_data)

        if protection is not None:
            prompt_raw = self._apply_content_protection(
                protection, cast(list[int], prompt_raw)
            )

        prompt = parse_dec_only_prompt(prompt_raw)

        if assistant_tokens_mask is not None:
            cast(dict, prompt)["_assistant_tokens_mask"] = assistant_tokens_mask

        # When `prompt_embeds` is mixed with other modality data,
        # `_process_tokens` runs `_process_multimodal` first (expanding
        # `<|AUDIO|>` / `<|IMAGE|>` placeholders) and then
        # `_apply_prompt_embeds_to_engine_input` augments the result.
        # Stash the tensors and placeholder ID for that override to consume.
        if prompt_embeds_tensors and mm_data:
            assert prompt_embeds_placeholder_token_id is not None
            cast(dict, prompt)["_prompt_embeds"] = (
                prompt_embeds_tensors,
                prompt_embeds_placeholder_token_id,
            )
            if params.mm_processor_kwargs:
                cast(dict, prompt)["mm_processor_kwargs"] = params.mm_processor_kwargs
        elif prompt_embeds_tensors:
            # Pure mode: no other MM data, mutate prompt to EmbedsPrompt shape.
            assert prompt_embeds_placeholder_token_id is not None
            self._apply_prompt_embeds_to_prompt(
                prompt,
                prompt_embeds_tensors,
                prompt_embeds_placeholder_token_id,
            )

        if mm_data is not None:
            prompt["multi_modal_data"] = mm_data
        if mm_uuids is not None:
            prompt["multi_modal_uuids"] = mm_uuids

        return prompt

    def render_messages(
        self,
        messages: list[ChatCompletionMessageParam],
        params: ChatParams,
    ) -> tuple[list[ConversationMessage], DictPrompt]:
        model_config = self.model_config
        tokenizer = self.get_tokenizer()

        prompt_embeds_placeholder_token_id: int | None = None
        if model_config.enable_prompt_embeds:
            prompt_embeds_placeholder_token_id = (
                _ensure_prompt_embeds_placeholder_token(tokenizer)
            )

        content_format = resolve_chat_template_content_format(
            chat_template=params.chat_template,
            tools=params.chat_template_kwargs.get("tools"),
            given_format=params.chat_template_content_format,
            tokenizer=tokenizer,
            model_config=model_config,
        )
        conversation, mm_data, mm_uuids = parse_chat_messages(
            messages,
            model_config,
            content_format=content_format,
            media_io_kwargs=params.media_io_kwargs,
            mm_processor_kwargs=params.mm_processor_kwargs,
        )

        # prompt_embeds tensors are carried by the tracker through mm_data,
        # but they must NOT be fed to the MM processor (which would reject
        # the unknown key). Extract them here.
        prompt_embeds_tensors: list[torch.Tensor] | None = None
        if mm_data is not None and "prompt_embeds" in mm_data:
            prompt_embeds_tensors = list(
                cast(Sequence[torch.Tensor], mm_data["prompt_embeds"])
            )
            mm_data = {k: v for k, v in mm_data.items() if k != "prompt_embeds"}
            if not mm_data:
                mm_data = None

        chat_template_kwargs = params.get_apply_chat_template_kwargs()
        if prompt_embeds_tensors:
            # prompt_embeds post-processing requires prompt_token_ids.
            if chat_template_kwargs.get("tokenize") is False:
                logger.warning_once(_TOKENIZE_OVERRIDE_WARNING)
            chat_template_kwargs["tokenize"] = True

        render_conversation = conversation
        protection = self._prepare_content_protection(
            conversation,
            params,
            prompt_embeds_tensors,
            content_format=content_format,
            has_multimodal=bool(mm_data),
        )
        if protection is not None:
            render_conversation = protection.skeleton_conversation
            if chat_template_kwargs.get("tokenize") is False:
                logger.warning_once(_CONTENT_PROTECTION_TOKENIZE_OVERRIDE_WARNING)
            chat_template_kwargs["tokenize"] = True

        # use_unified_vision_chunk (Kimi-K2.5) expands per-video prompts into the
        # rendered *string* before tokenization; see the two helpers above.
        kimi_vision_chunk, video_placeholder = self._resolve_video_chunk_placeholder(
            mm_uuids, mm_data, params
        )

        assistant_tokens_mask: list[int] | None = None
        if video_placeholder is not None:
            assert mm_data is not None  # implied by kimi_vision_chunk
            prompt_raw = self._render_with_video_chunk_substitution(
                tokenizer,
                render_conversation,
                chat_template_kwargs,
                mm_data,
                video_placeholder,
            )
        elif params.return_assistant_tokens_mask:
            prompt_raw, assistant_tokens_mask = safe_apply_chat_template(
                model_config,
                tokenizer,
                render_conversation,
                return_assistant_tokens_mask=True,
                **chat_template_kwargs,
            )
        else:
            prompt_raw = safe_apply_chat_template(
                model_config,
                tokenizer,
                render_conversation,
                **chat_template_kwargs,
            )

        prompt = self._finalize_rendered_prompt(
            prompt_raw,
            protection=protection,
            kimi_vision_chunk=kimi_vision_chunk,
            mm_data=mm_data,
            mm_uuids=mm_uuids,
            assistant_tokens_mask=assistant_tokens_mask,
            prompt_embeds_tensors=prompt_embeds_tensors,
            prompt_embeds_placeholder_token_id=prompt_embeds_placeholder_token_id,
            params=params,
        )
        return conversation, prompt

    async def render_messages_async(
        self,
        messages: list[ChatCompletionMessageParam],
        params: ChatParams,
    ) -> tuple[list[ConversationMessage], DictPrompt]:
        model_config = self.model_config
        tokenizer = self.get_tokenizer()

        prompt_embeds_placeholder_token_id: int | None = None
        if model_config.enable_prompt_embeds:
            prompt_embeds_placeholder_token_id = (
                _ensure_prompt_embeds_placeholder_token(tokenizer)
            )

        content_format = resolve_chat_template_content_format(
            chat_template=params.chat_template,
            tools=params.chat_template_kwargs.get("tools"),
            given_format=params.chat_template_content_format,
            tokenizer=tokenizer,
            model_config=model_config,
        )
        conversation, mm_data, mm_uuids = await parse_chat_messages_async(
            messages,
            model_config,
            content_format=content_format,
            media_io_kwargs=params.media_io_kwargs,
            mm_processor_kwargs=params.mm_processor_kwargs,
        )

        prompt_embeds_tensors: list[torch.Tensor] | None = None
        if mm_data is not None and "prompt_embeds" in mm_data:
            prompt_embeds_tensors = list(
                cast(Sequence[torch.Tensor], mm_data["prompt_embeds"])
            )
            mm_data = {k: v for k, v in mm_data.items() if k != "prompt_embeds"}
            if not mm_data:
                mm_data = None

        chat_template_kwargs = params.get_apply_chat_template_kwargs()
        if prompt_embeds_tensors:
            # prompt_embeds post-processing requires prompt_token_ids.
            if chat_template_kwargs.get("tokenize") is False:
                logger.warning_once(_TOKENIZE_OVERRIDE_WARNING)
            chat_template_kwargs["tokenize"] = True

        render_conversation = conversation
        protection = (
            await self._prepare_content_protection_async(
                conversation,
                params,
                prompt_embeds_tensors,
                content_format=content_format,
                has_multimodal=bool(mm_data),
            )
            if self._content_protection
            else None
        )
        if protection is not None:
            render_conversation = protection.skeleton_conversation
            if chat_template_kwargs.get("tokenize") is False:
                logger.warning_once(_CONTENT_PROTECTION_TOKENIZE_OVERRIDE_WARNING)
            chat_template_kwargs["tokenize"] = True

        # use_unified_vision_chunk (Kimi-K2.5) expands per-video prompts into the
        # rendered *string* before tokenization; see the two helpers above.
        kimi_vision_chunk, video_placeholder = self._resolve_video_chunk_placeholder(
            mm_uuids, mm_data, params
        )

        assistant_tokens_mask: list[int] | None = None
        prompt_raw: str | list[int]
        if video_placeholder is not None:
            assert mm_data is not None  # implied by kimi_vision_chunk
            prompt_raw = await self._render_with_video_chunk_substitution_async(
                tokenizer,
                render_conversation,
                chat_template_kwargs,
                mm_data,
                video_placeholder,
            )
        elif params.return_assistant_tokens_mask:
            result_with_mask = cast(
                tuple[list[int], list[int] | None],
                await make_async(
                    safe_apply_chat_template,
                    executor=self._executor,
                )(
                    model_config,
                    tokenizer,
                    render_conversation,
                    return_assistant_tokens_mask=True,  # type: ignore[arg-type]
                    **chat_template_kwargs,
                ),
            )
            prompt_raw = result_with_mask[0]
            assistant_tokens_mask = result_with_mask[1]
        else:
            prompt_raw = await self._apply_chat_template_async(
                model_config,
                tokenizer,
                render_conversation,
                **chat_template_kwargs,
            )

        prompt = self._finalize_rendered_prompt(
            prompt_raw,
            protection=protection,
            kimi_vision_chunk=kimi_vision_chunk,
            mm_data=mm_data,
            mm_uuids=mm_uuids,
            assistant_tokens_mask=assistant_tokens_mask,
            prompt_embeds_tensors=prompt_embeds_tensors,
            prompt_embeds_placeholder_token_id=prompt_embeds_placeholder_token_id,
            params=params,
        )
        return conversation, prompt

    @override
    def _process_tokens(
        self,
        prompt: TokensPrompt,
        *,
        skip_mm_cache: bool = False,
    ) -> TokensInput | MultiModalInput:
        """Pre-expand `prompt_embeds` sentinels before delegating to the MM
        processor, then attach `prompt_embeds` modality data to the result.

        Mixed mode only: the `_prompt_embeds` stash is set by
        `render_messages` when `prompt_embeds` co-exist with other MM data
        (images, audio, …).  We expand each 1-token sentinel to an N-token
        span *before* calling `super()._process_tokens()` so the MM
        processor records all placeholder offsets in the final (post-expansion)
        coordinate space, no offset shifting needed afterwards.
        """
        assistant_tokens_mask = cast(dict, prompt).pop("_assistant_tokens_mask", None)
        prompt_embeds_info = cast(dict, prompt).pop("_prompt_embeds", None)
        if prompt_embeds_info is not None:
            tensors, placeholder_token_id = prompt_embeds_info
            mm_updates = _build_prompt_embeds_updates(tensors, placeholder_token_id)
            cast(dict, prompt)["prompt_token_ids"] = _expand_prompt_embeds_placeholders(
                list(prompt["prompt_token_ids"]), mm_updates
            )
        engine_input = super()._process_tokens(prompt, skip_mm_cache=skip_mm_cache)
        if prompt_embeds_info is not None:
            tensors, _ = prompt_embeds_info
            self._apply_prompt_embeds_to_engine_input(
                cast(MultiModalInput, engine_input),
                tensors,
                mm_updates,
            )
        if assistant_tokens_mask is not None:
            engine_input["assistant_tokens_mask"] = assistant_tokens_mask
        return engine_input

    @override
    async def _process_tokens_async(
        self,
        prompt: TokensPrompt,
        *,
        skip_mm_cache: bool = False,
    ) -> TokensInput | MultiModalInput:
        """Async equivalent of `_process_tokens`."""
        assistant_tokens_mask = cast(dict, prompt).pop("_assistant_tokens_mask", None)
        prompt_embeds_info = cast(dict, prompt).pop("_prompt_embeds", None)
        if prompt_embeds_info is not None:
            tensors, placeholder_token_id = prompt_embeds_info
            mm_updates = _build_prompt_embeds_updates(tensors, placeholder_token_id)
            cast(dict, prompt)["prompt_token_ids"] = _expand_prompt_embeds_placeholders(
                list(prompt["prompt_token_ids"]), mm_updates
            )
        engine_input = await super()._process_tokens_async(
            prompt, skip_mm_cache=skip_mm_cache
        )
        if prompt_embeds_info is not None:
            tensors, _ = prompt_embeds_info
            self._apply_prompt_embeds_to_engine_input(
                cast(MultiModalInput, engine_input),
                tensors,
                mm_updates,
            )
        if assistant_tokens_mask is not None:
            engine_input["assistant_tokens_mask"] = assistant_tokens_mask
        return engine_input

    @staticmethod
    def _apply_prompt_embeds_to_prompt(
        prompt: DictPrompt,
        prompt_embeds_tensors: list[torch.Tensor],
        placeholder_token_id: int,
    ) -> None:
        """Mutate `prompt` from `TokensPrompt` to `EmbedsPrompt` shape.

        Pure `prompt_embeds` path only (no other MM modalities).  Expands
        each `<prompt_embeds>` sentinel token into an N-token span and builds
        the full-length `prompt_embeds` tensor + `prompt_is_token_ids` mask
        that the engine's `enable_prompt_embeds` worker branch consumes.
        """
        token_ids = cast(list[int] | None, prompt.get("prompt_token_ids"))
        if token_ids is None:
            raise RuntimeError(_MISSING_PROMPT_TOKEN_IDS_ERROR)

        embeds_orig_positions: list[int] = [
            i for i, tok in enumerate(token_ids) if tok == placeholder_token_id
        ]
        if len(embeds_orig_positions) != len(prompt_embeds_tensors):
            raise ValueError(
                f"Expected {len(prompt_embeds_tensors)} prompt_embeds "
                f"placeholder tokens in the rendered prompt, found "
                f"{len(embeds_orig_positions)}."
            )

        mm_updates = _build_prompt_embeds_updates(
            prompt_embeds_tensors, placeholder_token_id
        )
        expanded = _expand_prompt_embeds_placeholders(token_ids, mm_updates)
        positions = _build_prompt_embeds_positions(
            expanded, len(prompt_embeds_tensors), mm_updates
        )

        embeds_prompt = cast(EmbedsPrompt, prompt)
        embeds_prompt["prompt_token_ids"] = expanded
        full_embeds, is_token_ids_mask = _build_mixed_prompt_embeds(
            expanded, prompt_embeds_tensors, positions
        )
        embeds_prompt["prompt_embeds"] = full_embeds
        embeds_prompt["prompt_is_token_ids"] = is_token_ids_mask

    def _apply_prompt_embeds_to_engine_input(
        self,
        engine_input: MultiModalInput,
        prompt_embeds_tensors: list[torch.Tensor],
        mm_updates: MultiModalPromptUpdates,
    ) -> None:
        """Augment `engine_input` in-place with a `prompt_embeds` modality.

        Mixed mode: called after `_process_multimodal` has already run on the
        pre-expanded token IDs (expansion was done in `_process_tokens` before
        calling `super()`).  Locates the already-expanded `prompt_embeds` spans
        and adds `prompt_embeds` entries to `mm_kwargs`, `mm_hashes`, and
        `mm_placeholders`.
        """
        # token_ids already contain the pre-expanded N-token spans.
        token_ids = list(engine_input["prompt_token_ids"])

        positions = _build_prompt_embeds_positions(
            token_ids, len(prompt_embeds_tensors), mm_updates
        )

        pe_kwargs_items: list[MultiModalKwargsItem] = []
        pe_hashes: list[str] = []
        pe_placeholders: list[PlaceholderRange] = []
        mm_config = self.model_config.get_multimodal_config()
        for tensor, (start, length) in zip(
            prompt_embeds_tensors, positions, strict=True
        ):
            pe_kwargs_items.append(
                MultiModalKwargsItem(
                    {
                        "embedding": MultiModalFieldElem(
                            data=tensor,
                            field=MultiModalSharedField(batch_size=1),
                        )
                    }
                )
            )
            pe_hashes.append(
                MultiModalHasher.hash_kwargs(
                    mm_config.mm_hasher_algorithm, prompt_embeds=tensor
                )
            )
            # `is_embed=None` matches the existing image_embeds-style
            # "no encoder, just splice the tensor directly" semantics.
            pe_placeholders.append(
                PlaceholderRange(offset=start, length=length, is_embed=None)
            )

        cast(
            MultiModalKwargsItems[MultiModalKwargsItem | None],
            engine_input["mm_kwargs"],
        )["prompt_embeds"] = pe_kwargs_items
        engine_input["mm_hashes"] = {
            **engine_input["mm_hashes"],
            "prompt_embeds": pe_hashes,
        }
        cast(dict, engine_input["mm_placeholders"])["prompt_embeds"] = pe_placeholders
