# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Benchmark parser performance (streaming and non-streaming).

Replays token sequences through a parser, measures per-sample timing,
and optionally runs a scaling test to detect O(n) vs O(n^2) growth.

Supports two benchmark modes controlled by ``--mode``:
  - **streaming** (default): calls ``Parser.parse_delta()`` at the
    configured chunk size, matching the serving layer's streaming path.
  - **non-streaming**: calls ``Parser.parse()`` with the complete text,
    matching the serving layer's non-streaming path.

``--model`` selects both the trace-builder samples and (when no
``--tool-call-parser`` / ``--reasoning-parser`` flags are given)
the unified engine parser.

Available engine models:
    qwen3, glm47_moe, gemma4, minimax_m2, nemotron_v3

Examples:
    # Benchmark the unified engine parser (streaming, default)
    python benchmarks/benchmark_parsers.py --model qwen3

    # Benchmark a delegating parser (tool + reasoning)
    python benchmarks/benchmark_parsers.py \\
        --tool-call-parser glm47 --reasoning-parser glm45 --model glm47_moe

    # Non-streaming path
    python benchmarks/benchmark_parsers.py --model qwen3 --mode non-streaming

    # Both streaming and non-streaming
    python benchmarks/benchmark_parsers.py --model qwen3 --mode both

    # Fewer iterations for quick testing
    python benchmarks/benchmark_parsers.py --model qwen3 --iterations 10

    # Scaling test to detect O(n) vs O(n^2) behavior
    python benchmarks/benchmark_parsers.py --model qwen3 --scaling

    # Specific chunk size (tokens per parse_delta call)
    python benchmarks/benchmark_parsers.py --model qwen3 --chunk-size 5
"""

from __future__ import annotations

import argparse
import gc
import logging
import math
import statistics
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tests.parser.engine.replay_harness import (  # noqa: E402
    Sample,
    assert_parse_output,
    collect_output,
    make_mock_tokenizer,
    replay_non_streaming,
    replay_streaming,
)
from tests.parser.engine.trace_builder import (  # noqa: E402
    build_samples as _build_samples,
)
from tests.parser.engine.trace_builder import (
    build_scaling_sample as _build_scaling_sample,
)
from vllm.parser.abstract_parser import DelegatingParser, Parser  # noqa: E402
from vllm.parser.engine.registered_adapters import (  # noqa: E402
    Gemma4Parser,
    Glm47MoeParser,
    MinimaxM2Parser,
    NemotronV3Parser,
    Qwen3Parser,
)
from vllm.reasoning import ReasoningParserManager  # noqa: E402
from vllm.tool_parsers import ToolParserManager  # noqa: E402

logging.getLogger("vllm").setLevel(logging.WARNING)

ParserFactory = Callable[..., Parser]
ReplayFn = Callable[..., Any]


@dataclass
class TimingResult:
    sample_id: str
    token_count: int
    times_s: list[float] = field(default_factory=list)
    init_times_s: list[float] = field(default_factory=list)
    parse_times_s: list[float] = field(default_factory=list)

    @property
    def median_us(self) -> float:
        return statistics.median(self.times_s) * 1e6

    @property
    def stdev_us(self) -> float:
        if len(self.times_s) < 2:
            return 0.0
        return statistics.stdev(self.times_s) * 1e6

    @property
    def init_median_us(self) -> float:
        if not self.init_times_s:
            return 0.0
        return statistics.median(self.init_times_s) * 1e6

    @property
    def parse_median_us(self) -> float:
        if not self.parse_times_s:
            return 0.0
        return statistics.median(self.parse_times_s) * 1e6


class _FallbackVocab(dict):
    """Dict that auto-assigns synthetic IDs for missing tokens."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._next_synthetic = -1

    def __missing__(self, key: str) -> int:
        tid = self._next_synthetic
        self._next_synthetic -= 1
        self[key] = tid
        return tid


_ENGINE_PARSERS: dict[str, type[Parser]] = {
    "gemma4": Gemma4Parser,
    "glm47_moe": Glm47MoeParser,
    "minimax_m2": MinimaxM2Parser,
    "nemotron_v3": NemotronV3Parser,
    "qwen3": Qwen3Parser,
}


def _make_delegating_factory(
    tool_name: str | None,
    reasoning_name: str | None,
) -> tuple[str, ParserFactory]:
    """Build a ``(label, factory)`` for a DelegatingParser."""
    tool_cls = ToolParserManager.get_tool_parser(tool_name) if tool_name else None
    reasoning_cls = (
        ReasoningParserManager.get_reasoning_parser(reasoning_name)
        if reasoning_name
        else None
    )

    wrapped_cls = type(
        "_BenchWrappedParser",
        (DelegatingParser,),
        {
            "reasoning_parser_cls": reasoning_cls,
            "tool_parser_cls": tool_cls,
        },
    )

    parts = [p for p in (tool_name, reasoning_name) if p]
    label = f"DelegatingParser({', '.join(parts)})" if parts else "DelegatingParser"

    def factory(
        tokenizer: Any,
        tools: list[dict] | None = None,
        **kwargs: Any,
    ) -> Parser:
        original_vocab = tokenizer.get_vocab()
        tokenizer.set_vocab(_FallbackVocab(original_vocab))
        try:
            result = wrapped_cls(tokenizer, tools, **kwargs)
        finally:
            tokenizer.set_vocab(original_vocab)
        return result

    return label, factory


def _make_engine_factory(model: str) -> tuple[str, ParserFactory]:
    """Build a ``(label, factory)`` for an engine parser keyed by model."""
    parser_cls = _ENGINE_PARSERS.get(model)
    if parser_cls is None:
        raise ValueError(
            f"No engine parser for model {model!r}. "
            f"Available: {', '.join(sorted(_ENGINE_PARSERS))}"
        )

    def factory(
        tokenizer: Any,
        tools: list[dict] | None = None,
        **kwargs: Any,
    ) -> Parser:
        return parser_cls(tokenizer, tools, **kwargs)

    return parser_cls.__name__, factory


def smoke_test(
    factory: ParserFactory,
    sample: Sample,
    replay_fn: ReplayFn,
) -> tuple[bool, str]:
    """Run a single sample through *replay_fn* and check correctness."""
    tokenizer = make_mock_tokenizer(sample)
    extra_kwargs = {}
    if sample.chat_template_kwargs:
        extra_kwargs["chat_template_kwargs"] = sample.chat_template_kwargs
    parser = factory(tokenizer, sample.tools, **extra_kwargs)
    output = replay_fn(
        parser,
        sample.tokens,
        tools=sample.tools,
        prompt_token_ids=sample.prompt_token_ids,
    )
    output.reasoning = output.reasoning.strip()
    output.content = output.content.strip()
    trimmed = replace(
        sample,
        expected_reasoning=(
            sample.expected_reasoning.strip()
            if sample.expected_reasoning is not None
            else None
        ),
        expected_content=(
            sample.expected_content.strip()
            if sample.expected_content is not None
            else None
        ),
    )
    try:
        assert_parse_output(output, trimmed)
        return True, ""
    except AssertionError as e:
        return False, str(e)


@dataclass
class _RawTimings:
    total: list[float] = field(default_factory=list)
    init: list[float] = field(default_factory=list)
    parse: list[float] = field(default_factory=list)


def time_sample(
    factory: ParserFactory,
    sample: Sample,
    iterations: int,
    warmup: int,
    replay_fn: ReplayFn,
) -> _RawTimings:
    """Time replay of a sample through a parser."""
    tokenizer = make_mock_tokenizer(sample)
    extra_kwargs: dict[str, Any] = {}
    if sample.chat_template_kwargs:
        extra_kwargs["chat_template_kwargs"] = sample.chat_template_kwargs

    for _ in range(warmup):
        parser = factory(tokenizer, sample.tools, **extra_kwargs)
        replay_fn(
            parser,
            sample.tokens,
            tools=sample.tools,
            prompt_token_ids=sample.prompt_token_ids,
        )

    raw = _RawTimings()
    gc.collect()
    gc.disable()
    try:
        for _ in range(iterations):
            t0 = time.perf_counter()
            parser = factory(tokenizer, sample.tools, **extra_kwargs)
            t1 = time.perf_counter()
            replay_fn(
                parser,
                sample.tokens,
                tools=sample.tools,
                prompt_token_ids=sample.prompt_token_ids,
            )
            t2 = time.perf_counter()
            raw.total.append(t2 - t0)
            raw.init.append(t1 - t0)
            raw.parse.append(t2 - t1)
    finally:
        gc.enable()
    return raw


def print_results(
    results: list[TimingResult],
    parser_name: str,
    mode: str = "streaming",
) -> None:
    if not results:
        return

    name_w = max(9, *(len(r.sample_id) for r in results))

    print()
    print("=" * 70)
    print(f"RESULTS — {parser_name}, {mode} (median, microseconds)")
    print("=" * 70)
    hdr = (
        f"{'Sample':<{name_w}}  {'Tokens':>6}  {'Total':>8}  "
        f"{'Stdev':>7}  {'Init':>8}  {'Parse':>8}  {'Init%':>6}"
    )
    print(hdr)
    print("-" * len(hdr))

    for r in results:
        total = r.median_us
        init = r.init_median_us
        parse = r.parse_median_us
        pct = (init / total * 100) if total > 0 else 0
        print(
            f"{r.sample_id:<{name_w}}  {r.token_count:>6}  "
            f"{total:>8.1f}  {r.stdev_us:>7.1f}  "
            f"{init:>8.1f}  {parse:>8.1f}  {pct:>5.1f}%"
        )

    print("-" * len(hdr))
    all_medians = [r.median_us for r in results]
    agg_med = statistics.median(all_medians)
    agg_mean = statistics.mean(all_medians)
    print(
        f"{'AGGREGATE':<{name_w}}  {'':>6}  {agg_med:>8.1f} median, {agg_mean:.1f} mean"
    )
    print()


def _make_replay_fn(mode: str, chunk_size: int) -> ReplayFn:
    """Return the appropriate replay callable for *mode*."""
    if mode == "streaming":

        def _replay(parser, tokens, *, tools=None, prompt_token_ids=None):
            results = replay_streaming(
                parser,
                tokens,
                chunk_size=chunk_size,
                tools=tools,
                prompt_token_ids=prompt_token_ids,
            )
            return collect_output(results)

        return _replay
    return replay_non_streaming


def _run_benchmark_mode(
    factory: ParserFactory,
    parser_name: str,
    samples: list[Sample],
    iterations: int,
    warmup: int,
    chunk_size: int,
    mode: str,
) -> None:
    """Run smoke tests and timing for a single mode."""
    replay_fn = _make_replay_fn(mode, chunk_size)

    print(f"\nSmoke testing ({mode})...")
    passed_samples: list[Sample] = []
    for sample in samples:
        ok, err = smoke_test(factory, sample, replay_fn)
        if ok:
            passed_samples.append(sample)
        else:
            first_line = err.split("\n", 1)[0]
            print(f"  FAIL: {sample.id}: {first_line}")
    print(f"  {len(passed_samples)}/{len(samples)} passed")

    if not passed_samples:
        print(f"No samples passed {mode} smoke test.")
        return

    info = f"{iterations} iterations, {warmup} warmup"
    if mode == "streaming":
        info += f", chunk_size={chunk_size}"
    print(f"\nBenchmarking {len(passed_samples)} samples ({info})...")

    results: list[TimingResult] = []
    for sample in passed_samples:
        raw = time_sample(factory, sample, iterations, warmup, replay_fn)
        r = TimingResult(
            sample_id=sample.id,
            token_count=len(sample.tokens),
            times_s=raw.total,
            init_times_s=raw.init,
            parse_times_s=raw.parse,
        )
        results.append(r)
        print(f"  {sample.id} ({len(sample.tokens)} tokens) median={r.median_us:.1f}us")

    print_results(results, parser_name, mode=mode)


def run_benchmark(
    factory: ParserFactory,
    parser_name: str,
    samples: list[Sample],
    iterations: int,
    warmup: int,
    chunk_size: int,
    mode: str = "streaming",
) -> None:
    print(f"\nParser: {parser_name}")

    modes = ["streaming", "non-streaming"] if mode == "both" else [mode]
    for m in modes:
        _run_benchmark_mode(
            factory,
            parser_name,
            samples,
            iterations,
            warmup,
            chunk_size,
            m,
        )


def _run_scaling_mode(
    factory: ParserFactory,
    parser_name: str,
    sample: Sample,
    iterations: int,
    warmup: int,
    model: str | None,
    mode: str,
) -> None:
    """Run a scaling test for a single mode."""
    replay_fn = _make_replay_fn(mode, chunk_size=1)
    base_token_count = len(sample.tokens)
    multipliers = [1, 2, 4, 8, 16]

    print()
    print("=" * 70)
    print(f"SCALING TEST — {parser_name}, {mode}")
    print("=" * 70)
    print(f"Base sample: {sample.id} ({base_token_count} tokens)")
    print(f"Multipliers: {multipliers}")
    print(f"Iterations: {iterations}, Warmup: {warmup}")
    if model:
        print("Using trace-builder generated scaling samples")
    print()

    print(
        f"  {'Mult':>5}  {'Tokens':>7}  {'Median (us)':>12}  "
        f"{'Ratio':>8}  {'Complexity hint':>16}"
    )
    print(f"  {'-' * 5}  {'-' * 7}  {'-' * 12}  {'-' * 8}  {'-' * 16}")

    prev_median: float | None = None
    prev_token_count: int | None = None

    for mult in multipliers:
        if model:
            scaled_sample = _build_scaling_sample(model, base_token_count * mult)
        else:
            scaled_tokens = sample.tokens * mult
            scaled_sample = Sample(
                id=f"{sample.id}-x{mult}",
                description=f"Scaled {mult}x",
                source="benchmark",
                vocab=sample.vocab,
                tokens=scaled_tokens,
                expected_reasoning=None,
                expected_content=None,
                expected_tool_calls=None,
                tools=sample.tools,
            )

        raw = time_sample(
            factory,
            scaled_sample,
            iterations,
            warmup,
            replay_fn,
        )
        median = statistics.median(raw.total) * 1e6

        curr_token_count = len(scaled_sample.tokens)
        if prev_median is not None and prev_token_count is not None:
            ratio = median / prev_median
            token_ratio = curr_token_count / prev_token_count
            expected_nlogn = (
                token_ratio * math.log2(curr_token_count) / math.log2(prev_token_count)
            )
            expected_quadratic = token_ratio * token_ratio
            boundary_low = math.sqrt(token_ratio * expected_nlogn)
            boundary_high = math.sqrt(expected_nlogn * expected_quadratic)
            if ratio < boundary_low:
                hint = "~ O(n)"
            elif ratio < boundary_high:
                hint = "~ O(n log n)"
            else:
                hint = "~ O(n^2) !"
            ratio_str = f"{ratio:.2f}x"
        else:
            ratio_str = "-"
            hint = ""

        print(
            f"  {mult:>5}  {curr_token_count:>7}  "
            f"{median:>12.1f}  {ratio_str:>8}  {hint:>16}"
        )

        prev_median = median
        prev_token_count = curr_token_count

    print()


def run_scaling(
    factory: ParserFactory,
    parser_name: str,
    samples: list[Sample],
    iterations: int,
    warmup: int,
    scaling_sample_id: str | None,
    model: str | None = None,
    mode: str = "streaming",
) -> None:
    if scaling_sample_id:
        sample = next((s for s in samples if s.id == scaling_sample_id), None)
        if sample is None:
            print(f"ERROR: Sample '{scaling_sample_id}' not found.")
            print(f"Available: {', '.join(s.id for s in samples)}")
            sys.exit(1)
    else:
        sample = max(samples, key=lambda s: len(s.tokens))

    modes = ["streaming", "non-streaming"] if mode == "both" else [mode]
    for m in modes:
        _run_scaling_mode(
            factory,
            parser_name,
            sample,
            iterations,
            warmup,
            model,
            m,
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        metavar="MODEL",
        help="Model name for trace-builder samples and (when no "
        "--tool-call-parser/--reasoning-parser is given) the engine "
        "parser. Available: " + ", ".join(sorted(_ENGINE_PARSERS)),
    )
    parser.add_argument(
        "--tool-call-parser",
        type=str,
        default=None,
        metavar="NAME",
        help="Registered tool-call parser name (e.g. glm47, qwen3). "
        "When given, benchmarks a DelegatingParser instead of the "
        "engine parser.",
    )
    parser.add_argument(
        "--reasoning-parser",
        type=str,
        default=None,
        metavar="NAME",
        help="Registered reasoning parser name (e.g. glm45, qwen3). "
        "When given, benchmarks a DelegatingParser instead of the "
        "engine parser.",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=100,
        help="Measured iterations per sample (default: 100)",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=5,
        help="Warmup iterations (default: 5)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1,
        help="Tokens per parse_delta call (default: 1 = token-by-token)",
    )
    parser.add_argument(
        "--scaling",
        action="store_true",
        help="Run scaling test to detect O(n) vs O(n^2) behavior",
    )
    parser.add_argument(
        "--scaling-sample",
        type=str,
        default=None,
        help="Sample ID for scaling test (default: longest sample)",
    )
    parser.add_argument(
        "--mode",
        choices=["streaming", "non-streaming", "both"],
        default="streaming",
        help="Benchmark mode: streaming (parse_delta), "
        "non-streaming (parse), or both (default: streaming)",
    )
    args = parser.parse_args()

    if args.tool_call_parser or args.reasoning_parser:
        label, factory = _make_delegating_factory(
            args.tool_call_parser, args.reasoning_parser
        )
    else:
        label, factory = _make_engine_factory(args.model)

    try:
        samples = _build_samples(args.model)
    except KeyError:
        from tests.parser.engine.trace_builder import _BUILDERS

        parser.error(
            f"Unknown model {args.model!r}. Available: {', '.join(sorted(_BUILDERS))}"
        )

    if not samples:
        print(f"No samples found from trace-builder ({args.model})")
        sys.exit(1)

    print(f"Loaded {len(samples)} samples from trace-builder ({args.model})")
    total_tokens = sum(len(s.tokens) for s in samples)
    print(f"Total tokens across all samples: {total_tokens}")

    if args.scaling:
        run_scaling(
            factory,
            label,
            samples,
            args.iterations,
            args.warmup,
            args.scaling_sample,
            model=args.model,
            mode=args.mode,
        )
    else:
        run_benchmark(
            factory,
            label,
            samples,
            args.iterations,
            args.warmup,
            args.chunk_size,
            mode=args.mode,
        )


if __name__ == "__main__":
    main()
