# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Benchmark streaming parser performance.

Replays token sequences through a parser at token-by-token granularity,
measures per-sample timing, and optionally runs a scaling test to detect
O(n) vs O(n^2) growth.

The parser is specified as either an engine name (e.g. ``qwen3_engine``)
or a comma-separated tool,reasoning pair (e.g. ``qwen3_coder,qwen3``).

Examples:
    python benchmarks/benchmark_parsers.py qwen3_engine
    python benchmarks/benchmark_parsers.py qwen3_coder,qwen3 --model qwen3
    python benchmarks/benchmark_parsers.py qwen3_engine --scaling
"""

from __future__ import annotations

import argparse
import gc
import logging
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
    DeepSeekV4Parser,
    Gemma4Parser,
    NemotronV3Parser,
    Qwen3Parser,
    Qwen3XMLParser,
)
from vllm.reasoning import ReasoningParserManager  # noqa: E402
from vllm.tool_parsers import ToolParserManager  # noqa: E402

logging.getLogger("vllm").setLevel(logging.WARNING)

ParserFactory = Callable[..., Parser]


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
    "deepseek_v4_engine": DeepSeekV4Parser,
    "gemma4_engine": Gemma4Parser,
    "qwen3_engine": Qwen3Parser,
    "qwen3_xml_engine": Qwen3XMLParser,
    "qwen3_coder_engine": Qwen3XMLParser,
    "nemotron_v3_engine": NemotronV3Parser,
}

_PARSER_TO_MODEL: dict[str, str] = {
    name: name.removesuffix("_engine") for name in _ENGINE_PARSERS
}
_PARSER_TO_MODEL.update(
    {
        "qwen3_xml_engine": "qwen3",
        "qwen3_coder_engine": "qwen3",
        "deepseek_r1": "deepseek_v4",
    }
)


def _infer_model(spec: str) -> str | None:
    if "," in spec:
        _, reasoning = spec.split(",", 1)
        return _PARSER_TO_MODEL.get(reasoning, reasoning)
    return _PARSER_TO_MODEL.get(spec)


def _make_parser_factory(spec: str) -> tuple[str, ParserFactory]:
    """Build a ``(label, factory)`` from a parser spec."""
    if "," in spec:
        tool_name, reasoning_name = spec.split(",", 1)
        tool_cls = ToolParserManager.get_tool_parser(tool_name)
        reasoning_cls = ReasoningParserManager.get_reasoning_parser(reasoning_name)

        wrapped_cls = type(
            "_BenchWrappedParser",
            (DelegatingParser,),
            {
                "reasoning_parser_cls": reasoning_cls,
                "tool_parser_cls": tool_cls,
            },
        )

        def factory(
            tokenizer: Any,
            tools: list[dict] | None = None,
            **kwargs: Any,
        ) -> Parser:
            original_vocab = tokenizer.get_vocab()
            tokenizer.set_vocab(_FallbackVocab(original_vocab))
            result = wrapped_cls(tokenizer, tools, **kwargs)
            tokenizer.set_vocab(original_vocab)
            return result

        return spec, factory

    if spec not in _ENGINE_PARSERS:
        raise ValueError(
            f"Unknown parser: {spec!r}. Engine parsers: "
            f"{', '.join(sorted(_ENGINE_PARSERS))}"
        )
    parser_cls = _ENGINE_PARSERS[spec]

    def factory(
        tokenizer: Any,
        tools: list[dict] | None = None,
        **kwargs: Any,
    ) -> Parser:
        return parser_cls(tokenizer, tools, **kwargs)

    return spec, factory


def smoke_test(
    factory: ParserFactory,
    sample: Sample,
) -> tuple[bool, str]:
    """Run a single sample and check correctness."""
    tokenizer = make_mock_tokenizer(sample)
    extra_kwargs = {}
    if sample.chat_template_kwargs:
        extra_kwargs["chat_template_kwargs"] = sample.chat_template_kwargs
    parser = factory(tokenizer, sample.tools, **extra_kwargs)
    results = replay_streaming(parser, sample.tokens, chunk_size=1, tools=sample.tools)
    output = collect_output(results)
    output.reasoning = output.reasoning.strip()
    output.content = output.content.strip()
    _strip = lambda s: s.strip() if s is not None else None
    trimmed = replace(
        sample,
        expected_reasoning=_strip(sample.expected_reasoning),
        expected_content=_strip(sample.expected_content),
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
    chunk_size: int,
) -> _RawTimings:
    """Time replay of a sample through a parser."""
    tokenizer = make_mock_tokenizer(sample)
    extra_kwargs: dict[str, Any] = {}
    if sample.chat_template_kwargs:
        extra_kwargs["chat_template_kwargs"] = sample.chat_template_kwargs

    for _ in range(warmup):
        parser = factory(tokenizer, sample.tools, **extra_kwargs)
        replay_streaming(
            parser, sample.tokens, chunk_size=chunk_size, tools=sample.tools
        )

    raw = _RawTimings()
    gc.collect()
    gc.disable()
    try:
        for _ in range(iterations):
            t0 = time.perf_counter()
            parser = factory(tokenizer, sample.tools, **extra_kwargs)
            t1 = time.perf_counter()
            results = replay_streaming(
                parser,
                sample.tokens,
                chunk_size=chunk_size,
                tools=sample.tools,
            )
            collect_output(results)
            t2 = time.perf_counter()
            raw.total.append(t2 - t0)
            raw.init.append(t1 - t0)
            raw.parse.append(t2 - t1)
    finally:
        gc.enable()
    return raw


def print_results(results: list[TimingResult]) -> None:
    if not results:
        return

    name_w = max(len(r.sample_id) for r in results)
    name_w = max(name_w, 9)

    print()
    print("=" * 70)
    print("RESULTS (median, microseconds)")
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


def run_benchmark(
    factory: ParserFactory,
    parser_name: str,
    samples: list[Sample],
    iterations: int,
    warmup: int,
    chunk_size: int,
) -> None:
    print(f"\nParser: {parser_name}")

    print("Smoke testing...")
    passed_samples: list[Sample] = []
    for sample in samples:
        ok, err = smoke_test(factory, sample)
        if ok:
            passed_samples.append(sample)
        else:
            first_line = err.split("\n", 1)[0]
            print(f"  FAIL: {sample.id}: {first_line}")
    print(f"  {len(passed_samples)}/{len(samples)} passed")

    if not passed_samples:
        print("No samples passed smoke test.")
        return

    print(
        f"\nBenchmarking {len(passed_samples)} samples "
        f"({iterations} iterations, {warmup} warmup, "
        f"chunk_size={chunk_size})..."
    )

    results: list[TimingResult] = []
    for sample in passed_samples:
        raw = time_sample(factory, sample, iterations, warmup, chunk_size)
        r = TimingResult(
            sample_id=sample.id,
            token_count=len(sample.tokens),
            times_s=raw.total,
            init_times_s=raw.init,
            parse_times_s=raw.parse,
        )
        results.append(r)
        print(f"  {sample.id} ({len(sample.tokens)} tokens) median={r.median_us:.1f}us")

    print_results(results)


def run_scaling(
    factory: ParserFactory,
    parser_name: str,
    samples: list[Sample],
    iterations: int,
    warmup: int,
    scaling_sample_id: str | None,
    model: str | None = None,
) -> None:
    if scaling_sample_id:
        sample = next((s for s in samples if s.id == scaling_sample_id), None)
        if sample is None:
            print(f"ERROR: Sample '{scaling_sample_id}' not found.")
            print(f"Available: {', '.join(s.id for s in samples)}")
            sys.exit(1)
    else:
        sample = max(samples, key=lambda s: len(s.tokens))

    base_token_count = len(sample.tokens)
    multipliers = [1, 2, 4, 8, 16]

    print()
    print("=" * 70)
    print(f"SCALING TEST — {parser_name}")
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
    prev_mult: int | None = None

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

        raw = time_sample(factory, scaled_sample, iterations, warmup, chunk_size=1)
        median = statistics.median(raw.total) * 1e6

        if prev_median and prev_mult:
            ratio = median / prev_median
            token_ratio = mult / prev_mult
            if ratio < token_ratio * 1.3:
                hint = "~ O(n)"
            elif ratio < token_ratio * token_ratio * 0.8:
                hint = "~ O(n log n)"
            else:
                hint = "~ O(n^2) !"
            ratio_str = f"{ratio:.2f}x"
        else:
            ratio_str = "-"
            hint = ""

        print(
            f"  {mult:>5}  {len(scaled_sample.tokens):>7}  "
            f"{median:>12.1f}  {ratio_str:>8}  {hint:>16}"
        )

        prev_median = median
        prev_mult = mult

    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "parser",
        metavar="PARSER",
        help="Parser spec: engine name (e.g. qwen3_engine) or "
        "comma-separated tool,reasoning pair (e.g. qwen3_coder,qwen3)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        metavar="MODEL",
        help="Model name for trace-builder samples (e.g. qwen3, gemma4, "
        "deepseek_v4, nemotron_v3). Auto-inferred from parser name "
        "if omitted.",
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
    args = parser.parse_args()

    label, factory = _make_parser_factory(args.parser)

    if not args.model:
        args.model = _infer_model(args.parser)
        if not args.model:
            parser.error("Cannot infer model from parser name; provide --model")

    samples = _build_samples(args.model)

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
        )
    else:
        run_benchmark(
            factory,
            label,
            samples,
            args.iterations,
            args.warmup,
            args.chunk_size,
        )


if __name__ == "__main__":
    main()
