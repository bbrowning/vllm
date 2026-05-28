# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Benchmark comparing old (hand-rolled) vs new (grammar-based) streaming parsers.

Replays captured token sequences through parsers at token-by-token granularity,
measures per-sample timing, and optionally runs a scaling test to detect
O(n) vs O(n^2) growth.

Examples:
    # Compare old vs new Qwen3 parsers
    python benchmarks/benchmark_parsers.py \\
        tests/parser/grammar/data/qwen3.jsonl \\
        --old qwen3_xml,qwen3 --new qwen3_grammar

    # Scaling test to detect algorithmic complexity
    python benchmarks/benchmark_parsers.py \\
        tests/parser/grammar/data/qwen3.jsonl \\
        --old qwen3_xml,qwen3 --new qwen3_grammar --scaling

    # Benchmark only the new parser
    python benchmarks/benchmark_parsers.py \\
        tests/parser/grammar/data/qwen3.jsonl \\
        --new qwen3_grammar --iterations 200
"""

from __future__ import annotations

import argparse
import logging
import statistics
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Ensure project root is on sys.path so ``tests`` is importable.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tests.parser.grammar.replay_harness import (  # noqa: E402
    Sample,
    assert_parse_output,
    collect_output,
    load_samples_from_path,
    make_mock_tokenizer,
    replay_streaming,
)
from vllm.parser import ParserManager, _WrappedParser  # noqa: E402
from vllm.parser.abstract_parser import Parser  # noqa: E402
from vllm.reasoning import ReasoningParserManager  # noqa: E402
from vllm.tool_parsers import ToolParserManager  # noqa: E402

logging.getLogger("vllm").setLevel(logging.WARNING)


ParserFactory = Callable[..., Parser]


@dataclass
class TimingResult:
    sample_id: str
    parser_name: str
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
    def min_us(self) -> float:
        return min(self.times_s) * 1e6

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
    """Dict that auto-assigns synthetic IDs for missing tokens.

    Old parsers do hard ``self.vocab["<|turn>"]`` lookups for tokens that
    may not appear in the sample data.  This avoids KeyError by assigning
    a unique negative ID for any token not in the sample's vocab.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._next_synthetic = -1

    def __missing__(self, key: str) -> int:
        tid = self._next_synthetic
        self._next_synthetic -= 1
        self[key] = tid
        return tid


def _make_old_parser_factory(
    tool_parser_name: str, reasoning_parser_name: str
) -> ParserFactory:
    """Create a factory that builds a DelegatingParser from old-style parsers."""
    tool_cls = ToolParserManager.get_tool_parser(tool_parser_name)
    reasoning_cls = ReasoningParserManager.get_reasoning_parser(reasoning_parser_name)

    def factory(
        tokenizer: Any,
        tools: list[dict] | None = None,
        **kwargs: Any,
    ) -> Parser:
        # Enrich the mock tokenizer vocab so old parsers can look up
        # tokens that aren't in the sample data (e.g. <|turn>)
        original_vocab = tokenizer.get_vocab()
        tokenizer.set_vocab(_FallbackVocab(original_vocab))

        cls = type(
            "_BenchWrappedParser",
            (_WrappedParser,),
            {
                "reasoning_parser_cls": reasoning_cls,
                "tool_parser_cls": tool_cls,
            },
        )
        result = cls(tokenizer, tools, **kwargs)
        tokenizer.set_vocab(original_vocab)
        return result

    return factory


def _make_new_parser_factory(parser_name: str) -> ParserFactory:
    """Create a factory that builds a unified grammar parser."""
    parser_cls = ParserManager.get_parser_internal(parser_name)

    def factory(
        tokenizer: Any,
        tools: list[dict] | None = None,
        **kwargs: Any,
    ) -> Parser:
        return parser_cls(tokenizer, tools, **kwargs)

    return factory


def smoke_test(
    factory: ParserFactory,
    sample: Sample,
    name: str,
) -> tuple[bool, str]:
    """Run a single sample and check correctness.

    Returns ``(True, "")`` on pass, ``(False, error_message)`` on failure.
    Leading/trailing whitespace differences in reasoning and content are
    tolerated since we are comparing parser *work*, not exact formatting.
    """
    tokenizer = make_mock_tokenizer(sample)
    extra_kwargs = {}
    if sample.chat_template_kwargs:
        extra_kwargs["chat_template_kwargs"] = sample.chat_template_kwargs
    parser = factory(tokenizer, sample.tools, **extra_kwargs)
    results = replay_streaming(parser, sample.tokens, chunk_size=1)
    output = collect_output(results)
    output.reasoning = output.reasoning.strip()
    output.content = output.content.strip()
    trimmed_sample = Sample(
        id=sample.id,
        description=sample.description,
        source=sample.source,
        vocab=sample.vocab,
        tokens=sample.tokens,
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
        expected_tool_calls=sample.expected_tool_calls,
        tools=sample.tools,
        chat_template_kwargs=sample.chat_template_kwargs,
    )
    try:
        assert_parse_output(output, trimmed_sample)
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
    """Time replay of a sample through a parser. Returns timing breakdown."""
    tokenizer = make_mock_tokenizer(sample)
    extra_kwargs: dict[str, Any] = {}
    if sample.chat_template_kwargs:
        extra_kwargs["chat_template_kwargs"] = sample.chat_template_kwargs

    for _ in range(warmup):
        parser = factory(tokenizer, sample.tools, **extra_kwargs)
        replay_streaming(parser, sample.tokens, chunk_size=chunk_size)

    raw = _RawTimings()
    for _ in range(iterations):
        t0 = time.perf_counter()
        parser = factory(tokenizer, sample.tools, **extra_kwargs)
        t1 = time.perf_counter()
        results = replay_streaming(parser, sample.tokens, chunk_size=chunk_size)
        collect_output(results)
        t2 = time.perf_counter()
        raw.total.append(t2 - t0)
        raw.init.append(t1 - t0)
        raw.parse.append(t2 - t1)
    return raw


def print_comparison_table(
    results: dict[str, list[TimingResult]],
    parser_names: list[str],
    all_sample_ids: list[str] | None = None,
) -> None:
    """Print a per-sample timing comparison table.

    When *all_sample_ids* is provided, every sample is shown even if some
    parser+sample combinations failed the smoke test (displayed as FAIL).
    Aggregate statistics only include samples where **all** parsers passed.
    """
    lookup: dict[tuple[str, str], TimingResult] = {}
    for name, r_list in results.items():
        for r in r_list:
            lookup[(name, r.sample_id)] = r

    if all_sample_ids is not None:
        sample_ids = all_sample_ids
    else:
        sample_ids = []
        seen: set[str] = set()
        for r_list in results.values():
            for r in r_list:
                if r.sample_id not in seen:
                    sample_ids.append(r.sample_id)
                    seen.add(r.sample_id)

    token_counts: dict[str, int] = {}
    for r_list in results.values():
        for r in r_list:
            token_counts[r.sample_id] = r.token_count

    name_width = max((len(sid) for sid in sample_ids), default=8)
    name_width = max(name_width, 8)

    col_width = 22
    print()
    print("=" * 70)
    print("PER-SAMPLE RESULTS (median, microseconds)")
    print("=" * 70)

    header = f"{'Sample':<{name_width}}  {'Tokens':>6}"
    for name in parser_names:
        header += f"  {name:>{col_width}}"
    if len(parser_names) == 2:
        header += f"  {'Speedup':>8}"
    print(header)
    print("-" * len(header))

    for sid in sample_ids:
        tok_count = token_counts.get(sid, 0)
        tok_str = str(tok_count) if tok_count else "?"
        line = f"{sid:<{name_width}}  {tok_str:>6}"
        medians: list[float] = []
        for name in parser_names:
            r = lookup.get((name, sid))
            if r:
                med = r.median_us
                std = r.stdev_us
                cell = f"{med:>8.1f} +/- {std:<7.1f}"
                medians.append(med)
            else:
                cell = f"{'FAIL':>{col_width}}"
                medians.append(0.0)
            line += f"  {cell:>{col_width}}"

        if len(parser_names) == 2 and all(m > 0 for m in medians):
            speedup = medians[0] / medians[1]
            line += f"  {speedup:>7.2f}x"
        elif len(parser_names) == 2:
            line += f"  {'--':>8}"
        print(line)

    print("-" * len(header))

    common_ids = [
        sid for sid in sample_ids if all((name, sid) in lookup for name in parser_names)
    ]

    n_excluded = len(sample_ids) - len(common_ids)
    agg_label = "AGGREGATE"
    if n_excluded:
        agg_label += f" ({n_excluded} excluded)"

    agg_line = f"{agg_label:<{name_width}}  {'':>6}"
    agg_medians: list[float] = []
    for name in parser_names:
        all_medians = [
            lookup[(name, sid)].median_us for sid in common_ids if (name, sid) in lookup
        ]
        if all_medians:
            agg_med = statistics.median(all_medians)
            agg_mean = statistics.mean(all_medians)
            cell = f"{agg_med:>8.1f} (mean {agg_mean:.1f})"
            agg_medians.append(agg_med)
        else:
            cell = f"{'N/A':>{col_width}}"
            agg_medians.append(0.0)
        agg_line += f"  {cell:>{col_width}}"

    if len(parser_names) == 2 and all(m > 0 for m in agg_medians):
        speedup = agg_medians[0] / agg_medians[1]
        agg_line += f"  {speedup:>7.2f}x"
    print(agg_line)
    print()


def print_breakdown_table(
    results: dict[str, list[TimingResult]],
    parser_names: list[str],
) -> None:
    """Print init vs parse timing breakdown for each parser."""
    if not any(r.init_times_s for rl in results.values() for r in rl):
        return

    print("=" * 70)
    print("INIT vs PARSE BREAKDOWN (median, microseconds)")
    print("=" * 70)

    for name in parser_names:
        r_list = results.get(name, [])
        if not r_list:
            continue
        print(f"\n  {name}:")
        print(
            f"    {'Sample':<40}  {'Init':>8}  {'Parse':>8}  {'Total':>8}  {'Init%':>6}"
        )
        print(f"    {'-' * 40}  {'-' * 8}  {'-' * 8}  {'-' * 8}  {'-' * 6}")
        for r in r_list:
            init_us = r.init_median_us
            parse_us = r.parse_median_us
            total_us = r.median_us
            pct = (init_us / total_us * 100) if total_us > 0 else 0
            print(
                f"    {r.sample_id:<40}  {init_us:>8.1f}  "
                f"{parse_us:>8.1f}  {total_us:>8.1f}  {pct:>5.1f}%"
            )
    print()


def run_comparison(
    samples: list[Sample],
    factories: dict[str, ParserFactory],
    iterations: int,
    warmup: int,
    chunk_size: int,
) -> None:
    """Run the comparison benchmark."""
    parser_names = list(factories.keys())
    all_sample_ids = [s.id for s in samples]

    print("\nSmoke testing all parser+sample combinations...")
    passed: set[tuple[str, str]] = set()
    failed: list[tuple[str, str, str]] = []
    for name, factory in factories.items():
        for sample in samples:
            ok, err = smoke_test(factory, sample, name)
            if ok:
                passed.add((name, sample.id))
            else:
                failed.append((name, sample.id, err))

    n_total = len(factories) * len(samples)
    print(f"  {len(passed)}/{n_total} passed")
    if failed:
        for pname, sid, err in failed:
            first_line = err.split("\n", 1)[0]
            print(f"  FAIL: {pname} x {sid}: {first_line}")

    print(
        f"\nBenchmarking {len(samples)} samples x {len(factories)} parsers "
        f"({iterations} iterations, {warmup} warmup, "
        f"chunk_size={chunk_size})..."
    )

    results: dict[str, list[TimingResult]] = {n: [] for n in parser_names}
    for sample in samples:
        for name, factory in factories.items():
            if (name, sample.id) not in passed:
                print(f"  {name}: {sample.id} -- SKIPPED (smoke test failed)")
                continue
            raw = time_sample(factory, sample, iterations, warmup, chunk_size)
            results[name].append(
                TimingResult(
                    sample_id=sample.id,
                    parser_name=name,
                    token_count=len(sample.tokens),
                    times_s=raw.total,
                    init_times_s=raw.init,
                    parse_times_s=raw.parse,
                )
            )
            med_us = statistics.median(raw.total) * 1e6
            print(
                f"  {name}: {sample.id} ({len(sample.tokens)} tokens) "
                f"median={med_us:.1f}us"
            )

    print_comparison_table(results, parser_names, all_sample_ids)
    print_breakdown_table(results, parser_names)


def run_scaling(
    samples: list[Sample],
    factories: dict[str, ParserFactory],
    iterations: int,
    warmup: int,
    scaling_sample_id: str | None,
) -> None:
    """Run scaling test to detect O(n) vs O(n^2) behavior."""
    if scaling_sample_id:
        sample = next((s for s in samples if s.id == scaling_sample_id), None)
        if sample is None:
            print(f"ERROR: Sample '{scaling_sample_id}' not found.")
            print(f"Available: {', '.join(s.id for s in samples)}")
            sys.exit(1)
    else:
        sample = max(samples, key=lambda s: len(s.tokens))

    multipliers = [1, 2, 4, 8, 16]

    print()
    print("=" * 70)
    print("SCALING TEST")
    print("=" * 70)
    print(f"Base sample: {sample.id} ({len(sample.tokens)} tokens)")
    print(f"Multipliers: {multipliers}")
    print(f"Iterations: {iterations}, Warmup: {warmup}")
    print()

    for name, factory in factories.items():
        print(f"Parser: {name}")
        print(
            f"  {'Mult':>5}  {'Tokens':>7}  {'Median (us)':>12}  "
            f"{'Ratio':>8}  {'Complexity hint':>16}"
        )
        print(f"  {'-' * 5}  {'-' * 7}  {'-' * 12}  {'-' * 8}  {'-' * 16}")

        prev_median: float | None = None
        prev_mult: int | None = None

        for mult in multipliers:
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
                # O(n) -> ratio ~= token_ratio
                # O(n^2) -> ratio ~= token_ratio^2
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
                f"  {mult:>5}  {len(scaled_tokens):>7}  "
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
        "samples_file",
        type=Path,
        help="Path to JSONL samples file (e.g. tests/parser/grammar/data/qwen3.jsonl)",
    )
    parser.add_argument(
        "--old",
        type=str,
        default=None,
        metavar="TOOL,REASONING",
        help="Old parser: comma-separated tool parser name and reasoning "
        "parser name (e.g. qwen3_xml,qwen3)",
    )
    parser.add_argument(
        "--new",
        type=str,
        default=None,
        metavar="PARSER",
        help="New unified parser name (e.g. qwen3_grammar)",
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

    if not args.old and not args.new:
        parser.error("At least one of --old or --new is required")

    if not args.samples_file.exists():
        parser.error(f"Samples file not found: {args.samples_file}")

    factories: dict[str, ParserFactory] = {}
    if args.old:
        parts = args.old.split(",")
        if len(parts) != 2:
            parser.error(
                "--old requires exactly two comma-separated names: "
                "TOOL_PARSER,REASONING_PARSER"
            )
        tool_name, reasoning_name = parts
        factories[f"old ({tool_name})"] = _make_old_parser_factory(
            tool_name, reasoning_name
        )

    if args.new:
        factories[f"new ({args.new})"] = _make_new_parser_factory(args.new)

    samples = load_samples_from_path(args.samples_file)
    if not samples:
        print(f"No samples found in {args.samples_file}")
        sys.exit(1)

    print(f"Loaded {len(samples)} samples from {args.samples_file}")
    total_tokens = sum(len(s.tokens) for s in samples)
    print(f"Total tokens across all samples: {total_tokens}")

    if args.scaling:
        run_scaling(
            samples,
            factories,
            args.iterations,
            args.warmup,
            args.scaling_sample,
        )
    else:
        run_comparison(
            samples,
            factories,
            args.iterations,
            args.warmup,
            args.chunk_size,
        )


if __name__ == "__main__":
    main()
