# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Cross-language benchmark: vLLM Python parser vs Rust dynamo-parsers (Gemma4).

Compares the full parsing stacks fairly:
  - Reasoning: both parsers stream incrementally (apples-to-apples)
  - Tool calls: Python streams incrementally, Rust buffers then batch-parses
  - Full pipeline: combined reasoning + tool-call handling

Requires the dynamo-parsers PyO3 wrapper to be installed for Rust
measurements. Falls back to Python-only if not available.

Usage:
    # Python-only (always works)
    python benchmarks/benchmark_rust_comparison.py \\
        tests/parser/grammar/data/gemma4.jsonl

    # With Rust comparison (requires dynamo_parsers PyO3 module)
    python benchmarks/benchmark_rust_comparison.py \\
        tests/parser/grammar/data/gemma4.jsonl --with-rust

    # Scaling test
    python benchmarks/benchmark_rust_comparison.py \\
        tests/parser/grammar/data/gemma4.jsonl --with-rust --scaling
"""

from __future__ import annotations

import argparse
import gc
import json
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tests.parser.grammar.replay_harness import (  # noqa: E402
    Sample,
    collect_output,
    load_samples_from_path,
    make_mock_tokenizer,
    replay_streaming,
)
from vllm.parser import ParserManager  # noqa: E402
from vllm.tool_parsers.gemma4_utils import parse_tool_calls  # noqa: E402

try:
    import dynamo_parsers  # type: ignore[import-untyped]

    HAS_RUST = True
except ImportError:
    HAS_RUST = False


# ---------------------------------------------------------------------------
# Timing infrastructure
# ---------------------------------------------------------------------------


@dataclass
class TimingResult:
    config: str
    sample_id: str
    token_count: int
    times_us: list[float] = field(default_factory=list)
    ttfo_us: float | None = None

    @property
    def median_us(self) -> float:
        return statistics.median(self.times_us)

    @property
    def stdev_us(self) -> float:
        return statistics.stdev(self.times_us) if len(self.times_us) > 1 else 0.0

    @property
    def min_us(self) -> float:
        return min(self.times_us)


def _time_fn(fn, iterations: int, warmup: int) -> list[float]:
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(iterations):
        gc.collect()
        gc.disable()
        t0 = time.perf_counter()
        fn()
        t1 = time.perf_counter()
        gc.enable()
        times.append((t1 - t0) * 1e6)
    return times


# ---------------------------------------------------------------------------
# Python incremental (our grammar parser)
# ---------------------------------------------------------------------------


def _make_py_incremental_factory():
    parser_cls = ParserManager.get_parser_internal("gemma4_grammar")

    def run(sample: Sample, iterations: int, warmup: int) -> TimingResult:
        tokenizer = make_mock_tokenizer(sample)
        extra_kwargs: dict[str, Any] = {}
        if sample.chat_template_kwargs:
            extra_kwargs["chat_template_kwargs"] = sample.chat_template_kwargs

        def fn():
            parser = parser_cls(tokenizer, sample.tools, **extra_kwargs)
            results = replay_streaming(parser, sample.tokens, chunk_size=1)
            collect_output(results)

        times = _time_fn(fn, iterations, warmup)

        # Measure TTFO separately
        parser = parser_cls(tokenizer, sample.tools, **extra_kwargs)
        from vllm.entrypoints.openai.chat_completion.protocol import (
            ChatCompletionRequest,
        )

        request = ChatCompletionRequest(
            model="test", messages=[{"role": "user", "content": "t"}]
        )
        t0 = time.perf_counter()
        for i, (tid, text) in enumerate(sample.tokens):
            result = parser.parse_delta(
                text,
                [tid],
                request,
                prompt_token_ids=[] if i == 0 else None,
            )
            if result is not None:
                ttfo = (time.perf_counter() - t0) * 1e6
                break
        else:
            ttfo = None

        r = TimingResult(
            config="py-incremental",
            sample_id=sample.id,
            token_count=len(sample.tokens),
            times_us=times,
            ttfo_us=ttfo,
        )
        return r

    return run


# ---------------------------------------------------------------------------
# Python batch tool-call parse
# ---------------------------------------------------------------------------


def _make_py_batch_factory():
    def run(sample: Sample, iterations: int, warmup: int) -> TimingResult:
        full_text = "".join(text for _, text in sample.tokens)

        def fn():
            parse_tool_calls(full_text, strict=True)

        times = _time_fn(fn, iterations, warmup)
        return TimingResult(
            config="py-batch",
            sample_id=sample.id,
            token_count=len(sample.tokens),
            times_us=times,
        )

    return run


# ---------------------------------------------------------------------------
# Rust reasoning streaming (via PyO3)
# ---------------------------------------------------------------------------


def _make_rust_reasoning_factory():
    def run(sample: Sample, iterations: int, warmup: int) -> TimingResult:
        token_texts = [text for _, text in sample.tokens]

        def fn():
            parser = dynamo_parsers.Gemma4ReasoningParser()
            for chunk in token_texts:
                parser.parse_chunk(chunk)

        times = _time_fn(fn, iterations, warmup)

        # TTFO: time to first non-empty reasoning output
        parser = dynamo_parsers.Gemma4ReasoningParser()
        t0 = time.perf_counter()
        ttfo = None
        for chunk in token_texts:
            reasoning, normal = parser.parse_chunk(chunk)
            if reasoning or normal:
                ttfo = (time.perf_counter() - t0) * 1e6
                break

        return TimingResult(
            config="rust-reasoning",
            sample_id=sample.id,
            token_count=len(sample.tokens),
            times_us=times,
            ttfo_us=ttfo,
        )

    return run


# ---------------------------------------------------------------------------
# Rust batch tool-call parse (via PyO3)
# ---------------------------------------------------------------------------


def _make_rust_batch_factory():
    def run(sample: Sample, iterations: int, warmup: int) -> TimingResult:
        full_text = "".join(text for _, text in sample.tokens)

        def fn():
            dynamo_parsers.try_tool_call_parse_gemma4(full_text)

        times = _time_fn(fn, iterations, warmup)
        return TimingResult(
            config="rust-batch",
            sample_id=sample.id,
            token_count=len(sample.tokens),
            times_us=times,
        )

    return run


# ---------------------------------------------------------------------------
# Rust full pipeline: reasoning streaming -> tool-call jail -> batch parse
# ---------------------------------------------------------------------------


def _make_rust_pipeline_factory():
    def run(sample: Sample, iterations: int, warmup: int) -> TimingResult:
        token_texts = [text for _, text in sample.tokens]

        def fn():
            parser = dynamo_parsers.Gemma4ReasoningParser()
            normal_accum = []
            detected = False
            for chunk in token_texts:
                _reasoning, normal = parser.parse_chunk(chunk)
                if normal:
                    normal_accum.append(normal)
                    if not detected:
                        buf = "".join(normal_accum)
                        detected = dynamo_parsers.detect_tool_call_start_gemma4(buf)
                    if detected:
                        buf = "".join(normal_accum)
                        pos = dynamo_parsers.find_tool_call_end_position_gemma4(buf)
                        if pos is not None:
                            dynamo_parsers.try_tool_call_parse_gemma4(buf)

        times = _time_fn(fn, iterations, warmup)

        # TTFO: time to first output from either stage
        parser = dynamo_parsers.Gemma4ReasoningParser()
        t0 = time.perf_counter()
        ttfo = None
        for chunk in token_texts:
            reasoning, normal = parser.parse_chunk(chunk)
            if reasoning or normal:
                ttfo = (time.perf_counter() - t0) * 1e6
                break

        return TimingResult(
            config="rust-pipeline",
            sample_id=sample.id,
            token_count=len(sample.tokens),
            times_us=times,
            ttfo_us=ttfo,
        )

    return run


# ---------------------------------------------------------------------------
# Printing
# ---------------------------------------------------------------------------

COL_W = 22


def _print_table(
    title: str,
    results: dict[str, list[TimingResult]],
    configs: list[str],
    sample_ids: list[str],
    token_counts: dict[str, int],
) -> None:
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)

    name_w = max((len(s) for s in sample_ids), default=8)
    name_w = max(name_w, 8)

    header = f"{'Sample':<{name_w}}  {'Toks':>5}"
    for cfg in configs:
        header += f"  {cfg:>{COL_W}}"
    if len(configs) == 2:
        header += f"  {'Speedup':>8}"
    print(header)
    print("-" * len(header))

    for sid in sample_ids:
        toks = token_counts.get(sid, 0)
        line = f"{sid:<{name_w}}  {toks:>5}"
        medians = []
        for cfg in configs:
            r = next(
                (x for x in results.get(cfg, []) if x.sample_id == sid),
                None,
            )
            if r:
                cell = f"{r.median_us:>8.1f} +/- {r.stdev_us:<7.1f}"
                medians.append(r.median_us)
            else:
                cell = f"{'N/A':>{COL_W}}"
                medians.append(0.0)
            line += f"  {cell:>{COL_W}}"
        if len(configs) == 2 and all(m > 0 for m in medians):
            speedup = medians[0] / medians[1]
            line += f"  {speedup:>7.2f}x"
        print(line)

    print("-" * len(header))

    common = [
        sid
        for sid in sample_ids
        if all(any(r.sample_id == sid for r in results.get(cfg, [])) for cfg in configs)
    ]
    if common:
        agg_line = f"{'AGGREGATE':<{name_w}}  {'':>5}"
        agg_meds = []
        for cfg in configs:
            meds = [
                next(r for r in results[cfg] if r.sample_id == sid).median_us
                for sid in common
            ]
            agg = statistics.median(meds)
            agg_meds.append(agg)
            cell = f"{agg:>8.1f} (mean {statistics.mean(meds):.1f})"
            agg_line += f"  {cell:>{COL_W}}"
        if len(configs) == 2 and all(m > 0 for m in agg_meds):
            agg_line += f"  {agg_meds[0] / agg_meds[1]:>7.2f}x"
        print(agg_line)
    print()


def _print_ttfo_table(
    results: dict[str, list[TimingResult]],
    configs: list[str],
    sample_ids: list[str],
    token_counts: dict[str, int],
) -> None:
    print()
    print("=" * 70)
    print("TIME-TO-FIRST-OUTPUT (TTFO)")
    print("=" * 70)

    name_w = max((len(s) for s in sample_ids), default=8)
    name_w = max(name_w, 8)

    header = f"{'Sample':<{name_w}}  {'Toks':>5}"
    for cfg in configs:
        header += f"  {cfg + ' TTFO(us)':>20}"
    print(header)
    print("-" * len(header))

    for sid in sample_ids:
        toks = token_counts.get(sid, 0)
        line = f"{sid:<{name_w}}  {toks:>5}"
        for cfg in configs:
            r = next(
                (x for x in results.get(cfg, []) if x.sample_id == sid),
                None,
            )
            if r and r.ttfo_us is not None:
                cell = f"{r.ttfo_us:>8.1f}"
            else:
                cell = f"{'N/A':>8}"
            line += f"  {cell:>20}"
        print(line)

    print()


# ---------------------------------------------------------------------------
# Scaling test
# ---------------------------------------------------------------------------


def run_scaling(
    samples: list[Sample],
    configs: dict[str, Any],
    iterations: int,
    warmup: int,
    scaling_sample_id: str | None,
) -> None:
    if scaling_sample_id:
        sample = next((s for s in samples if s.id == scaling_sample_id), None)
        if sample is None:
            print(f"ERROR: Sample '{scaling_sample_id}' not found.")
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
    print()

    for cfg_name, run_fn in configs.items():
        print(f"Config: {cfg_name}")
        print(
            f"  {'Mult':>5}  {'Tokens':>7}  {'Median(us)':>12}  "
            f"{'Ratio':>8}  {'Hint':>16}"
        )
        print(f"  {'-' * 5}  {'-' * 7}  {'-' * 12}  {'-' * 8}  {'-' * 16}")

        prev_median = None
        prev_mult = None

        for mult in multipliers:
            scaled = Sample(
                id=f"{sample.id}-x{mult}",
                description=f"Scaled {mult}x",
                source="benchmark",
                vocab=sample.vocab,
                tokens=sample.tokens * mult,
                expected_reasoning=None,
                expected_content=None,
                expected_tool_calls=None,
                tools=sample.tools,
                chat_template_kwargs=sample.chat_template_kwargs,
            )
            result = run_fn(scaled, iterations, warmup)
            median = result.median_us

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
                f"  {mult:>5}  {len(scaled.tokens):>7}  "
                f"{median:>12.1f}  {ratio_str:>8}  {hint:>16}"
            )
            prev_median = median
            prev_mult = mult

        print()


# ---------------------------------------------------------------------------
# Correctness check
# ---------------------------------------------------------------------------


def verify_correctness(
    samples: list[Sample],
) -> None:
    """Verify Python and Rust parsers produce the same output for all samples."""
    if not HAS_RUST:
        return

    print("\nCorrecntess check: Python vs Rust...")
    ok = 0
    fail = 0

    for sample in samples:
        full_text = "".join(text for _, text in sample.tokens)

        # Python batch parse
        py_calls = parse_tool_calls(full_text, strict=True)

        # Rust batch parse
        rust_calls_raw, _normal = dynamo_parsers.try_tool_call_parse_gemma4(full_text)
        rust_calls = [
            {
                "name": c["name"],
                "arguments": json.loads(c["arguments"]),
            }
            for c in rust_calls_raw
        ]

        if len(py_calls) != len(rust_calls):
            print(
                f"  FAIL {sample.id}: tool call count "
                f"py={len(py_calls)} rust={len(rust_calls)}"
            )
            fail += 1
            continue

        match = True
        for i, (pc, rc) in enumerate(zip(py_calls, rust_calls)):
            if pc["name"] != rc["name"]:
                print(
                    f"  FAIL {sample.id}: call {i} name "
                    f"py={pc['name']!r} rust={rc['name']!r}"
                )
                match = False
                break
            if pc["arguments"] != rc["arguments"]:
                print(f"  FAIL {sample.id}: call {i} args differ")
                match = False
                break
        if match:
            ok += 1
        else:
            fail += 1

    print(f"  {ok}/{ok + fail} passed")
    if fail:
        print("  WARNING: some samples produced different results!")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "samples_file",
        type=Path,
        help="Path to JSONL samples file",
    )
    parser.add_argument(
        "--with-rust",
        action="store_true",
        help="Include Rust parser measurements (requires dynamo_parsers)",
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
        "--scaling",
        action="store_true",
        help="Run scaling test",
    )
    parser.add_argument(
        "--scaling-sample",
        type=str,
        default=None,
        help="Sample ID for scaling test (default: longest)",
    )
    args = parser.parse_args()

    if args.with_rust and not HAS_RUST:
        parser.error(
            "--with-rust requires the dynamo_parsers module. "
            "Build it with: cd frontend-crates/parsers-pyo3 && "
            "maturin develop --release"
        )

    samples = load_samples_from_path(args.samples_file)
    if not samples:
        print(f"No samples found in {args.samples_file}")
        sys.exit(1)

    print(f"Loaded {len(samples)} samples from {args.samples_file}")
    total_tokens = sum(len(s.tokens) for s in samples)
    print(f"Total tokens: {total_tokens}")
    if args.with_rust:
        print("Rust parser: ENABLED")
    else:
        print("Rust parser: DISABLED (use --with-rust)")

    token_counts = {s.id: len(s.tokens) for s in samples}
    all_ids = [s.id for s in samples]

    has_tool_call = [
        s for s in samples if any("<|tool_call>" in text for _, text in s.tokens)
    ]
    reasoning_only = [s for s in samples if s not in has_tool_call]

    tc_ids = [s.id for s in has_tool_call]
    reason_ids = [s.id for s in reasoning_only]

    # -- Build config runners --
    py_incr = _make_py_incremental_factory()
    py_batch = _make_py_batch_factory()

    rust_reason = _make_rust_reasoning_factory() if args.with_rust else None
    rust_batch = _make_rust_batch_factory() if args.with_rust else None
    rust_pipeline = _make_rust_pipeline_factory() if args.with_rust else None

    # -- Run correctness check --
    if args.with_rust:
        verify_correctness(has_tool_call)

    iters = args.iterations
    wu = args.warmup

    if args.scaling:
        scaling_configs: dict[str, Any] = {"py-incremental": py_incr}
        if args.with_rust:
            scaling_configs["rust-reasoning"] = rust_reason
            scaling_configs["rust-pipeline"] = rust_pipeline
        run_scaling(samples, scaling_configs, iters, wu, args.scaling_sample)
        return

    # -- Collect results --
    results: dict[str, list[TimingResult]] = {}

    configs_to_run: list[tuple[str, Any, list[Sample]]] = [
        ("py-incremental", py_incr, samples),
    ]
    if has_tool_call:
        configs_to_run.append(("py-batch", py_batch, has_tool_call))
    if args.with_rust:
        configs_to_run.append(("rust-reasoning", rust_reason, samples))
        if has_tool_call:
            configs_to_run.append(("rust-batch", rust_batch, has_tool_call))
            configs_to_run.append(("rust-pipeline", rust_pipeline, has_tool_call))

    for cfg_name, run_fn, applicable_samples in configs_to_run:
        results[cfg_name] = []
        for sample in applicable_samples:
            print(
                f"  {cfg_name}: {sample.id} ({len(sample.tokens)} tokens)...",
                end="",
                flush=True,
            )
            r = run_fn(sample, iters, wu)
            results[cfg_name].append(r)
            print(f" {r.median_us:.1f} us")

    # -- Print tables --

    # Table 1: Reasoning-only (apples-to-apples streaming)
    if reasoning_only:
        reason_configs = ["py-incremental"]
        if args.with_rust:
            reason_configs.append("rust-reasoning")
        _print_table(
            "REASONING-ONLY SAMPLES (apples-to-apples incremental streaming)",
            results,
            reason_configs,
            reason_ids,
            token_counts,
        )

    # Table 2: Full pipeline (tool-call samples)
    if has_tool_call:
        pipeline_configs = ["py-incremental"]
        if args.with_rust:
            pipeline_configs.append("rust-pipeline")
        _print_table(
            "TOOL-CALL SAMPLES — FULL PIPELINE (production-realistic)",
            results,
            pipeline_configs,
            tc_ids,
            token_counts,
        )

    # Table 3: Batch tool-call only
    if has_tool_call:
        batch_configs = ["py-batch"]
        if args.with_rust:
            batch_configs.append("rust-batch")
        _print_table(
            "BATCH TOOL-CALL PARSE (complete message -> tool calls)",
            results,
            batch_configs,
            tc_ids,
            token_counts,
        )

    # Table 4: TTFO
    ttfo_configs = ["py-incremental"]
    if args.with_rust:
        ttfo_configs.extend(["rust-reasoning", "rust-pipeline"])
    _print_ttfo_table(results, ttfo_configs, all_ids, token_counts)

    # Production context
    print("=" * 70)
    print("PRODUCTION CONTEXT")
    print("=" * 70)
    print("  At 51K tok/s on 8xH100, the Python parser uses <11% of 1 CPU core")
    print("  Even 100x faster parsing saves <0.11% of one core")
    print("  GPU inference is the bottleneck, not CPU parsing")
    print(
        "  DeltaMessage (Pydantic) construction is ~2.1 us/token"
        " (half the per-token cost)"
    )
    print()


if __name__ == "__main__":
    main()
