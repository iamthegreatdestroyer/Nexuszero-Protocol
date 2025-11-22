"""Aggregate Criterion benchmark estimates into unified JSON & Markdown.

Usage:
    python scripts/parse_benchmarks.py \
        --criterion-dir target/criterion \
        --output-dir . \
        [--baseline benchmark_summary.json] \
        [--regression-threshold 0.10]

Outputs:
    benchmark_summary.json
    benchmark_summary.md

Logic:
  - Walk criterion directory; each benchmark has: <bench>/new/estimates.json
  - Extract mean.point_estimate & std_dev.point_estimate (nanoseconds)
  - Convert ns -> microseconds (us) & milliseconds (ms) for readability
  - Compare against optional baseline file to compute relative change
  - Flag regressions above threshold
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Any


def find_estimate_files(root: Path) -> List[Path]:
    files = []
    if not root.exists():
        return files
    for bench_dir in root.iterdir():
        est = bench_dir / "new" / "estimates.json"
        if est.is_file():
            files.append(est)
    return sorted(files)


def load_baseline(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        # Expect structure: {"benchmarks": [{"name":..., "mean_ns":...}, ...]}
        baseline_map = {
            b["name"]: b for b in data.get("benchmarks", []) if "name" in b
        }
        return baseline_map
    except Exception:
        return {}


def parse_estimate_file(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    bench_name = path.parent.parent.name  # <bench>/new/estimates.json
    # Criterion stores point_estimates in nanoseconds
    mean_ns = float(raw["mean"]["point_estimate"])  # ns
    std_ns = (
        float(raw["std_dev"]["point_estimate"]) if raw.get("std_dev") else 0.0
    )
    mean_us = mean_ns / 1_000.0
    mean_ms = mean_ns / 1_000_000.0
    std_us = std_ns / 1_000.0
    std_ms = std_ns / 1_000_000.0
    return {
        "name": bench_name,
        "mean_ns": mean_ns,
        "mean_us": mean_us,
        "mean_ms": mean_ms,
        "std_ns": std_ns,
        "std_us": std_us,
        "std_ms": std_ms,
    }


def format_markdown(
    results: List[Dict[str, Any]], regression_threshold: float
) -> str:
    header = (
        "# Benchmark Summary\n\n"
        "| Benchmark | Mean (µs) | StdDev (µs) | Mean (ms) | Δ% | Regr |\n"
        "|-----------|-----------:|------------:|----------:|----:|------|\n"
    )
    rows = []
    for r in results:
        delta_pct = r.get("relative_change_pct")
        delta_str = (
            f"{delta_pct:+.2f}%" if delta_pct is not None else "-"
        )
        reg = (
            "YES" if (
                delta_pct is not None
                and delta_pct > regression_threshold * 100
            ) else ""
        )  # indicates slowdown
        rows.append(
            f"| {r['name']} | {r['mean_us']:.2f} | {r['std_us']:.2f} | "
            f"{r['mean_ms']:.4f} | {delta_str} | {reg} |"
        )
    return header + "\n".join(rows) + "\n"


def main():
    ap = argparse.ArgumentParser(
        description="Aggregate Criterion benchmark results."
    )
    ap.add_argument(
        "--criterion-dir",
        default="target/criterion",
        help="Path to criterion output root",
    )
    ap.add_argument(
        "--output-dir",
        default=".",
        help="Directory for summary outputs",
    )
    ap.add_argument(
        "--baseline",
        default="benchmark_summary.json",
        help="Optional baseline summary for diff",
    )
    ap.add_argument(
        "--regression-threshold",
        type=float,
        default=0.10,
        help="Slowdown percent threshold (e.g. 0.10 = 10%)",
    )
    args = ap.parse_args()

    criterion_root = Path(args.criterion_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    baseline_map = load_baseline(Path(args.baseline))
    files = find_estimate_files(criterion_root)
    if not files:
        print(f"No estimate files found under {criterion_root}")
        return

    parsed = [parse_estimate_file(p) for p in files]

    # Compute relative changes vs baseline
    for item in parsed:
        base = baseline_map.get(item["name"])
        if base and "mean_ns" in base:
            prev = float(base["mean_ns"])
            if prev > 0:
                rel = (item["mean_ns"] - prev) / prev * 100.0
                item["relative_change_pct"] = rel
        else:
            item["relative_change_pct"] = None

    # Compose JSON structure
    summary = {
        "regression_threshold_pct": args.regression_threshold * 100,
        "benchmarks": parsed,
    }

    json_path = output_dir / "benchmark_summary.json"
    md_path = output_dir / "benchmark_summary.md"
    with json_path.open("w", encoding="utf-8") as jf:
        json.dump(summary, jf, indent=2)
    with md_path.open("w", encoding="utf-8") as mf:
        mf.write(format_markdown(parsed, args.regression_threshold))

    print(f"Wrote {json_path} and {md_path}")


if __name__ == "__main__":
    main()
