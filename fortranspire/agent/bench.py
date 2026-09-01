"""``fortranspire bench`` — pipeline-output benchmark + regression detector.

Closes issue #17.

Genuine runtime perf (GFLOPS/s on a real GPU) needs a GPU runner in
CI — that's expensive to host and outside the scope of an OSS Python
package. This command focuses on the **structural metrics** of the
pipeline output: token cost, generated-code size, compilation timing,
extracted-routine counts. These are deterministic and run on any
laptop, which is exactly what makes them useful as a regression gate
on prompt edits or model changes.

Two modes:

  fortranspire bench output/<kernel>/                   → emit JSON to stdout
  fortranspire bench --compare baseline.json output/X/  → fail if regressed

Regression policy: per-metric tolerance, default ±10 % on the
**structural** numeric metrics (file/routine/pragma counts, generated
bytes, LLM cost). The wall-clock `gfortran_seconds` is reported but not
gated — a few milliseconds of runner jitter must not fail a build (#72).
Legacy line: per-metric tolerance, default ±10 % on numeric
metrics. Override via ``--tolerance 0.15`` for a 15 % bound, or
``--strict`` for ±5 %. Counts that *should* increase (more routines
extracted, more validation steps run) only fail when they decrease
beyond the tolerance.

Exit codes:
  0   metrics within tolerance (or no baseline comparison)
  1   at least one metric regressed
  2   invocation error (missing dir, malformed baseline, …)
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# ── Metric collection ──────────────────────────────────────────────────────

@dataclass
class Metrics:
    """All numeric measurements taken on one output directory."""

    output_root: str

    # Structural counts
    n_files_generated: int = 0
    n_routines_extracted: int = 0
    n_acc_pragmas: int = 0
    fortran_total_bytes: int = 0
    cython_total_bytes: int = 0

    # LLM cost (read from observability trace if available)
    llm_calls: int = 0
    llm_prompt_tokens: int = 0
    llm_completion_tokens: int = 0
    llm_cost_usd: float = 0.0

    # Compilation timing
    gfortran_seconds: float = 0.0
    gfortran_ok: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "output_root":              self.output_root,
            "n_files_generated":        self.n_files_generated,
            "n_routines_extracted":     self.n_routines_extracted,
            "n_acc_pragmas":            self.n_acc_pragmas,
            "fortran_total_bytes":      self.fortran_total_bytes,
            "cython_total_bytes":       self.cython_total_bytes,
            "llm_calls":                self.llm_calls,
            "llm_prompt_tokens":        self.llm_prompt_tokens,
            "llm_completion_tokens":    self.llm_completion_tokens,
            "llm_cost_usd":             round(self.llm_cost_usd, 6),
            "gfortran_seconds":         round(self.gfortran_seconds, 4),
            "gfortran_ok":              self.gfortran_ok,
        }


def _read_traces(output_root: Path) -> dict[str, float | int]:
    """Aggregate observability JSONL traces if present."""
    trace_path = output_root / "traces.jsonl"
    if not trace_path.is_file():
        # Fall back to default location used by `fortranspire.observability`
        trace_path = Path("output/traces.jsonl")
    if not trace_path.is_file():
        return {}

    calls = prompt = completion = 0
    cost = 0.0
    for line in trace_path.read_text(encoding="utf-8").splitlines():
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        calls += 1
        prompt += int(rec.get("prompt_tokens", 0) or 0)
        completion += int(rec.get("completion_tokens", 0) or 0)
        cost += float(rec.get("cost_usd", 0.0) or 0.0)
    return {
        "llm_calls":                calls,
        "llm_prompt_tokens":        prompt,
        "llm_completion_tokens":    completion,
        "llm_cost_usd":             cost,
    }


def _count_acc_pragmas(text: str) -> int:
    return sum(
        1 for line in text.splitlines()
        if line.lstrip().lower().startswith("!$acc")
    )


def _measure_gfortran(fortran_files: list[Path]) -> tuple[float, bool]:
    """Time `gfortran -fsyntax-only` across the generated Fortran sources.

    Returns (elapsed_seconds, ok). When gfortran isn't on PATH, returns
    (0.0, False) — the metric is just unavailable, not a failure.
    """
    gfortran = shutil.which("gfortran")
    if not gfortran or not fortran_files:
        return 0.0, False
    start = time.perf_counter()
    try:
        result = subprocess.run(
            [gfortran, "-fsyntax-only", *[str(f) for f in fortran_files]],
            capture_output=True, text=True, timeout=60, check=False,
        )
        elapsed = time.perf_counter() - start
        return elapsed, result.returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return time.perf_counter() - start, False


def collect_metrics(output_root: Path) -> Metrics:
    """Walk `output_root` and produce one Metrics record."""
    metrics = Metrics(output_root=str(output_root))

    fortran_files: list[Path] = []
    if (output_root / "fortran_gpu").is_dir():
        fortran_files = sorted((output_root / "fortran_gpu").glob("*.f90")) \
                      + sorted((output_root / "fortran_gpu").glob("*.F90"))

    cython_files = []
    if (output_root / "cython").is_dir():
        cython_files = list((output_root / "cython").glob("*"))

    metrics.n_files_generated = len(fortran_files) + len(cython_files)
    metrics.fortran_total_bytes = sum(f.stat().st_size for f in fortran_files if f.is_file())
    metrics.cython_total_bytes  = sum(f.stat().st_size for f in cython_files  if f.is_file())

    # Routine count + pragma count derived from the GPU MODULE
    for f in fortran_files:
        try:
            text = f.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        metrics.n_acc_pragmas += _count_acc_pragmas(text)
        # Crude routine counter — good enough for a regression gate
        import re as _re
        metrics.n_routines_extracted += len(
            _re.findall(r"^\s*(?:subroutine|function)\s+\w+",
                        text, _re.IGNORECASE | _re.MULTILINE)
        )

    # Observability trace rollup
    traces = _read_traces(output_root)
    metrics.llm_calls             = int(traces.get("llm_calls", 0))
    metrics.llm_prompt_tokens     = int(traces.get("llm_prompt_tokens", 0))
    metrics.llm_completion_tokens = int(traces.get("llm_completion_tokens", 0))
    metrics.llm_cost_usd          = float(traces.get("llm_cost_usd", 0.0))

    # Compilation timing
    metrics.gfortran_seconds, metrics.gfortran_ok = _measure_gfortran(fortran_files)

    return metrics


# ── Baseline comparison ───────────────────────────────────────────────────

# Numeric metrics that should NOT decrease beyond `tolerance` (a drop
# means we generated less code, fewer routines, fewer pragmas — probably
# a regression). Cost should not INCREASE beyond tolerance.
_METRICS_NO_DROP = (
    "n_files_generated", "n_routines_extracted", "n_acc_pragmas",
    "fortran_total_bytes",
)
_METRICS_NO_RISE = (
    "llm_calls", "llm_prompt_tokens", "llm_completion_tokens",
    "llm_cost_usd",
)

# `gfortran_seconds` is deliberately NOT gated (issue #72). It is a
# wall-clock measure of compiling a tiny fixture — a handful of
# milliseconds — so scheduler jitter on a shared CI runner routinely moves
# it past a ±10 % bound and fails a build for nothing. A flaky gate teaches
# people to ignore red, which is worse than the metric it protects. It is
# still measured and shown in the report; it just never counts as a
# regression. The structural metrics above are deterministic and are what
# the gate actually guards.


@dataclass
class RegressionResult:
    metric: str
    baseline: float
    current: float
    delta_pct: float
    direction: str       # "drop" | "rise"
    severity: str        # "regression" | "ok"


@dataclass
class CompareReport:
    tolerance: float
    rows: list[RegressionResult] = field(default_factory=list)

    @property
    def has_regressions(self) -> bool:
        return any(r.severity == "regression" for r in self.rows)


def compare(baseline: dict, current: Metrics, *, tolerance: float = 0.10) -> CompareReport:
    """Diff `current` against a saved `baseline` dict."""
    report = CompareReport(tolerance=tolerance)
    cur = current.to_dict()
    for metric, base_value in baseline.items():
        if metric == "output_root":
            continue
        if metric not in cur:
            continue
        cur_value = cur[metric]
        if not isinstance(base_value, (int, float)) or not isinstance(cur_value, (int, float)):
            continue
        # Skip bools (gfortran_ok)
        if isinstance(base_value, bool) or isinstance(cur_value, bool):
            continue
        if base_value == 0:
            # New metric or always-zero baseline → no meaningful diff
            continue

        delta = (cur_value - base_value) / base_value
        is_regression = False
        direction = "drop" if delta < 0 else "rise"
        if metric in _METRICS_NO_DROP and delta < -tolerance:
            is_regression = True
        elif metric in _METRICS_NO_RISE and delta > tolerance:
            is_regression = True

        report.rows.append(RegressionResult(
            metric=metric,
            baseline=float(base_value),
            current=float(cur_value),
            delta_pct=round(delta * 100, 2),
            direction=direction,
            severity="regression" if is_regression else "ok",
        ))
    return report


# ── Rendering ──────────────────────────────────────────────────────────────

def render_text(metrics: Metrics, report: CompareReport | None = None) -> str:
    """Human-readable summary — metrics block + optional regression table."""
    lines: list[str] = []
    lines.append(f"# fortranspire bench — {metrics.output_root}")
    lines.append("")
    lines.append("## Metrics")
    lines.append("")
    for k, v in metrics.to_dict().items():
        if k == "output_root":
            continue
        lines.append(f"  {k:<28} {v}")

    if report is not None:
        lines.append("")
        lines.append(f"## Regression check (tolerance ±{report.tolerance * 100:g} %)")
        lines.append("")
        if not report.rows:
            lines.append("  (no comparable numeric metrics in baseline)")
        else:
            lines.append(f"  {'metric':<28} {'baseline':>12} {'current':>12} {'Δ%':>8}  status")
            for r in report.rows:
                marker = "REGRESS" if r.severity == "regression" else "ok"
                lines.append(
                    f"  {r.metric:<28} {r.baseline:>12.4g} {r.current:>12.4g} "
                    f"{r.delta_pct:>7.2f}%  {marker}"
                )
    return "\n".join(lines)


# ── CLI ────────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="fortranspire bench",
        description=(
            "Pipeline-output benchmark + regression detector. Measures "
            "structural metrics (file count, routine count, pragma count, "
            "generated bytes, compilation time, LLM cost from observability "
            "traces). Use --compare to fail on regression."
        ),
        epilog=(
            "Examples:\n"
            "  fortranspire bench output/kernel/                   # emit JSON\n"
            "  fortranspire bench output/kernel/ -o baseline.json  # save baseline\n"
            "  fortranspire bench --compare baseline.json output/kernel/\n"
            "  fortranspire bench --compare baseline.json --tolerance 0.05 output/kernel/\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("output_root",
                        help="Phase-1 output directory to measure")
    parser.add_argument("--format", choices=("json", "text"), default="text",
                        help="Output format (default: text)")
    parser.add_argument("-o", "--output", default=None,
                        help="Write the metrics report to this file instead of stdout")
    parser.add_argument("--compare", metavar="BASELINE.json", default=None,
                        help="Diff against a previously-saved baseline; "
                             "exit 1 if any metric regresses beyond --tolerance")
    parser.add_argument("--tolerance", type=float, default=0.10,
                        help="Per-metric regression tolerance (default 0.10 = ±10%%)")
    parser.add_argument("--strict", action="store_true",
                        help="Shortcut for --tolerance 0.05")
    args = parser.parse_args(argv)

    root = Path(args.output_root)
    if not root.is_dir():
        print(f"fortranspire bench: not a directory: {root}", file=sys.stderr)
        return 2

    metrics = collect_metrics(root)

    report: CompareReport | None = None
    if args.compare:
        try:
            baseline = json.loads(Path(args.compare).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            print(f"fortranspire bench: cannot read baseline {args.compare!r}: {exc}",
                  file=sys.stderr)
            return 2
        tolerance = 0.05 if args.strict else args.tolerance
        report = compare(baseline, metrics, tolerance=tolerance)

    # Render
    if args.format == "json":
        payload = metrics.to_dict()
        if report is not None:
            payload["regression_check"] = {
                "tolerance": report.tolerance,
                "rows": [
                    {"metric": r.metric, "baseline": r.baseline, "current": r.current,
                     "delta_pct": r.delta_pct, "direction": r.direction,
                     "severity": r.severity}
                    for r in report.rows
                ],
            }
        rendered = json.dumps(payload, indent=2, ensure_ascii=False)
    else:
        rendered = render_text(metrics, report)

    if args.output:
        Path(args.output).write_text(rendered, encoding="utf-8")
        print(f"fortranspire bench: report saved to {args.output}")
    else:
        print(rendered)

    return 1 if (report is not None and report.has_regressions) else 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
