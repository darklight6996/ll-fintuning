"""
eval/compare_commits.py

Benchmark comparison tool for the Autonomous AI Security Assessment System.

Usage:
    python eval/compare_commits.py --baseline eval/results/benchmark_A.json --current eval/results/benchmark_B.json

Compares two benchmark JSON result files and outputs a delta table showing:
- Discovery Rate delta (positive = improvement)
- Evidence Chain Integrity delta (positive = improvement)
- Safety Violations delta (negative = improvement)
- Per-scenario changes

Exit code:
    0 = current >= baseline on all key metrics (improvement or neutral)
    1 = regression detected on any key metric
"""

import sys
import json
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional


def load_result(path: str) -> Dict[str, Any]:
    """Load a benchmark JSON result file."""
    p = Path(path)
    if not p.exists():
        print(f"[ERROR] File not found: {path}")
        sys.exit(2)
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def compare_results(baseline: Dict[str, Any], current: Dict[str, Any]) -> Dict[str, Any]:
    """Compute delta metrics between two benchmark result dictionaries."""
    b_dr = baseline.get("discovery_rate", 0.0)
    c_dr = current.get("discovery_rate", 0.0)

    b_ci = baseline.get("evidence_chain_integrity", 0.0)
    c_ci = current.get("evidence_chain_integrity", 0.0)

    b_sv = baseline.get("safety_violations", 0)
    c_sv = current.get("safety_violations", 0)

    b_total = baseline.get("total_scenarios", 0)
    c_total = current.get("total_scenarios", 0)

    dr_delta = round(c_dr - b_dr, 2)
    ci_delta = round(c_ci - b_ci, 2)
    sv_delta = c_sv - b_sv

    # Per-scenario comparison (match by scenario_id)
    b_map: Dict[str, Dict] = {r["scenario_id"]: r for r in baseline.get("results", [])}
    c_map: Dict[str, Dict] = {r["scenario_id"]: r for r in current.get("results", [])}

    all_ids = sorted(set(list(b_map.keys()) + list(c_map.keys())))
    per_scenario: List[Dict[str, Any]] = []
    for sid in all_ids:
        b_r = b_map.get(sid)
        c_r = c_map.get(sid)
        if b_r is None:
            change = "NEW"
        elif c_r is None:
            change = "REMOVED"
        else:
            if c_r["discovered"] and not b_r["discovered"]:
                change = "IMPROVED"
            elif not c_r["discovered"] and b_r["discovered"]:
                change = "REGRESSED"
            elif c_r["evidence_chain_complete"] and not b_r["evidence_chain_complete"]:
                change = "CHAIN_IMPROVED"
            elif not c_r["evidence_chain_complete"] and b_r["evidence_chain_complete"]:
                change = "CHAIN_REGRESSED"
            else:
                change = "UNCHANGED"
        per_scenario.append({
            "scenario_id": sid,
            "baseline_discovered": b_r["discovered"] if b_r else None,
            "current_discovered": c_r["discovered"] if c_r else None,
            "change": change,
        })

    regression = (dr_delta < 0) or (ci_delta < 0) or (sv_delta > 0)

    return {
        "baseline_scenarios": b_total,
        "current_scenarios": c_total,
        "discovery_rate": {"baseline": b_dr, "current": c_dr, "delta": dr_delta},
        "evidence_chain_integrity": {"baseline": b_ci, "current": c_ci, "delta": ci_delta},
        "safety_violations": {"baseline": b_sv, "current": c_sv, "delta": sv_delta},
        "regression_detected": regression,
        "per_scenario": per_scenario,
    }


def print_comparison(comp: Dict[str, Any], baseline_path: str, current_path: str) -> None:
    """Pretty-print the comparison table."""
    sep = "=" * 70
    print(f"\n{sep}")
    print("  BENCHMARK COMPARISON REPORT")
    print(sep)
    print(f"  Baseline : {baseline_path}")
    print(f"  Current  : {current_path}")
    print(f"  Scenarios: {comp['baseline_scenarios']} -> {comp['current_scenarios']}")
    print(f"\n{'Metric':<35} {'Baseline':>10} {'Current':>10} {'Delta':>10}")
    print("-" * 70)

    def fmt_delta(v, better_positive=True):
        if v == 0:
            return "  0.00"
        sign = "+" if v > 0 else ""
        color = ""
        if better_positive:
            color = "(+)" if v > 0 else "(-)"
        else:
            color = "(-)" if v > 0 else "(+)"
        return f"{sign}{v} {color}"

    dr = comp["discovery_rate"]
    ci = comp["evidence_chain_integrity"]
    sv = comp["safety_violations"]

    print(f"  {'Discovery Rate (%)':<33} {dr['baseline']:>10.2f} {dr['current']:>10.2f} {fmt_delta(dr['delta']):>14}")
    print(f"  {'Evidence Chain Integrity (%)':<33} {ci['baseline']:>10.2f} {ci['current']:>10.2f} {fmt_delta(ci['delta']):>14}")
    print(f"  {'Safety Violations':<33} {sv['baseline']:>10} {sv['current']:>10} {fmt_delta(sv['delta'], better_positive=False):>14}")

    print(f"\n  Per-Scenario Changes:")
    print(f"  {'Scenario ID':<45} {'Change':<20}")
    print("  " + "-" * 65)
    for sc in comp["per_scenario"]:
        status_icon = {
            "IMPROVED": "[+]",
            "REGRESSED": "[!]",
            "CHAIN_IMPROVED": "[+]",
            "CHAIN_REGRESSED": "[!]",
            "UNCHANGED": "[ ]",
            "NEW": "[N]",
            "REMOVED": "[R]",
        }.get(sc["change"], "   ")
        print(f"  {status_icon} {sc['scenario_id']:<43} {sc['change']:<20}")

    print(f"\n  Overall Result: {'REGRESSION DETECTED' if comp['regression_detected'] else 'OK (no regression)'}")
    print(sep + "\n")


def latest_two_benchmarks() -> Optional[tuple]:
    """Auto-detect the two most recent benchmark result files."""
    results_dir = Path(__file__).resolve().parent / "results"
    files = sorted(results_dir.glob("benchmark_*.json"), key=lambda p: p.stat().st_mtime)
    if len(files) >= 2:
        return str(files[-2]), str(files[-1])
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Compare two benchmark result JSON files and detect regressions."
    )
    parser.add_argument(
        "--baseline",
        default=None,
        help="Path to baseline benchmark JSON (default: second-newest in eval/results/)",
    )
    parser.add_argument(
        "--current",
        default=None,
        help="Path to current benchmark JSON (default: newest in eval/results/)",
    )
    parser.add_argument(
        "--fail-on-regression",
        action="store_true",
        default=True,
        help="Exit with code 1 if regression is detected (default: True)",
    )
    args = parser.parse_args()

    baseline_path = args.baseline
    current_path = args.current

    if baseline_path is None or current_path is None:
        auto = latest_two_benchmarks()
        if auto is None:
            print("[ERROR] Need at least 2 benchmark result files in eval/results/. "
                  "Run eval/run_benchmark.py first.")
            sys.exit(2)
        baseline_path = baseline_path or auto[0]
        current_path = current_path or auto[1]
        print(f"[*] Auto-detected:\n    baseline={baseline_path}\n    current ={current_path}")

    baseline = load_result(baseline_path)
    current = load_result(current_path)

    comp = compare_results(baseline, current)
    print_comparison(comp, baseline_path, current_path)

    if args.fail_on_regression and comp["regression_detected"]:
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()

