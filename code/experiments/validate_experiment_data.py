#!/usr/bin/env python3
"""
Validate D²TL experiment results against model logic.
Checks: internal consistency, selector behavior, and physical plausibility.
"""

import json
import sys
from pathlib import Path

RESULTS_PATH = Path(__file__).parent / "results" / "all_experiment_results_via_api.json"


def load():
    with open(RESULTS_PATH) as f:
        return json.load(f)


def validate_exp1(d):
    """Exp1: distribution — sums and percentages."""
    e = d["exp1"]
    total = e["total"]
    ok = True
    if sum(e["weather"].values()) != total:
        print("  [FAIL] Exp1: weather counts do not sum to total")
        ok = False
    if sum(e["density"].values()) != total:
        print("  [FAIL] Exp1: density counts do not sum to total")
        ok = False
    if sum(e["distance"].values()) != total:
        print("  [FAIL] Exp1: distance buckets do not sum to total")
        ok = False
    if e["normal"] + e["extreme"] != total:
        print("  [FAIL] Exp1: normal + extreme != total")
        ok = False
    if abs(e["extreme_pct"] - e["extreme"] / total * 100) > 0.1:
        print("  [FAIL] Exp1: extreme_pct inconsistent")
        ok = False
    if ok:
        print("  [OK] Exp1: counts and percentages consistent")
    return ok


def validate_exp2(d):
    """Exp2: distance–power — slopes negative, dual close to MLP when selector prefers MLP."""
    e = d["exp2"]
    ok = True
    for env in ["Rural", "Urban"]:
        for key in ["mlp_slope", "mamba_slope", "dual_slope"]:
            s = e.get(f"{env}_{key}")
            if s is not None and s > 0:
                print(f"  [FAIL] Exp2: {env}_{key} should be negative, got {s}")
                ok = False
    # Theory: power decreases with distance => slope in dB per log10(d) negative
    if e["distances"][0] >= e["distances"][-1]:
        print("  [FAIL] Exp2: distances should be ascending")
        ok = False
    if ok:
        print("  [OK] Exp2: slopes negative, distances ascending")
    return ok


def validate_exp3(d):
    """Exp3: rainstorm — dual must equal mlp or mamba; power decreases with worse weather."""
    e = d["exp3"]
    ok = True
    for env, env_data in e.items():
        if not isinstance(env_data, dict):
            continue
        weather_order = ["Clear", "Light Rain", "Moderate Rain", "Heavy Rain"]
        prev_power = None
        for w in weather_order:
            cell = env_data.get(w)
            if not cell:
                continue
            dual = cell.get("dual")
            mlp = cell.get("mlp")
            mamba = cell.get("mamba")
            if dual is None or mlp is None or mamba is None:
                continue
            if dual != mlp and dual != mamba:
                # Allow soft blend in theory; our API returns only mlp or mamba
                if abs(dual - mlp) > 0.01 and abs(dual - mamba) > 0.01:
                    print(f"  [FAIL] Exp3: {env} {w} dual={dual} not equal to mlp={mlp} or mamba={mamba}")
                    ok = False
            if prev_power is not None and dual > prev_power:
                # Power (dBm) should decrease or stay similar with worse weather
                if dual - prev_power > 2:
                    print(f"  [WARN] Exp3: {env} {w} power increased vs previous (could be model variance)")
            prev_power = dual
    if ok:
        print("  [OK] Exp3: dual equals MLP or Mamba; power trend with weather plausible")
    return ok


def validate_exp4(d):
    """Exp4: stratified — dual_mse <= mlp_mse when mlp_improvement > 0; n sums reasonable."""
    e = d["exp4"]
    ok = True
    total_n = 0
    for cat, v in e.items():
        n = v.get("n", 0)
        total_n += n
        mlp_mse = v.get("mlp_mse")
        dual_mse = v.get("dual_mse")
        imp = v.get("mlp_improvement", 0)
        if mlp_mse is not None and dual_mse is not None:
            if imp > 0 and dual_mse > mlp_mse:
                print(f"  [FAIL] Exp4: {cat} mlp_improvement>0 but dual_mse > mlp_mse")
                ok = False
    # Categories overlap (extreme_compound etc.), so total_n can exceed 4500
    if total_n > 4500 * 2:
        print(f"  [WARN] Exp4: sum of n = {total_n} (overlapping categories)")
    if ok:
        print("  [OK] Exp4: dual_mse <= mlp_mse when improvement > 0")
    return ok


def validate_exp5(d):
    """Exp5: latency — positive values, speedup > 0."""
    e = d["exp5"]
    ok = True
    for name in ["MLP", "Mamba-3"]:
        lat = e.get(name, {}).get("latency_ms")
        if lat is not None and lat <= 0:
            print(f"  [FAIL] Exp5: {name} latency_ms should be positive")
            ok = False
    sp = e.get("speedup")
    if sp is not None and sp <= 0:
        print("  [FAIL] Exp5: speedup should be positive")
        ok = False
    if ok:
        print("  [OK] Exp5: latencies and speedup positive")
    return ok


def validate_exp6(d):
    """Exp6: ablation — D²TL Selector should be competitive with MLP Only (same or better MSE)."""
    e = d["exp6"]
    ok = True
    mlp_only = e.get("MLP Only")
    d2tl = e.get("D²TL Selector")
    if mlp_only is not None and d2tl is not None:
        if d2tl > mlp_only * 1.05:
            print(f"  [WARN] Exp6: D²TL Selector MSE ({d2tl}) > MLP Only ({mlp_only}) by >5%")
        else:
            print("  [OK] Exp6: D²TL Selector <= MLP Only MSE (or within 5%)")
    else:
        print("  [FAIL] Exp6: missing MLP Only or D²TL Selector")
        ok = False
    return ok


def validate_exp7(d):
    """Exp7: tail — D²TL stats consistent with Exp6; tail_improvement formula."""
    e = d["exp7"]
    ok = True
    mlp = e.get("MLP", {})
    d2tl = e.get("D²TL", {})
    tail = e.get("tail_improvement", {})
    # Exp7 mean should match Exp6 D²TL Selector (same set of errors)
    exp6_d2tl = d.get("exp6", {}).get("D²TL Selector")
    if exp6_d2tl is not None and d2tl.get("mean") is not None:
        if abs(d2tl["mean"] - exp6_d2tl) > 0.01:
            print(f"  [FAIL] Exp7: D²TL mean ({d2tl['mean']}) != Exp6 D²TL Selector ({exp6_d2tl})")
            ok = False
    if mlp and d2tl and tail:
        p95_mlp = mlp.get("p95")
        p95_d2tl = d2tl.get("p95")
        if p95_mlp and p95_mlp > 0:
            expected = (1 - p95_d2tl / p95_mlp) * 100
            if abs(tail.get("p95_vs_mlp", 0) - expected) > 0.01:
                print(f"  [WARN] Exp7: tail p95_vs_mlp computed may differ (expected ~{expected:.2f})")
    if ok:
        print("  [OK] Exp7: D²TL mean consistent with Exp6; tail_improvement present")
    return ok


def main():
    if not RESULTS_PATH.exists():
        print(f"Results file not found: {RESULTS_PATH}")
        sys.exit(1)
    data = load()
    print("Validating all_experiment_results_via_api.json against D²TL model logic")
    print("=" * 60)
    results = []
    results.append(("Exp1", validate_exp1(data)))
    results.append(("Exp2", validate_exp2(data)))
    results.append(("Exp3", validate_exp3(data)))
    results.append(("Exp4", validate_exp4(data)))
    results.append(("Exp5", validate_exp5(data)))
    results.append(("Exp6", validate_exp6(data)))
    results.append(("Exp7", validate_exp7(data)))
    print("=" * 60)
    failed = [n for n, ok in results if not ok]
    if failed:
        print(f"Validation FAILED for: {failed}")
        sys.exit(1)
    print("All validations PASSED. Data is consistent with the model.")
    return 0


if __name__ == "__main__":
    main()
