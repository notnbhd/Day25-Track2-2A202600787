#!/usr/bin/env python3
"""verify.py — one-command green check for Lab 25 (zero-key, no GPU).

Regenerates the synthetic data, runs all five missions, and asserts the key
FinOps results. Prints a PASS/FAIL table and exits non-zero on any failure.
"""
from __future__ import annotations
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from data import generate
from missions import m1_efficiency_audit, m2_inference_levers, m3_purchasing, m4_allocation, m5_report

CHECKS = []


def check(name, cond, detail=""):
    CHECKS.append((name, bool(cond), detail))


def main() -> int:
    generate.main()  # deterministic regenerate

    r1 = m1_efficiency_audit.run(verbose=False)
    r2 = m2_inference_levers.run(verbose=False)
    r3 = m3_purchasing.run(verbose=False)
    r4 = m4_allocation.run(verbose=False)
    r5 = m5_report.run(verbose=False)

    lie_ids = [l["gpu_id"] for l in r1["lies"]]
    check("M1 flags the GPU-Util lie (gpu-h100-4)", "gpu-h100-4" in lie_ids, str(lie_ids))
    check("M1 detects idle waste", r1["idle_waste_daily"] > 0, f"${r1['idle_waste_daily']}/day")

    check("M2 $/1M-token drops after optimization",
          r2["optimized_per_m"] < r2["baseline_per_m"],
          f"{r2['baseline_per_m']} -> {r2['optimized_per_m']}")
    check("M2 inference savings in 60-95% band", 60 <= r2["savings_pct"] <= 95, f"{r2['savings_pct']}%")

    tiers = {r["tier"] for r in r3["recommendations"]}
    check("M3 recommends a spot tier", "spot" in tiers, str(tiers))
    check("M3 recommends a reserved tier", "reserved" in tiers, str(tiers))
    check("M3 purchasing saves money", r3["savings_pct"] > 0, f"{r3['savings_pct']}%")

    check("M4 tag coverage 85-100%", 0.85 <= r4["tag_coverage"] <= 1.0, f"{r4['tag_coverage']:.0%}")
    check("M4 chargeback gate is open", r4["chargeback_ready"] is True, str(r4["chargeback_ready"]))

    check("M5 total savings in 40-95% band", 40 <= r5["total_savings_pct"] <= 95,
          f"{r5['total_savings_pct']}%")
    check("M5 report.md written", os.path.exists(os.path.join(ROOT, "outputs", "report.md")))

    print("\n" + "=" * 60)
    print("  LAB 25 VERIFY")
    print("=" * 60)
    passed = 0
    for name, ok, detail in CHECKS:
        mark = "PASS" if ok else "FAIL"
        passed += ok
        print(f"  [{mark}] {name}" + (f"  ({detail})" if detail else ""))
    print("-" * 60)
    print(f"  {passed}/{len(CHECKS)} checks passed")
    print("=" * 60)
    return 0 if passed == len(CHECKS) else 1


if __name__ == "__main__":
    sys.exit(main())
