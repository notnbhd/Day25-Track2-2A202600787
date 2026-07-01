"""Extension 5 — Carbon-aware scheduling.

Interruptible training jobs don't care *where* they run, so we can move them to
the cleanest (and often cheapest) grid. This script:

  1. Estimates each interruptible job's energy from GPU watts x GPU-hours.
  2. Prices every region on $ (electricity) AND carbon (gCO2e).
  3. Reports the carbon and $ saved by moving all interruptible jobs from the
     default us-east-1 to the cleanest region, and names the optimal region per
     criterion (cheapest $, cleanest CO2, balanced).

Grading question: "Vùng nào là 'tối ưu' thực sự? Phụ thuộc ưu tiên nào của công ty?"

Run: python extensions/ext5_carbon_aware.py
"""
from __future__ import annotations
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from missions._common import load_csv, num, catalog_by_type
from finops import sustainability as S

DAYS = 30
DEFAULT_REGION = "us-east-1"


def _interruptible_energy_kwh(jobs, cat):
    """Total monthly energy (kWh) of the interruptible fleet, from watts x hours."""
    kwh = 0.0
    detail = []
    for j in jobs:
        if not int(num(j["interruptible"])):
            continue
        c = cat[j["gpu_type"]]
        gpu_hours = num(j["hours_per_day"]) * DAYS * int(num(j["num_gpus"]))
        job_kwh = num(c["watts"]) / 1000.0 * gpu_hours
        kwh += job_kwh
        detail.append((j["job_id"], j["gpu_type"], round(job_kwh)))
    return kwh, detail


def run(verbose: bool = True) -> dict:
    jobs = load_csv("workloads.csv")
    cat = catalog_by_type()
    kwh, detail = _interruptible_energy_kwh(jobs, cat)

    # Price every region on $ and carbon for this energy.
    regions = []
    for reg in S.REGION_CARBON:
        wh = kwh * 1000.0
        regions.append({
            "region": reg,
            "usd_kwh": S.REGION_PRICE_KWH.get(reg, 0.12),
            "gco2_kwh": S.REGION_CARBON[reg],
            "elec_usd": S.energy_cost_usd(wh, reg),
            "carbon_kg": S.carbon_g(wh, reg) / 1000.0,
        })

    cheapest = min(regions, key=lambda r: r["elec_usd"])
    cleanest = min(regions, key=lambda r: r["carbon_kg"])
    # "balanced" = min of normalized ($ + carbon) rank sum
    max_usd = max(r["elec_usd"] for r in regions)
    max_co2 = max(r["carbon_kg"] for r in regions)
    balanced = min(regions, key=lambda r: r["elec_usd"] / max_usd + r["carbon_kg"] / max_co2)

    default = next(r for r in regions if r["region"] == DEFAULT_REGION)
    carbon_saved = default["carbon_kg"] - cleanest["carbon_kg"]
    usd_saved = default["elec_usd"] - cleanest["elec_usd"]

    if verbose:
        print("== Extension 5: carbon-aware scheduling ==\n")
        print(f"interruptible fleet energy: {kwh:,.0f} kWh/month")
        for jid, g, k in detail:
            print(f"   {jid:18}{g:7}{k:>8,} kWh")
        print()
        print(f"{'region':16}{'$/kWh':>8}{'gCO2/kWh':>10}{'elec $/mo':>12}{'carbon kg/mo':>14}")
        for r in sorted(regions, key=lambda x: x["carbon_kg"]):
            print(f"{r['region']:16}{r['usd_kwh']:>8.3f}{r['gco2_kwh']:>10}{r['elec_usd']:>12,.0f}{r['carbon_kg']:>14,.0f}")
        print()
        print(f"default region      : {DEFAULT_REGION}")
        print(f"cheapest ($)        : {cheapest['region']}  (${cheapest['elec_usd']:,.0f}/mo)")
        print(f"cleanest (CO2)      : {cleanest['region']}  ({cleanest['carbon_kg']:,.0f} kg/mo)")
        print(f"balanced ($+CO2)    : {balanced['region']}")
        print(f"\nmove interruptible fleet {DEFAULT_REGION} -> {cleanest['region']}:")
        print(f"   carbon saved: {carbon_saved:,.0f} kg CO2e/month  ({carbon_saved/default['carbon_kg']*100:.0f}% cut)")
        print(f"   elec $ saved: ${usd_saved:,.0f}/month")
        print(f"\nTrade-off: the cleanest grid ({cleanest['region']}) is far from most users, so it fits")
        print(f"interruptible *training* (latency-insensitive), not user-facing inference.")

    return {
        "fleet_kwh_month": round(kwh),
        "cheapest_region": cheapest["region"],
        "cleanest_region": cleanest["region"],
        "balanced_region": balanced["region"],
        "carbon_saved_kg_month": round(carbon_saved),
        "usd_saved_month": round(usd_saved, 2),
        "regions": regions,
    }


if __name__ == "__main__":
    run()
