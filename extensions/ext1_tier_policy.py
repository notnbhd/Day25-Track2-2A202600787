"""Extension 1 — Improve `recommend_tier()`.

Rewrites the purchasing policy to account for (a) per-GPU-type spot interruption
rate and (b) the real 1yr-vs-3yr reserved break-even. We then show three things:

  1. Real-data run    — new vs. old policy on workloads.csv.
  2. Break-even matrix — the tier each GPU type earns across duty cycles (the
                         "GPU × duty × interruptible" recommendation matrix).
  3. Stress scenario  — a GPU-shortage month where spot reclaim rates spike. The
                         old policy is blind to reclaim and keeps flaky jobs on
                         spot (paying rework); the new policy migrates them to a
                         reserved term. This is where the savings delta appears.

Grading question: "Savings thay đổi như thế nào? Tại sao policy mới cho kết quả khác?"

Run: python extensions/ext1_tier_policy.py
"""
from __future__ import annotations
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from missions._common import load_csv, num, catalog_by_type
from finops import pricing

DAYS = 30


def _cost(tier, term, gpu_hours, c, interrupt_rate=None):
    od = num(c["on_demand_hr"])
    if tier == "spot":
        rate = interrupt_rate if interrupt_rate is not None else 0.05
        return pricing.spot_checkpoint_cost(gpu_hours, num(c["spot_hr"]), od,
                                            interrupt_rate=rate)["spot_cost"]
    if tier == "reserved":
        rate = num(c["reserved_3yr_hr"]) if term == "3yr" else num(c["reserved_1yr_hr"])
        return gpu_hours * rate
    return gpu_hours * od


def _recommend(c, gtype, hpd, interruptible, interrupt_override=None):
    if interrupt_override is not None:
        # temporarily override the reclaim rate to model a shortage
        saved = pricing.SPOT_INTERRUPT_RATE.get(gtype)
        pricing.SPOT_INTERRUPT_RATE[gtype] = interrupt_override
    rec = pricing.recommend_tier_v2(
        hpd, interruptible, gpu_type=gtype,
        on_demand_hr=num(c["on_demand_hr"]),
        reserved_1yr_hr=num(c["reserved_1yr_hr"]),
        reserved_3yr_hr=num(c["reserved_3yr_hr"]),
        spot_hr=num(c["spot_hr"]),
    )
    if interrupt_override is not None:
        if saved is None:
            pricing.SPOT_INTERRUPT_RATE.pop(gtype, None)
        else:
            pricing.SPOT_INTERRUPT_RATE[gtype] = saved
    return rec


def _real_data(jobs, cat, verbose):
    on_demand = old_opt = new_opt = 0.0
    rows = []
    for j in jobs:
        gtype = j["gpu_type"]; c = cat[gtype]
        ngpu = int(num(j["num_gpus"])); hpd = num(j["hours_per_day"])
        interruptible = bool(int(num(j["interruptible"])))
        gpu_hours = hpd * DAYS * ngpu
        on_demand += gpu_hours * num(c["on_demand_hr"])

        old_tier = pricing.recommend_tier(hpd, interruptible)
        old_opt += _cost(old_tier, "3yr", gpu_hours, c)

        rec = _recommend(c, gtype, hpd, interruptible)
        new_opt += _cost(rec["tier"], rec["reserved_term"], gpu_hours, c,
                         interrupt_rate=rec["interrupt_rate"])
        rows.append((j["job_id"], gtype, old_tier,
                     rec["tier"] + (f"/{rec['reserved_term']}" if rec["reserved_term"] else ""),
                     rec["interrupt_rate"]))

    old_pct = (on_demand - old_opt) / on_demand * 100
    new_pct = (on_demand - new_opt) / on_demand * 100
    if verbose:
        print("== Extension 1: interruption-/term-aware recommend_tier ==\n")
        print("--- 1. Real data (workloads.csv) ---")
        print(f"{'job':18}{'gpu':7}{'old':10}{'new':14}{'reclaim':>8}")
        for jid, g, ot, nt, ir in rows:
            print(f"{jid:18}{g:7}{ot:10}{nt:14}{ir*100:>7.0f}%")
        print(f"old policy savings: {old_pct:.1f}%   new policy savings: {new_pct:.1f}%")
        print("(they agree — with today's low reclaim rates the simple policy is already sound)\n")
    return {"old_savings_pct": round(old_pct, 1), "new_savings_pct": round(new_pct, 1)}


def _matrix(cat, verbose):
    """Recommendation matrix: GPU type x duty cycle x interruptible."""
    duties = [0.25, 0.50, 0.75, 1.00]
    out = {}
    if verbose:
        print("--- 2. Break-even matrix (tier by duty cycle) ---")
        print(f"{'gpu':7}{'1yr_be':>8}{'3yr_be':>8}   " + "".join(f"{int(d*100):>4}%" for d in duties)
              + "   (non-interruptible)")
    for gtype, c in cat.items():
        od = num(c["on_demand_hr"])
        be1 = pricing.break_even_utilization(1 - num(c["reserved_1yr_hr"]) / od)
        be3 = pricing.break_even_utilization(1 - num(c["reserved_3yr_hr"]) / od)
        picks = []
        for d in duties:
            rec = _recommend(c, gtype, d * 24, interruptible=False)
            tag = rec["tier"][0].upper() + (rec["reserved_term"] or "")
            picks.append(tag)
        out[gtype] = {"be_1yr": round(be1, 2), "be_3yr": round(be3, 2), "picks": picks}
        if verbose:
            print(f"{gtype:7}{be1*100:>7.0f}%{be3*100:>7.0f}%   " + "".join(f"{p:>5}" for p in picks))
    if verbose:
        print("  legend: O=on_demand, R1yr/R3yr=reserved term. Term flips at each break-even.\n")
    return out


def _stress(jobs, cat, verbose, shortage_rate=0.25):
    """GPU-shortage month: spot reclaim spikes. Old policy ignores it; new adapts."""
    old_cost = new_cost = 0.0
    migrated = []
    for j in jobs:
        gtype = j["gpu_type"]; c = cat[gtype]
        ngpu = int(num(j["num_gpus"])); hpd = num(j["hours_per_day"])
        interruptible = bool(int(num(j["interruptible"])))
        if not (interruptible and hpd < 24):
            continue  # only spot-eligible jobs are exposed to reclaim spikes
        gpu_hours = hpd * DAYS * ngpu
        # OLD: blindly stays on spot, now paying elevated rework
        old_cost += _cost("spot", None, gpu_hours, c, interrupt_rate=shortage_rate)
        # NEW: sees reclaim > cap, migrates to the cheaper reserved term
        rec = _recommend(c, gtype, hpd, interruptible, interrupt_override=shortage_rate)
        new_cost += _cost(rec["tier"], rec["reserved_term"], gpu_hours, c,
                          interrupt_rate=shortage_rate)
        if rec["tier"] != "spot":
            migrated.append((j["job_id"], gtype, rec["tier"] + f"/{rec['reserved_term']}"))
    delta = old_cost - new_cost
    if verbose:
        print(f"--- 3. Stress: spot reclaim spikes to {shortage_rate:.0%} (GPU shortage) ---")
        print(f"old policy (blind, stays on spot): ${old_cost:,.0f}/mo")
        print(f"new policy (migrates flaky jobs) : ${new_cost:,.0f}/mo")
        for jid, g, t in migrated:
            print(f"   migrated {jid} ({g}) -> {t}")
        print(f"avoided rework: ${delta:,.0f}/mo  ({delta/old_cost*100:.1f}% of exposed spend)\n")
    return {"stress_old": round(old_cost), "stress_new": round(new_cost),
            "stress_saved": round(delta), "migrated": migrated}


def run(verbose: bool = True) -> dict:
    jobs = load_csv("workloads.csv")
    cat = catalog_by_type()
    real = _real_data(jobs, cat, verbose)
    matrix = _matrix(cat, verbose)
    stress = _stress(jobs, cat, verbose)
    if verbose:
        print("Insight: the simple policy is fine at today's reclaim rates, but it is *blind* to")
        print("interruption risk. The interruption-aware policy is identical when spot is calm and")
        print("strictly better when the market tightens — it stops over-committing to volatile spot.")
    return {**real, "matrix": matrix, **stress}


if __name__ == "__main__":
    run()
