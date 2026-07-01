"""Extension 4 — Reasoning budget.

Reasoning ("extended thinking") traffic is a small slice of requests but an
outsized slice of energy: the deck puts a reasoning query at ~74-86x the energy
of a small-model query because it emits many hidden thinking tokens before the
answer. This script separates $ and Wh for is_reasoning=1 vs 0, shows how skewed
the split is, and estimates the savings from a routing cap.

Grading question: "Reasoning traffic chiếm bao nhiêu % tổng? Tại sao nó tốn ~80x?"

Run: python extensions/ext4_reasoning_budget.py
"""
from __future__ import annotations
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from missions._common import load_csv, num
from finops import pricing, sustainability
from missions.m2_inference_levers import MODEL_PRICES

TARGET_REASONING_FRAC = 0.03  # policy: cap reasoning to ~3% of traffic


def run(verbose: bool = True) -> dict:
    rows = load_csv("token_usage.csv")
    agg = {0: {"n": 0, "cost": 0.0, "wh": 0.0, "tokens": 0},
           1: {"n": 0, "cost": 0.0, "wh": 0.0, "tokens": 0}}
    for r in rows:
        inp, out = int(num(r["input_tokens"])), int(num(r["output_tokens"]))
        cached = int(num(r["cached_input_tokens"]))
        is_batch = bool(int(num(r["is_batch"])))
        is_reason = int(num(r["is_reasoning"]))
        pin, pout = MODEL_PRICES[r["route_tier"]]
        cost = pricing.request_cost(inp, out, pin, pout, cached_in=cached, batch=is_batch)
        wh = sustainability.wh_per_query(inp + out, is_reasoning=bool(is_reason))
        a = agg[is_reason]
        a["n"] += 1; a["cost"] += cost; a["wh"] += wh; a["tokens"] += inp + out

    n = agg[0]["n"] + agg[1]["n"]
    tot_cost = agg[0]["cost"] + agg[1]["cost"]
    tot_wh = agg[0]["wh"] + agg[1]["wh"]
    frac_traffic = agg[1]["n"] / n
    frac_cost = agg[1]["cost"] / tot_cost
    frac_wh = agg[1]["wh"] / tot_wh

    # --- Cap policy: route reasoning only for the hardest tasks (cap to 3%). ---
    # Model: excess reasoning requests fall back to a normal query of the same
    # token count -> lose the 80x energy multiplier and the extra thinking spend.
    excess = max(0, agg[1]["n"] - int(TARGET_REASONING_FRAC * n))
    avg_wh_reason = agg[1]["wh"] / agg[1]["n"]
    avg_wh_plain = avg_wh_reason / sustainability.REASONING_ENERGY_MULTIPLIER
    wh_saved = excess * (avg_wh_reason - avg_wh_plain)
    # $ side: reasoning re-billed as a plain query drops the extra output-heavy
    # thinking tokens; approximate the saving as the mean reasoning-request cost
    # premium over a same-tier plain request.
    avg_cost_reason = agg[1]["cost"] / agg[1]["n"]
    avg_cost_plain = agg[0]["cost"] / agg[0]["n"]
    cost_saved = excess * max(0.0, avg_cost_reason - avg_cost_plain)
    # monthly energy $ at us-east-1
    wh_saved_month = wh_saved  # dataset already ~ one day
    usd_energy_saved = sustainability.energy_cost_usd(wh_saved, "us-east-1")

    if verbose:
        print("== Extension 4: reasoning budget ==\n")
        print(f"{'bucket':12}{'reqs':>7}{'%traffic':>10}{'$/day':>10}{'Wh/day':>12}")
        for k, label in [(0, "plain"), (1, "reasoning")]:
            a = agg[k]
            print(f"{label:12}{a['n']:>7}{a['n']/n*100:>9.1f}%{a['cost']:>10.2f}{a['wh']:>12.1f}")
        print(f"{'TOTAL':12}{n:>7}{'100.0%':>10}{tot_cost:>10.2f}{tot_wh:>12.1f}\n")
        print(f"reasoning = {frac_traffic:.1%} of traffic  ->  {frac_cost:.1%} of $  and  {frac_wh:.1%} of energy")
        print(f"energy concentration factor: {frac_wh/frac_traffic:.1f}x its traffic share")
        print(f"\nWhy ~80x: a reasoning request emits a long hidden chain-of-thought before its")
        print(f"visible answer, so it burns roughly {sustainability.REASONING_ENERGY_MULTIPLIER:.0f}x the tokens (=energy) of a plain query.\n")
        print(f"Routing rule: only escalate to reasoning when task complexity is high; cap at "
              f"{TARGET_REASONING_FRAC:.0%} of traffic.")
        print(f"  excess reasoning reqs to downgrade: {excess}")
        print(f"  energy saved : {wh_saved:,.0f} Wh/day  (~${usd_energy_saved:.2f}/day grid cost)")
        print(f"  spend saved  : ${cost_saved:,.2f}/day  (~${cost_saved*30:,.0f}/month)")

    return {
        "reasoning_frac_traffic": round(frac_traffic, 4),
        "reasoning_frac_cost": round(frac_cost, 4),
        "reasoning_frac_energy": round(frac_wh, 4),
        "energy_concentration_x": round(frac_wh / frac_traffic, 1),
        "excess_reqs": excess,
        "wh_saved_day": round(wh_saved),
        "usd_saved_month": round(cost_saved * 30, 2),
    }


if __name__ == "__main__":
    run()
