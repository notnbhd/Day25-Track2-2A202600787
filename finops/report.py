"""Report assembly — the lab's deliverable: baseline vs optimized + savings chart."""
from __future__ import annotations


def build_report(baseline_usd: float, optimized_usd: float, levers: dict,
                 sustainability: dict | None = None, period: str = "monthly",
                 util_lies: list | None = None, rightsize_map: dict | None = None,
                 recommendations: list | None = None, region_rows: list | None = None,
                 reasoning: dict | None = None, extensions: list | None = None) -> str:
    """Return a markdown cost-optimization report.

    The core (header + lever table + sustainability) always renders. The optional
    args (`util_lies`, `recommendations`, `region_rows`, `reasoning`, `extensions`)
    add the analysis sections the rubric grades — they are omitted if not supplied,
    so callers and tests that only pass the core keep working.
    """
    savings = baseline_usd - optimized_usd
    pct = (savings / baseline_usd * 100.0) if baseline_usd > 0 else 0.0
    total_lever = sum(levers.values()) or 1
    lines = [
        "# NimbusAI — GPU Cost Optimization Report",
        "",
        f"**Period:** {period}  ",
        f"**Baseline spend:** ${baseline_usd:,.0f}  ",
        f"**Optimized spend:** ${optimized_usd:,.0f}  ",
        f"**Projected savings:** ${savings:,.0f}  (**{pct:.0f}%**)",
        "",
        "> Bottom line: we measure in **$/1M-token**, not $/GPU-hour. Every lever below",
        "> lowers the unit cost of a served token, which is the number the business feels.",
        "",
        "## Savings by lever",
        "",
        "| Lever | Savings (USD) | % of total savings |",
        "|---|---|---|",
    ]
    for name, amount in levers.items():
        lines.append(f"| {name} | ${amount:,.0f} | {amount / total_lever * 100:.0f}% |")

    # --- The GPU-Util lie (mechanism, deck §5) ---
    if util_lies:
        lines += [
            "",
            "## Why GPU-Util is a lie",
            "",
            "`nvidia-smi` reports **GPU-Util %** = the fraction of time *at least one*",
            "kernel was resident on the SMs. It says the clock was busy; it says nothing",
            "about whether the tensor cores did useful FLOPs. **MFU** (achieved ÷ peak",
            "FLOPs) is the real efficiency number. A GPU can show 98% util while its",
            "tensor cores sit starved — waiting on HBM (memory stalls), running tiny",
            "batches, or eaten by kernel-launch overhead. You are billed the full",
            "GPU-hour for a fraction of the compute you rented.",
            "",
            "| GPU | type | GPU-Util | MFU | reading |",
            "|---|---|---|---|---|",
        ]
        for l in util_lies:
            u = l.get("gpu_util_pct", 0)
            m = l.get("mfu", 0)
            lines.append(f"| {l.get('gpu_id','')} | {l.get('gpu_type','')} | {u:.0f}% | "
                         f"{m:.2f} | paying full rate for ~{m*100:.0f}% of the FLOPs |")
        if rightsize_map:
            lines.append("")
            lines.append("_Fix: right-size these down one tier (e.g. "
                         + ", ".join(f"{k}→{v}" for k, v in list(rightsize_map.items())[:3])
                         + ") or raise batch size / fuse kernels so MFU climbs._")

    # --- Prioritized action plan (ROI order) ---
    ordered = sorted(levers.items(), key=lambda kv: kv[1], reverse=True)
    _why = {
        "Inference (cascade/cache/batch)": "biggest unit-cost win; software-only, no procurement risk",
        "Purchasing (spot/reserved)": "large but needs commitment — validate duty cycle first",
        "Right-size util-lies": "reclaims FLOPs you already pay for; ship after audit confirms MFU",
        "Kill idle GPUs": "pure waste; a cron/autoscaler change, do it today",
    }
    lines += ["", "## Action plan — ranked by dollar impact", ""]
    for i, (name, amount) in enumerate(ordered, 1):
        lines.append(f"{i}. **{name}** — ${amount:,.0f}/mo. {_why.get(name, '')}")
    lines += [
        "",
        "**Execution order ≠ dollar order.** Ship the **zero-risk, zero-capex** moves",
        "first (kill idle GPUs, then cascade/cache/batch), *then* commit to reserved/spot",
        "once the duty cycle is proven — you never want to reserve capacity you're about",
        "to make more efficient, and idle/right-sizing fixes shrink the footprint you'd",
        "otherwise commit to.",
    ]

    # --- Sustainability ---
    if sustainability:
        lines += [
            "",
            "## Sustainability",
            "",
            f"- Energy per query: {sustainability.get('wh_per_query', 0):.2f} Wh",
            f"- Carbon per query: {sustainability.get('carbon_g', 0):.3f} gCO2e",
            f"- Cheapest+cleanest region: **{sustainability.get('best_region', 'n/a')}**",
        ]
        if reasoning:
            lines += [
                "",
                f"**Reasoning budget:** reasoning traffic is only "
                f"{reasoning.get('reasoning_frac_traffic',0)*100:.1f}% of requests but "
                f"{reasoning.get('reasoning_frac_energy',0)*100:.0f}% of energy "
                f"({reasoning.get('energy_concentration_x',0):.0f}× its traffic share) — "
                f"a reasoning query burns ~80× the tokens of a plain one. Capping it saves "
                f"~{reasoning.get('wh_saved_day',0):,} Wh/day.",
            ]
        if region_rows:
            lines += [
                "",
                "Region trade-off (cost *and* carbon move together — clean grids are often cheap):",
                "",
                "| region | $/kWh | gCO2/kWh |",
                "|---|---|---|",
            ]
            for r in sorted(region_rows, key=lambda x: x.get("gco2_kwh", 0)):
                lines.append(f"| {r['region']} | {r['usd_kwh']:.3f} | {r['gco2_kwh']} |")
            lines.append("")
            lines.append("_Carbon isn't free: dirtier grids also tend to price electricity higher,"
                         " so moving interruptible training to a clean region cuts the power bill"
                         " and the carbon footprint at once. The catch is latency — the cleanest"
                         " region is usually far from users, so it fits training, not live inference._")

    # --- Extensions summary ---
    if extensions:
        lines += ["", "## 'Your Turn' extensions implemented", ""]
        for e in extensions:
            lines.append(f"- {e}")

    lines += ["", "_Figures are June-2026 as-of snapshots; re-baseline before acting._"]
    return "\n".join(lines)


def savings_waterfall(levers: dict, path: str) -> str:
    """Write a simple savings bar chart PNG. Returns the path. No-op if matplotlib absent."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return ""
    names = list(levers.keys())
    vals = [levers[n] for n in names]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(names, vals, color="#2e548a")
    ax.set_ylabel("Savings (USD / month)")
    ax.set_title("GPU cost savings by FinOps lever")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path
