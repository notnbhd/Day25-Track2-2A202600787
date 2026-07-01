"""Pricing & purchasing economics — measure in $/1M-token, not $/GPU-hr.

Figures are June-2026 as-of snapshots from the deck's RESEARCH dossier; treat
live prices as fast-moving (re-baseline before each cohort).
"""
from __future__ import annotations


def request_cost(
    input_tok: int,
    output_tok: int,
    price_in_per_m: float,
    price_out_per_m: float,
    cached_in: int = 0,
    cache_discount: float = 0.10,   # Anthropic cached-read ~0.1x (=-90%)
    batch: bool = False,
    batch_discount: float = 0.50,   # Batch API ~ -50%
) -> float:
    """USD cost of a single request. Cached input billed at cache_discount x price."""
    cached_in = min(max(0, cached_in), input_tok)
    uncached_in = input_tok - cached_in
    cost = (
        (uncached_in / 1e6) * price_in_per_m
        + (cached_in / 1e6) * price_in_per_m * cache_discount
        + (output_tok / 1e6) * price_out_per_m
    )
    if batch:
        cost *= batch_discount
    return cost


def dollars_per_million(total_cost_usd: float, total_tokens: int) -> float:
    """Aggregate unit economics: $ per 1,000,000 tokens served."""
    if total_tokens <= 0:
        return 0.0
    return total_cost_usd / (total_tokens / 1e6)


def discount_stack(
    batch: bool = False,
    cache_hit_frac: float = 0.0,
    batch_discount: float = 0.50,
    cache_discount: float = 0.10,
) -> float:
    """Effective fraction of the naive bill after stacking discounts (input-heavy view).

    Discounts MULTIPLY: cache applies to the cached share of input, batch to the
    whole bill. batch + 100% cache-hit -> 0.5 * 0.1 = 0.05 (~95% off).
    """
    cache_mult = cache_hit_frac * cache_discount + (1.0 - cache_hit_frac)
    batch_mult = batch_discount if batch else 1.0
    return cache_mult * batch_mult


def break_even_utilization(discount_frac: float) -> float:
    """Utilization at which a commitment pays off ~= 1 - discount.

    A 45% reserved discount needs ~55% utilization (~13.2h/day) to beat on-demand.
    """
    return max(0.0, min(1.0, 1.0 - discount_frac))


def recommend_tier(hours_per_day: float, interruptible: bool, reserved_discount: float = 0.45) -> str:
    """Pick a purchasing tier from a workload's duty cycle + interruptibility.

    DOCUMENTED simple policy (instructor extension point — swap in your own):
      - interruptible & not 24/7  -> 'spot'      (checkpoint and ride the discount)
      - duty cycle >= break-even  -> 'reserved'  (steady, high utilization)
      - otherwise                 -> 'on_demand' (spiky / low duty)
    """
    duty = max(0.0, hours_per_day) / 24.0
    be = break_even_utilization(reserved_discount)
    if interruptible and hours_per_day < 24:
        return "spot"
    if duty >= be:
        return "reserved"
    return "on_demand"


# --- "Your Turn" Extension 1 support ---------------------------------------
# Per-GPU-type spot interruption rate (per-hour reclaim probability). Scarcer /
# newer / bigger accelerators are reclaimed more aggressively on the spot market.
SPOT_INTERRUPT_RATE = {
    "B200": 0.18, "H200": 0.12, "H100": 0.08, "MI300X": 0.10,
    "A100": 0.05, "A10G": 0.03, "L4": 0.02,
}


def interruption_rate_for(gpu_type: str, default: float = 0.05) -> float:
    """Look up the spot reclaim rate for a GPU type (Extension 1)."""
    return SPOT_INTERRUPT_RATE.get(gpu_type, default)


def recommend_tier_v2(
    hours_per_day: float,
    interruptible: bool,
    gpu_type: str = "H100",
    on_demand_hr: float = 1.0,
    reserved_1yr_hr: float | None = None,
    reserved_3yr_hr: float | None = None,
    spot_hr: float | None = None,
    max_spot_interrupt: float = 0.15,
) -> dict:
    """Interruption-aware, term-aware purchasing policy (Extension 1).

    Improvements over `recommend_tier`:
      1. Chooses the reserved TERM (1yr vs 3yr) from the real break-even of each
         term's discount instead of assuming 3yr always wins.
      2. Uses the per-GPU-type spot interruption rate. If a job is interruptible
         but the accelerator is reclaimed too aggressively (rate > max_spot_interrupt),
         spot rework can eat the discount, so we fall back to a reserved term.

    Returns a dict: {tier, reserved_term, interrupt_rate, reason}.
    """
    duty = max(0.0, hours_per_day) / 24.0
    rate = interruption_rate_for(gpu_type)

    # Effective discounts of each reserved term, from live catalog prices.
    disc_1yr = 1.0 - (reserved_1yr_hr / on_demand_hr) if reserved_1yr_hr else 0.0
    disc_3yr = 1.0 - (reserved_3yr_hr / on_demand_hr) if reserved_3yr_hr else 0.0
    be_1yr = break_even_utilization(disc_1yr)
    be_3yr = break_even_utilization(disc_3yr)

    def _reserved_choice(reason: str) -> dict:
        # 3yr needs higher utilization to amortize; only commit to it above its
        # break-even, else the shorter 1yr term is the safer commitment.
        if duty >= be_3yr and disc_3yr >= disc_1yr:
            return {"tier": "reserved", "reserved_term": "3yr",
                    "interrupt_rate": rate, "reason": reason + f"; duty {duty:.0%} >= 3yr break-even {be_3yr:.0%}"}
        return {"tier": "reserved", "reserved_term": "1yr",
                "interrupt_rate": rate, "reason": reason + f"; duty {duty:.0%} below 3yr break-even {be_3yr:.0%}, 1yr safer"}

    if interruptible and hours_per_day < 24:
        if rate <= max_spot_interrupt:
            return {"tier": "spot", "reserved_term": None, "interrupt_rate": rate,
                    "reason": f"interruptible & spot reclaim {rate:.0%} <= {max_spot_interrupt:.0%} cap"}
        # Too flaky for spot — commit to reserved instead of paying rework.
        return _reserved_choice(f"interruptible but spot reclaim {rate:.0%} too high")

    if duty >= be_1yr:
        return _reserved_choice("steady high-duty workload")
    return {"tier": "on_demand", "reserved_term": None, "interrupt_rate": rate,
            "reason": f"spiky/low duty {duty:.0%} below 1yr break-even {be_1yr:.0%}"}


def cache_is_worth_it(avg_reads: float, write_cost: float = 1.25,
                      read_discount: float = 0.10) -> bool:
    """Whether prompt caching pays off given how many times a cached prefix is re-read.

    Writing to the cache costs ~`write_cost`x a normal input token; each cached
    read costs only `read_discount`x. Break-even reads N* solves:
        write_cost + N*read_discount  <  (N+1)*1   (vs paying full price each time)
    Caching wins once average reads clear that threshold.
    """
    breakeven = (write_cost - 1.0) / (1.0 - read_discount) if read_discount < 1.0 else float("inf")
    return avg_reads > breakeven


def spot_checkpoint_cost(
    job_hours: float,
    spot_hr: float,
    on_demand_hr: float,
    interrupt_rate: float = 0.05,      # per-hour chance (H100 spot ~<5%)
    ckpt_overhead_frac: float = 0.03,  # steady cost of writing checkpoints
    rework_hours_per_interrupt: float = 0.5,
) -> dict:
    """Effective cost of running a checkpointable job on spot vs on-demand.

    Interruptions waste the compute since the last checkpoint (rework); checkpointing
    adds a small steady overhead. Spot still wins for interruptible jobs.
    """
    expected_interrupts = job_hours * interrupt_rate
    rework_hours = expected_interrupts * rework_hours_per_interrupt
    effective_hours = job_hours * (1.0 + ckpt_overhead_frac) + rework_hours
    spot_cost = effective_hours * spot_hr
    on_demand_cost = job_hours * on_demand_hr
    savings_pct = (1.0 - spot_cost / on_demand_cost) * 100.0 if on_demand_cost > 0 else 0.0
    return {
        "spot_effective_hours": round(effective_hours, 2),
        "spot_cost": round(spot_cost, 2),
        "on_demand_cost": round(on_demand_cost, 2),
        "savings_pct": round(savings_pct, 1),
    }
