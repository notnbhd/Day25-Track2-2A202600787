"""Tests for the 'Your Turn' extensions (new — original 15 tests untouched)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from finops import pricing
from extensions import ext1_tier_policy, ext4_reasoning_budget, ext5_carbon_aware


# --- Extension 1: interruption-/term-aware recommend_tier ---
def test_recommend_tier_v2_backward_compatible_shape():
    rec = pricing.recommend_tier_v2(2, True, gpu_type="A100",
                                    on_demand_hr=1.79, reserved_1yr_hr=1.4, reserved_3yr_hr=1.0)
    assert rec["tier"] == "spot" and set(rec) >= {"tier", "reserved_term", "interrupt_rate", "reason"}


def test_recommend_tier_v2_high_reclaim_falls_back_to_reserved():
    # B200 has a high spot reclaim rate -> interruptible job should NOT stay on spot
    rec = pricing.recommend_tier_v2(10, True, gpu_type="B200",
                                    on_demand_hr=5.09, reserved_1yr_hr=4.2, reserved_3yr_hr=3.2)
    assert rec["tier"] == "reserved"


def test_recommend_tier_v2_term_selection():
    # 100% duty non-interruptible clears the 3yr break-even -> 3yr term
    rec = pricing.recommend_tier_v2(24, False, gpu_type="H100",
                                    on_demand_hr=2.5, reserved_1yr_hr=2.0, reserved_3yr_hr=1.4)
    assert rec["tier"] == "reserved" and rec["reserved_term"] == "3yr"


def test_ext1_stress_saves_money():
    out = ext1_tier_policy.run(verbose=False)
    assert out["stress_saved"] > 0  # adapting to reclaim spikes avoids rework


# --- Extension 3 helper: cache economics ---
def test_cache_is_worth_it_threshold():
    # write 1.25x, read 0.10x -> break-even ~0.28 reads; 1 read already wins, 0 does not
    assert pricing.cache_is_worth_it(2.0) is True
    assert pricing.cache_is_worth_it(0.0) is False


# --- Extension 4: reasoning budget ---
def test_ext4_reasoning_energy_dominates():
    out = ext4_reasoning_budget.run(verbose=False)
    # small traffic share, but energy share is far larger (concentration > 1)
    assert out["energy_concentration_x"] > 1.0
    assert out["reasoning_frac_energy"] > out["reasoning_frac_traffic"]


# --- Extension 5: carbon-aware scheduling ---
def test_ext5_cleanest_region_and_savings():
    out = ext5_carbon_aware.run(verbose=False)
    assert out["cleanest_region"] == "europe-north1"
    assert out["carbon_saved_kg_month"] > 0
