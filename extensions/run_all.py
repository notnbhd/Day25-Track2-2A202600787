"""Run all completed "Your Turn" extensions in sequence.

Run: python extensions/run_all.py
"""
from __future__ import annotations
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from extensions import ext1_tier_policy, ext4_reasoning_budget, ext5_carbon_aware


def main():
    for mod in (ext1_tier_policy, ext4_reasoning_budget, ext5_carbon_aware):
        mod.run(verbose=True)
        print("\n" + "-" * 70 + "\n")


if __name__ == "__main__":
    main()
