#!/usr/bin/env python3
"""Thin entry-point: run the rule-based baseline model."""

from __future__ import annotations

import argparse

from apu_sentinel.config import load_config
from apu_sentinel.models.rule_based import RuleBasedModel


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="local", choices=["local", "colab"])
    args = parser.parse_args()

    load_config(args.config)
    model = RuleBasedModel()
    print(f"baseline stub ready with config={args.config}, model={model}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
