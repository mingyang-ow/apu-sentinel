#!/usr/bin/env python3
"""Thin episode-level evaluation entry-point."""

from __future__ import annotations

import argparse

from apu_sentinel.config import load_config
from apu_sentinel.pipeline import run_pipeline


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="local", choices=["local", "colab"])
    args = parser.parse_args()

    settings = load_config(args.config)
    result = run_pipeline(settings)
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
