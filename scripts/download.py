#!/usr/bin/env python3
"""Thin wrapper over apu_sentinel.data.download."""

from __future__ import annotations

import argparse
from pathlib import Path

from apu_sentinel.config import load_config
from apu_sentinel.data.download import download


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="local", choices=["local", "colab"])
    args = parser.parse_args()

    settings = load_config(args.config)
    download(Path(settings.data.raw_dir), settings.data.checksum)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
