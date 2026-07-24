#!/usr/bin/env python3
"""Thin wrapper over apu_sentinel.data.download.

Establish-then-pin checksum flow (see data/download.py docstring):
    1. uv run python scripts/download.py         # establish mode (checksum: null)
       -> copy the printed "Computed SHA256: ..." hash into configs/base.yaml
    2. uv run python scripts/download.py         # now runs in verify mode

Note: relies on apu_sentinel.config.load_config, which is a stub until its
own (later) pass implements the base/local/colab merge -- this wrapper is
wired for that once it lands.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from apu_sentinel.config import load_config
from apu_sentinel.data.download import download


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="local", choices=["local", "colab"])
    parser.add_argument(
        "--force", action="store_true", help="re-download even if a raw file already exists"
    )
    args = parser.parse_args()

    settings = load_config(args.config)
    download(
        raw_dir=Path(settings.data.raw_dir),
        url=settings.data.url,
        raw_filename=settings.data.raw_filename,
        checksum=settings.data.checksum,
        force=args.force,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
