"""Fetch MetroPT-3 and verify it against the documented checksum.

Stub: download + verification logic implemented in a later pass.
"""

from __future__ import annotations

from pathlib import Path


def download(raw_dir: Path, checksum: str) -> Path:
    """Download MetroPT-3 into raw_dir, verifying against checksum.

    Returns the path to the verified raw data file.
    """
    raise NotImplementedError


def verify_checksum(path: Path, expected_checksum: str) -> bool:
    """Verify path's sha256 matches expected_checksum."""
    raise NotImplementedError
