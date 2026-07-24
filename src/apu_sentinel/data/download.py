"""Fetch MetroPT-3 into data/raw/ and verify its integrity by SHA256.

Two-mode checksum flow, driven by config `data.checksum` (see CLAUDE.md /
data-brief.md "Build Pass 2"):

  - ``checksum is None`` (establish mode): download (or reuse an existing
    file), compute its SHA256, PRINT it clearly, and do NOT raise -- there is
    nothing to compare against yet.
  - ``checksum`` is a string (verify mode): download (or reuse an existing
    file), compute its SHA256, and raise ChecksumMismatchError if it does not
    match.

Establish-then-pin usage:
    1. ``uv run python scripts/download.py``  (with `data.checksum: null` in
       configs/base.yaml) -- copy the printed "Computed SHA256: ..." hash
       into configs/base.yaml (`data.checksum`).
    2. Re-run -- now in verify mode, failing loudly on any corruption.

MetroPT-3 ships from the UCI ML Repository as a zip archive. The checksum is
computed over the downloaded archive itself (the transferred artifact); the
archive is then extracted so `raw_filename` (the CSV `load.py` reads) is
available directly in `raw_dir`. If the configured URL already points at a
plain (non-zip) file whose name matches `raw_filename`, no extraction step
is needed.
"""

from __future__ import annotations

import hashlib
import shutil
import urllib.request
import zipfile
from pathlib import Path

CHUNK_SIZE = 1024 * 1024  # 1 MiB -- stream the file, never load it whole into memory
SMALL_FILE_WARNING_BYTES = 10_000  # a few KB likely means an error page, not real data


class ChecksumMismatchError(RuntimeError):
    """Raised in verify mode when a file's SHA256 doesn't match the expected value."""


def _sha256sum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fetch(url: str, dest: Path) -> None:
    with urllib.request.urlopen(url) as response, dest.open("wb") as out:  # noqa: S310
        shutil.copyfileobj(response, out, length=CHUNK_SIZE)


def verify_checksum(path: Path, expected_checksum: str) -> str:
    """Compute path's SHA256 and raise ChecksumMismatchError if it doesn't
    match expected_checksum. Returns the computed hash on success.
    """
    path = Path(path)
    computed = _sha256sum(path)
    if computed != expected_checksum:
        raise ChecksumMismatchError(
            f"SHA256 mismatch for {path}: expected {expected_checksum}, got {computed}"
        )
    return computed


def download(
    raw_dir: Path,
    url: str,
    raw_filename: str,
    checksum: str | None,
    force: bool = False,
) -> Path:
    """Download MetroPT-3 into raw_dir, then establish or verify its checksum.

    Idempotent: if the archive already exists in raw_dir, it is verified
    rather than re-downloaded. Pass force=True to re-download and re-extract.

    Returns the path to the raw CSV (raw_dir / raw_filename), ready for
    data/load.py. Does not parse, split, scale, or window anything.
    """
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)

    archive_name = Path(url).name
    archive_path = raw_dir / archive_name
    data_path = raw_dir / raw_filename

    if force or not archive_path.exists():
        print(f"Downloading {url} -> {archive_path}")
        _fetch(url, archive_path)

    size_bytes = archive_path.stat().st_size
    print(f"Downloaded file size: {size_bytes:,} bytes")
    if size_bytes < SMALL_FILE_WARNING_BYTES:
        print(
            f"WARNING: {archive_path} is only {size_bytes} bytes -- "
            "this looks like an error page, not real data."
        )

    computed = _sha256sum(archive_path)

    if checksum is None:
        print(f"Computed SHA256: {computed}")
        print(
            "Establish mode: no checksum configured yet -- copy the hash above "
            "into configs/base.yaml (data.checksum) to switch to verify mode."
        )
    else:
        if computed != checksum:
            raise ChecksumMismatchError(
                f"SHA256 mismatch for {archive_path}: expected {checksum}, got {computed}"
            )
        print(f"Checksum verified: {computed}")

    if data_path != archive_path and (force or not data_path.exists()):
        if archive_path.suffix.lower() == ".zip":
            with zipfile.ZipFile(archive_path) as zf:
                zf.extract(raw_filename, path=raw_dir)
        else:
            raise RuntimeError(
                f"Cannot locate '{raw_filename}' from downloaded '{archive_path.name}' "
                "(only .zip archives are auto-extracted)."
            )

    return data_path
