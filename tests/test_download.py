"""Tests for the two-mode checksum download flow (data-brief.md Build Pass 2).

Uses small local temp files via file:// URLs -- never downloads the real
MetroPT-3 dataset.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from apu_sentinel.data.download import ChecksumMismatchError, download, verify_checksum


@pytest.fixture
def small_file(tmp_path: Path) -> Path:
    path = tmp_path / "sample.bin"
    path.write_bytes(b"metropt-3 synthetic payload for checksum tests")
    return path


def test_verify_checksum_rejects_mismatch(small_file: Path):
    """The crown test of this pass: verify mode must reject a mismatch."""
    wrong_hash = "0" * 64
    with pytest.raises(ChecksumMismatchError):
        verify_checksum(small_file, wrong_hash)


def test_verify_checksum_accepts_match(small_file: Path):
    correct_hash = hashlib.sha256(small_file.read_bytes()).hexdigest()
    assert verify_checksum(small_file, correct_hash) == correct_hash


@pytest.fixture
def source_csv(tmp_path: Path) -> Path:
    source = tmp_path / "source" / "sample.csv"
    source.parent.mkdir()
    source.write_bytes(b"timestamp,TP2\n2020-01-01,1.0\n")
    return source


def test_download_establish_mode_computes_hash_without_raising(
    tmp_path: Path, source_csv: Path, capsys
):
    raw_dir = tmp_path / "raw"
    result = download(
        raw_dir=raw_dir,
        url=source_csv.as_uri(),
        raw_filename="sample.csv",
        checksum=None,
    )

    assert result == raw_dir / "sample.csv"
    assert result.exists()
    out = capsys.readouterr().out
    assert "Computed SHA256:" in out


def test_download_verify_mode_accepts_match(tmp_path: Path, source_csv: Path):
    raw_dir = tmp_path / "raw"
    correct_hash = hashlib.sha256(source_csv.read_bytes()).hexdigest()
    result = download(
        raw_dir=raw_dir,
        url=source_csv.as_uri(),
        raw_filename="sample.csv",
        checksum=correct_hash,
    )
    assert result.exists()


def test_download_verify_mode_rejects_mismatch(tmp_path: Path, source_csv: Path):
    raw_dir = tmp_path / "raw"
    with pytest.raises(ChecksumMismatchError):
        download(
            raw_dir=raw_dir,
            url=source_csv.as_uri(),
            raw_filename="sample.csv",
            checksum="0" * 64,
        )


def test_download_is_idempotent_reuses_existing_file(tmp_path: Path, source_csv: Path):
    raw_dir = tmp_path / "raw"
    original_bytes = source_csv.read_bytes()

    first = download(
        raw_dir=raw_dir, url=source_csv.as_uri(), raw_filename="sample.csv", checksum=None
    )

    # Mutate the source after the first download; a non-forced re-run must
    # NOT re-fetch (idempotent), so the raw copy stays the original bytes.
    source_csv.write_bytes(b"different content, should not be picked up")
    second = download(
        raw_dir=raw_dir, url=source_csv.as_uri(), raw_filename="sample.csv", checksum=None
    )

    assert first == second
    assert (raw_dir / "sample.csv").read_bytes() == original_bytes
