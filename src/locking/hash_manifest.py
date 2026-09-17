"""Hash utilities for manifest and file integrity."""
from __future__ import annotations

import hashlib
import csv
from pathlib import Path
from typing import Dict


def hash_file(path: Path) -> str:
    """Compute SHA256 hash of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


def hash_directory(path: Path) -> Dict[str, str]:
    """Compute SHA256 hash of every file in a directory."""
    hashes = {}
    for p in sorted(path.rglob("*")):
        if p.is_file() and not p.name.startswith("."):
            hashes[str(p.relative_to(path))] = hash_file(p)
    return hashes


def hash_csv_manifest(path: Path) -> str:
    """Compute a canonical hash of a CSV manifest (order-independent rows)."""
    with open(path) as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = sorted(tuple(r) for r in reader)
    import io
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(header)
    for r in rows:
        writer.writerow(r)
    return hashlib.sha256(buf.getvalue().encode()).hexdigest()
