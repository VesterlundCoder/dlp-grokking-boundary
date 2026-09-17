"""Create prediction locks for the prospective prediction protocol.

P0 lock: created at t=0 (before any training).
P1 lock: created at probe_steps (after 1% of training, before resuming).

Locks are append-only. Never overwrite.
"""
from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from .hash_manifest import hash_file, hash_csv_manifest


def _get_git_sha() -> str:
    """Get the current git commit SHA."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True,
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


def _get_git_diff_hash() -> str:
    """Hash of uncommitted changes (to detect if working tree is dirty)."""
    try:
        result = subprocess.run(
            ["git", "diff", "--stat"],
            capture_output=True, text=True, check=True,
        )
        import hashlib
        return hashlib.sha256(result.stdout.encode()).hexdigest()
    except Exception:
        return "unknown"


def create_prediction_lock(
    phase: str,                    # e.g., "CONF_A"
    predictor_version: str,        # e.g., "P0_v1"
    predictions_csv: Path,         # predictions file
    run_manifest: Path,            # run manifest CSV
    group_manifest: Path,           # groups.csv
    latent_manifests: List[Path],   # latent manifest files
    encoding_manifests: List[Path], # encoding manifest files
    predictor_source_dir: Path,    # src/predictors/ directory
    feature_table: Optional[Path] = None,  # pretraining features
    predictor_coefficients: Optional[Path] = None,
    locks_dir: Path = Path("locks"),
    extra_files: Optional[Dict[str, Path]] = None,
) -> Path:
    """Create a P0 prediction lock.

    The lock must be created BEFORE any training starts for the target runs.
    """
    locks_dir.mkdir(parents=True, exist_ok=True)

    now_utc = datetime.now(timezone.utc).isoformat()
    now_local = datetime.now().isoformat()
    git_sha = _get_git_sha()
    git_diff_hash = _get_git_diff_hash()

    lock = {
        "phase": phase,
        "predictor_version": predictor_version,
        "lock_type": "P0_pretraining",
        "utc_timestamp": now_utc,
        "local_timestamp": now_local,
        "git_sha": git_sha,
        "git_diff_hash": git_diff_hash,
        "hashes": {
            "predictions_csv": hash_file(predictions_csv),
            "run_manifest": hash_csv_manifest(run_manifest),
            "group_manifest": hash_csv_manifest(group_manifest),
            "latent_manifests": {
                str(p): hash_csv_manifest(p) for p in latent_manifests
            },
            "encoding_manifests": {
                str(p): hash_file(p) for p in encoding_manifests
            },
            "predictor_source": {
                str(p.relative_to(predictor_source_dir)): hash_file(p)
                for p in sorted(predictor_source_dir.rglob("*.py"))
                if p.is_file()
            },
        },
    }

    if feature_table is not None:
        lock["hashes"]["feature_table"] = hash_file(feature_table)
    if predictor_coefficients is not None:
        lock["hashes"]["predictor_coefficients"] = hash_file(predictor_coefficients)
    if extra_files is not None:
        lock["hashes"]["extra"] = {
            str(k): hash_file(v) for k, v in extra_files.items()
        }

    lock_path = locks_dir / f"{phase}_{predictor_version}_P0_lock.json"

    if lock_path.exists():
        raise FileExistsError(
            f"Lock already exists: {lock_path}\n"
            f"Locks are append-only. Create a new version instead."
        )

    with open(lock_path, "w") as f:
        json.dump(lock, f, indent=2)

    print(f"Created P0 lock: {lock_path}")
    return lock_path


def create_probe_lock(
    phase: str,
    predictor_version: str,
    predictions_csv: Path,
    probe_step: int,
    total_budget: int,
    locks_dir: Path = Path("locks"),
    extra_hashes: Optional[Dict[str, str]] = None,
) -> Path:
    """Create a P1 (early-probe) prediction lock.

    Must be created after exactly probe_steps of training,
    before resuming the remaining 99%.
    """
    locks_dir.mkdir(parents=True, exist_ok=True)

    now_utc = datetime.now(timezone.utc).isoformat()
    now_local = datetime.now().isoformat()
    git_sha = _get_git_sha()
    git_diff_hash = _get_git_diff_hash()

    lock = {
        "phase": phase,
        "predictor_version": predictor_version,
        "lock_type": "P1_early_probe",
        "probe_step": probe_step,
        "total_budget": total_budget,
        "probe_fraction": probe_step / total_budget,
        "utc_timestamp": now_utc,
        "local_timestamp": now_local,
        "git_sha": git_sha,
        "git_diff_hash": git_diff_hash,
        "hashes": {
            "predictions_csv": hash_file(predictions_csv),
        },
    }

    if extra_hashes is not None:
        lock["hashes"].update(extra_hashes)

    lock_path = locks_dir / f"{phase}_{predictor_version}_P1_lock.json"

    if lock_path.exists():
        raise FileExistsError(
            f"Lock already exists: {lock_path}\n"
            f"Locks are append-only. Create a new version instead."
        )

    with open(lock_path, "w") as f:
        json.dump(lock, f, indent=2)

    print(f"Created P1 probe lock: {lock_path}")
    return lock_path
