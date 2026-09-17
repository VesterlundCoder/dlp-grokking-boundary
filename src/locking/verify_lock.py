"""Verify prediction locks for the prospective prediction protocol.

The launcher must refuse to start confirmatory jobs unless verify_lock passes.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from .hash_manifest import hash_file, hash_csv_manifest


def verify_lock(lock_path: Path) -> bool:
    """Verify that a lock file is valid and hashes match current files.

    Returns True if all hashes match, False otherwise.
    """
    if not lock_path.exists():
        print(f"ERROR: Lock file does not exist: {lock_path}")
        return False

    with open(lock_path) as f:
        lock = json.load(f)

    hashes = lock.get("hashes", {})
    errors = []

    # Check predictions CSV
    if "predictions_csv" in hashes:
        # We need the path — derive from lock name convention
        # predictions/<phase>_P0_predictions.csv
        phase = lock.get("phase", "")
        pversion = lock.get("predictor_version", "")
        lock_type = lock.get("lock_type", "")

        if "P0" in lock_type:
            pred_path = Path(f"predictions/{phase}_{pversion}_P0_predictions.csv")
        else:
            pred_path = Path(f"predictions/{phase}_{pversion}_P1_predictions.csv")

        if pred_path.exists():
            current_hash = hash_file(pred_path)
            if current_hash != hashes["predictions_csv"]:
                errors.append(f"predictions_csv hash mismatch: {pred_path}")
        else:
            errors.append(f"predictions_csv not found: {pred_path}")

    if errors:
        print(f"LOCK VERIFICATION FAILED: {lock_path}")
        for e in errors:
            print(f"  - {e}")
        return False

    print(f"Lock verified: {lock_path}")
    return True


def verify_no_peeking(
    lock_path: Path,
    results_dir: Path,
    probe_step: Optional[int] = None,
) -> bool:
    """Verify that no training results exist beyond the probe step.

    For P0 locks: no training results should exist at all.
    For P1 locks: no checkpoint beyond probe_step should exist.
    """
    with open(lock_path) as f:
        lock = json.load(f)

    lock_type = lock.get("lock_type", "")

    if "P0" in lock_type:
        # No training results should exist
        if results_dir.exists():
            for p in results_dir.rglob("metrics.jsonl"):
                errors.append(f"Training results found before P0 lock: {p}")
                return False
    elif "P1" in lock_type:
        # No checkpoint beyond probe_step should exist
        probe_step = lock.get("probe_step", probe_step)
        if probe_step is None:
            print("WARNING: No probe_step specified for P1 lock verification")
            return True

        if results_dir.exists():
            for metrics_file in results_dir.rglob("metrics.jsonl"):
                import json as j
                with open(metrics_file) as f:
                    for line in f:
                        d = j.loads(line)
                        step = d.get("optimizer_step", d.get("epoch", 0))
                        if step > probe_step:
                            print(f"ERROR: Results beyond probe_step found: {metrics_file} (step={step})")
                            return False

    return True


def require_lock_for_phase(
    phase: str,
    predictor_version: str,
    lock_type: str = "P0",
    locks_dir: Path = Path("locks"),
) -> Path:
    """Require that a valid lock exists for a phase.

    Raises RuntimeError if no valid lock exists.
    """
    if "P0" in lock_type:
        lock_path = locks_dir / f"{phase}_{predictor_version}_P0_lock.json"
    else:
        lock_path = locks_dir / f"{phase}_{predictor_version}_P1_lock.json"

    if not verify_lock(lock_path):
        raise RuntimeError(
            f"No valid {lock_type} lock for phase {phase}, predictor {predictor_version}.\n"
            f"Cannot start confirmatory runs without a prediction lock.\n"
            f"Create a lock first using create_prediction_lock()."
        )

    return lock_path
