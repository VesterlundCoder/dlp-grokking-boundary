"""Locking infrastructure for prospective prediction protocol.

No CONFIRMATORY run may start without a valid prediction lock.
Locks are append-only. Never overwrite a lock.

Lock JSON contains SHA256 hashes for:
    git commit, group manifest, latent manifest, encoding manifest,
    run manifest, predictor source code, predictor coefficients,
    pretraining feature table, exact seeds, exact optimizer configs,
    exact model configs, predictions CSV.

Also records UTC timestamp, local timestamp, Git commit SHA.
"""
from .create_lock import create_prediction_lock, create_probe_lock
from .verify_lock import verify_lock, verify_no_peeking
from .hash_manifest import hash_file, hash_directory, hash_csv_manifest

__all__ = [
    "create_prediction_lock",
    "create_probe_lock",
    "verify_lock",
    "verify_no_peeking",
    "hash_file",
    "hash_directory",
    "hash_csv_manifest",
]
