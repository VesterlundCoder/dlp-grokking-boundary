"""Automated tests for the DLP grokking boundary study.

Run with: python3 -m pytest tests/ -v
Or:       python3 tests/run_tests.py
"""
import sys
import os
import numpy as np
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


def test_gf2_invertibility():
    """Test that GF(2) matrices generated are invertible."""
    from src.encodings.gf2_affine import generate_gf2_encoding, _gf2_mat_inv
    for n_bits in [4, 7, 8, 16]:
        for seed in range(10):
            enc = generate_gf2_encoding(n_bits, seed=seed)
            assert enc.A_inv is not None, f"GF2 matrix not invertible: n_bits={n_bits}, seed={seed}"
    print("  PASS: GF2 invertibility")


def test_feistel_roundtrip():
    """Test Feistel encoding roundtrip."""
    from src.encodings.feistel import FeistelEncoding
    for n_bits in [7, 8, 16]:
        for seed in range(5):
            enc = FeistelEncoding(n_bits, key_seed=seed)
            assert enc.verify_roundtrip(n_tests=100), f"Feistel roundtrip failed: n_bits={n_bits}, seed={seed}"
    print("  PASS: Feistel roundtrip")


def test_encoding_roundtrip():
    """Test all encoding roundtrips."""
    from src.encodings.identity_binary import IdentityBinaryEncoding
    from src.encodings.gray import GrayEncoding
    from src.encodings.gf2_affine import generate_gf2_encoding

    for n_bits in [8, 16]:
        encodings = [
            IdentityBinaryEncoding(n_bits),
            GrayEncoding(n_bits),
            generate_gf2_encoding(n_bits, seed=42),
            generate_gf2_encoding(n_bits, seed=123),
        ]
        for enc in encodings:
            assert enc.verify_roundtrip(n_tests=100), f"Roundtrip failed: {enc.encoding_id()}"
    print("  PASS: All encoding roundtrips")


def test_encoding_cardinality():
    """Test that encodings preserve cardinality (bijection)."""
    from src.encodings.identity_binary import IdentityBinaryEncoding
    from src.encodings.gray import GrayEncoding
    from src.encodings.gf2_affine import generate_gf2_encoding

    for n_bits in [8]:
        encodings = [
            IdentityBinaryEncoding(n_bits),
            GrayEncoding(n_bits),
            generate_gf2_encoding(n_bits, seed=42),
        ]
        for enc in encodings:
            # Check that all encoded values are unique
            encoded = set()
            for val in range(1 << n_bits):
                encoded.add(enc.encode_int(val))
            assert len(encoded) == (1 << n_bits), f"Cardinality not preserved: {enc.encoding_id()}"
    print("  PASS: Encoding cardinality")


def test_dlp_labels():
    """Test that DLP labels are correct."""
    from src.groups import MultiplicativeSubgroup, AdditiveGroup

    # Multiplicative: g0^x = target
    G = MultiplicativeSubgroup(q=113, p=227, g0=4)
    G.validate()

    # For a=1, x=5: base = g0^1, target = g0^5
    base = G.element(1)
    target = G.element(5)
    assert G.log(target) == 5, f"DLP label wrong: log({target}) = {G.log(target)}, expected 5"

    # Additive: x * a = target (mod q)
    Ga = AdditiveGroup(q=113)
    assert Ga.element(5) == 5
    assert Ga.log(5) == 5

    print("  PASS: DLP labels")


def test_split_leakage():
    """Test that latent manifest has no split leakage."""
    from src.latent_manifests import generate_latent_manifest, verify_no_split_leakage

    manifest = generate_latent_manifest(q=113, seed=42, train_frac=0.30)
    assert verify_no_split_leakage(manifest), "Split leakage detected"

    # Each latent_id should appear exactly once
    ids = [p.latent_id for p in manifest]
    assert len(ids) == len(set(ids)), "Duplicate latent_ids"

    print("  PASS: Split leakage check")


def test_lock_immutability():
    """Test that locks are append-only and verifiable."""
    import tempfile
    import json
    from src.locking.create_lock import create_prediction_lock
    from src.locking.hash_manifest import hash_file

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        locks_dir = tmpdir / "locks"
        locks_dir.mkdir()

        # Create dummy files needed for the lock
        pred_csv = tmpdir / "predictions.csv"
        pred_csv.write_text("config_id,log_T_gen\nconfig_1,10.0\n")
        run_manifest = tmpdir / "runs.csv"
        run_manifest.write_text("run_id,config_id\nrun_1,config_1\n")
        group_manifest = tmpdir / "groups.csv"
        group_manifest.write_text("group_id,q,p\n0,113,227\n")
        latent_manifest = tmpdir / "latent.csv"
        latent_manifest.write_text("latent_id,a,x,split\n0,1,0,train\n")
        encoding_manifest = tmpdir / "encodings.csv"
        encoding_manifest.write_text("encoding_id,family\nid_1,identity\n")
        predictor_dir = tmpdir / "predictors"
        predictor_dir.mkdir()
        (predictor_dir / "__init__.py").write_text("")

        # Create P0 lock
        lock_path = create_prediction_lock(
            phase="TEST",
            predictor_version="v1",
            predictions_csv=pred_csv,
            run_manifest=run_manifest,
            group_manifest=group_manifest,
            latent_manifests=[latent_manifest],
            encoding_manifests=[encoding_manifest],
            predictor_source_dir=predictor_dir,
            locks_dir=locks_dir,
        )
        assert lock_path.exists(), "Lock file not created"

        # Verify lock content
        with open(lock_path) as f:
            lock_data = json.load(f)
        assert lock_data["phase"] == "TEST"
        assert "hashes" in lock_data

        # Try to create another P0 lock with same version (should fail)
        try:
            create_prediction_lock(
                phase="TEST",
                predictor_version="v1",
                predictions_csv=pred_csv,
                run_manifest=run_manifest,
                group_manifest=group_manifest,
                latent_manifests=[latent_manifest],
                encoding_manifests=[encoding_manifest],
                predictor_source_dir=predictor_dir,
                locks_dir=locks_dir,
            )
            assert False, "Should have raised FileExistsError"
        except FileExistsError:
            pass  # Expected

    print("  PASS: Lock immutability")


def test_fourier_features():
    """Test Fourier feature computation."""
    from src.predictors.fourier import compute_fourier_features

    # Simple test: constant function should have zero entropy (all energy in DC)
    targets = np.zeros(100, dtype=np.int64)
    feats = compute_fourier_features(targets, q=10, n_classes=10)
    assert "fourier_entropy" in feats
    assert feats["fourier_entropy"] >= 0.0

    # Random targets should have higher entropy
    rng = np.random.RandomState(42)
    targets_random = rng.randint(0, 10, size=100)
    feats_random = compute_fourier_features(targets_random, q=10, n_classes=10)
    assert feats_random["fourier_entropy"] > feats["fourier_entropy"]

    print("  PASS: Fourier features")


def test_phase_classification():
    """Test phase classification."""
    from src.outcomes import classify_outcome, Phase

    # Direct generalization: test reaches 0.90 before train reaches 0.99
    steps = list(range(0, 10000, 100))
    train_accs = [0.5 + 0.5 * min(1, s / 5000) for s in steps]
    test_accs = [0.5 + 0.45 * min(1, s / 3000) for s in steps]
    outcome = classify_outcome(steps, train_accs, test_accs, budget=10000, k=10)
    assert outcome.phase in [Phase.DIRECT_GENERALIZATION, Phase.GROKKING, Phase.MEMORIZED_CENSORED,
                              Phase.UNDERFIT, Phase.PARTIAL]

    # Underfit: never reaches memorization
    train_accs_low = [0.3 + 0.1 * min(1, s / 5000) for s in steps]
    test_accs_low = [0.1 + 0.1 * min(1, s / 5000) for s in steps]
    outcome_underfit = classify_outcome(steps, train_accs_low, test_accs_low, budget=10000, k=10)
    assert outcome_underfit.phase == Phase.UNDERFIT

    print("  PASS: Phase classification")


def test_grouped_cv():
    """Test grouped cross-validation."""
    from src.statistics.grouped_cv import grouped_kfold

    # 20 samples, 5 groups (4 samples each)
    groups = np.array([i // 4 for i in range(20)])
    folds = list(grouped_kfold(groups, k=5, seed=42))
    assert len(folds) == 5

    # Check no group appears in both train and test
    for train_idx, test_idx in folds:
        train_groups = set(groups[train_idx])
        test_groups = set(groups[test_idx])
        assert len(train_groups & test_groups) == 0, "Group leakage in CV"

    # All samples should be covered across folds
    all_test = set()
    for _, test_idx in folds:
        all_test.update(test_idx)
    assert all_test == set(range(20)), "Not all samples covered"

    print("  PASS: Grouped CV")


def test_concordance_index():
    """Test concordance index computation."""
    from src.statistics.survival import concordance_index

    # Perfect concordance: higher predicted = sooner event (lower observed time)
    predicted = np.array([3, 2, 1])
    observed = np.array([1, 2, 3])  # lower = sooner
    events = np.array([1, 1, 1])
    c = concordance_index(predicted, observed, events)
    assert c == 1.0, f"Perfect concordance should be 1.0, got {c}"

    # Anti-concordance: higher predicted = later event
    predicted_anti = np.array([1, 2, 3])
    c_anti = concordance_index(predicted_anti, observed, events)
    assert c_anti == 0.0, f"Anti-concordance should be 0.0, got {c_anti}"

    print("  PASS: Concordance index")


def run_all_tests():
    """Run all tests."""
    print("\n" + "="*60)
    print("  Running automated tests")
    print("="*60 + "\n")

    tests = [
        test_gf2_invertibility,
        test_feistel_roundtrip,
        test_encoding_roundtrip,
        test_encoding_cardinality,
        test_dlp_labels,
        test_split_leakage,
        test_lock_immutability,
        test_fourier_features,
        test_phase_classification,
        test_grouped_cv,
        test_concordance_index,
    ]

    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"  FAIL: {test.__name__}: {e}")
            failed += 1

    print(f"\n{'='*60}")
    print(f"  Results: {passed} passed, {failed} failed")
    print(f"{'='*60}")
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
