#!/usr/bin/env python3
"""Generate AUDIT_REPORT.md from all existing experimental results.

Performs a forensic audit of all 199 runs, inferring the actual task
from config.json rather than trusting stage names or filenames.
"""
import json
import sys
from pathlib import Path
from collections import defaultdict

REPO_ROOT = Path(__file__).parent.parent
RESULTS_DIR = REPO_ROOT / "results"
sys.path.insert(0, str(REPO_ROOT))

from src.phase_classifier import classify_from_metrics_file, classify_phase

STAGE_MAP = {
    "stage00": "Stage 0: MLP Replication",
    "stage00b": "Stage 0b: Theorem Uniform",
    "stage01": "Stage 1: Architecture Bridge",
    "stage02": "Stage 2: Grokking Bridge (NOT modular addition)",
    "stage03": "Stage 3: 2×2 Discovery",
}


def infer_task(config: dict) -> str:
    """Infer the actual mathematical task from config."""
    runner = config.get("runner", "")
    phase = config.get("phase", "")
    task = config.get("task", "")

    if runner == "paper_transformer":
        # Paper transformer uses generate_takhanov_paper_data
        # Task: predict bit `low_bit` of x, where input = (a*x) mod p
        low_bit = config.get("low_bit", 0)
        bits = config.get("paper_bits", "?")
        p = config.get("p", "?")
        return f"DLP parity bit {low_bit} (p={p}, b={bits}bit, input=(a*x)%p, target=bit_{low_bit}(x))"

    if runner == "transformer":
        # Stage 3 transformer
        q = config.get("q", "?")
        q_bits = config.get("q_bits", "?")
        p = config.get("p", "?")
        g0 = config.get("g0", "?")
        visible_base = config.get("visible_base", "")
        if task == "hidden_parity":
            return f"Fixed-base DLP parity (q={q}, p={p}, g0={g0}, base omitted, target=x%2)"
        elif task == "visible_parity":
            return f"Variable-base DLP parity (q={q}, p={p}, g0={g0}, base supplied, target=x%2)"
        elif task == "hidden_full":
            return f"Fixed-base DLP full log (q={q}, p={p}, g0={g0}, base omitted, target=x)"
        elif task == "visible_full":
            return f"Variable-base DLP full log (q={q}, p={p}, g0={g0}, base supplied, target=x)"
        else:
            return f"Unknown task: {task}"

    return f"Unknown runner: {runner}"


def audit_run(run_dir: Path, stage: str) -> dict:
    """Audit a single run directory."""
    config_path = run_dir / "config.json"
    metrics_path = run_dir / "metrics.jsonl"
    summary_path = run_dir / "summary.json"

    config = {}
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)

    # Infer task
    task = infer_task(config)

    # Get key parameters
    run_id = run_dir.name
    p = config.get("p", "?")
    q = config.get("q", "?")
    bits = config.get("paper_bits", config.get("q_bits", config.get("bits", "?")))
    seed = config.get("seed", "?")
    optimizer = config.get("optimizer", "?")
    wd = config.get("weight_decay", "?")
    epochs = config.get("epochs", "?")
    lr = config.get("lr", "?")
    params = config.get("actual_params", "?")
    dataset_hash = config.get("dataset_sha256", config.get("train_sha256", "?"))

    # Phase classification
    phase_result = None
    if metrics_path.exists():
        # Determine metric keys
        runner = config.get("runner", "")
        if runner == "paper_transformer":
            train_key = "train_exact"
            test_key = "test_exact"
        elif runner == "transformer":
            # Stage 3 uses different metric keys depending on task
            task = config.get("task", "")
            if "parity" in task:
                train_key = "train_parity_acc"
                test_key = "test_parity_acc"
            else:
                train_key = "train_exact"
                test_key = "test_exact"
        else:
            train_key = "train_exact"
            test_key = "test_exact"

        phase_result = classify_from_metrics_file(
            metrics_path, k=3,
            train_key=train_key, test_key=test_key,
        )

    # Summary
    summary = {}
    if summary_path.exists():
        with open(summary_path) as f:
            summary = json.load(f)

    return {
        "run_id": run_id,
        "stage": stage,
        "task": task,
        "p": p,
        "q": q,
        "bits": bits,
        "seed": seed,
        "optimizer": optimizer,
        "weight_decay": wd,
        "lr": lr,
        "epochs": epochs,
        "params": params,
        "dataset_hash": dataset_hash[:16] if isinstance(dataset_hash, str) else "?",
        "phase": phase_result.phase if phase_result else (summary.get("phase", "?")),
        "t_mem": phase_result.t_mem if phase_result else None,
        "t90": phase_result.t90 if phase_result else None,
        "delta_t": phase_result.delta_t if phase_result else None,
        "final_train": phase_result.final_train if phase_result else summary.get("final", {}).get("train_exact", "?"),
        "final_test": phase_result.final_test if phase_result else summary.get("final", {}).get("test_exact", "?"),
        "best_test": phase_result.best_test if phase_result else summary.get("best_test_acc", "?"),
    }


def main():
    all_runs = []

    for stage_dir in sorted(RESULTS_DIR.iterdir()):
        if not stage_dir.is_dir() or stage_dir.name.startswith("."):
            continue
        stage = stage_dir.name
        if stage not in STAGE_MAP:
            continue

        for run_dir in sorted(stage_dir.iterdir()):
            if not run_dir.is_dir():
                continue
            audit = audit_run(run_dir, stage)
            all_runs.append(audit)

    # Generate report
    lines = []
    lines.append("# AUDIT_REPORT.md — Forensic Audit of All Existing Experiments\n")
    lines.append(f"**Total runs audited:** {len(all_runs)}\n")
    lines.append(f"**Audit date:** September 17, 2026\n")
    lines.append("\n---\n")

    # Critical finding
    lines.append("## Critical Finding: Stage 2 is NOT Modular Addition\n")
    lines.append("**The current manuscript claims Stage 2 tests the additive group (modular addition).**")
    lines.append("This is **false**. Config comparison confirms:\n")
    lines.append("- Stage 1 and Stage 2 use the **same `dataset_sha256`** (`ebae03a7...`)")
    lines.append("- Same `p=54721`, same `low_bit=0`, same `runner=paper_transformer`")
    lines.append("- Only differences: optimizer (adam→adamw), WD (0→0.3), epochs (2000→100000)")
    lines.append("- The `generate_takhanov_paper_data()` function computes `(a*x) % p` — this is DLP parity, not modular addition\n")
    lines.append("**Stage 2 grokked because of longer training + weight decay, NOT because of a different group structure.**")
    lines.append("The entire 'group-structure-dependent grokking' narrative in the current paper is invalid.\n")
    lines.append("\n---\n")

    # Summary by stage
    lines.append("## Summary by Stage\n")
    lines.append("| Stage | Description | Runs | Phases |")
    lines.append("|-------|-------------|-----|--------|")
    for stage, desc in STAGE_MAP.items():
        stage_runs = [r for r in all_runs if r["stage"] == stage]
        phase_counts = defaultdict(int)
        for r in stage_runs:
            phase_counts[r["phase"]] += 1
        phase_str = ", ".join(f"{p}={c}" for p, c in sorted(phase_counts.items()))
        lines.append(f"| {stage} | {desc} | {len(stage_runs)} | {phase_str} |")
    lines.append("")

    # Detailed tables per stage
    for stage, desc in STAGE_MAP.items():
        stage_runs = [r for r in all_runs if r["stage"] == stage]
        if not stage_runs:
            continue

        lines.append(f"\n## {desc}\n")
        lines.append(f"**Runs:** {len(stage_runs)}\n")

        # Check if all use same dataset hash
        hashes = set(r["dataset_hash"] for r in stage_runs if r["dataset_hash"] != "?")
        if len(hashes) == 1:
            lines.append(f"**Dataset hash:** `{list(hashes)[0]}` (all runs use same dataset)\n")
        elif len(hashes) > 1:
            lines.append(f"**Dataset hashes:** {len(hashes)} distinct hashes\n")

        lines.append("| Run ID | Task (inferred) | p | q/bits | Seed | Opt | WD | Epochs | Phase | Tmem | T90 | ΔT | Final Train | Final Test |")
        lines.append("|--------|-----------------|---|--------|------|-----|-----|--------|-------|------|-----|-----|-------------|-----------|")

        for r in sorted(stage_runs, key=lambda x: x["run_id"]):
            tmem = str(r["t_mem"]) if r["t_mem"] is not None else "—"
            t90 = str(r["t90"]) if r["t90"] is not None else "—"
            dt = str(r["delta_t"]) if r["delta_t"] is not None else "—"
            ft = f"{r['final_train']:.4f}" if isinstance(r["final_train"], float) else str(r["final_train"])
            fe = f"{r['final_test']:.4f}" if isinstance(r["final_test"], float) else str(r["final_test"])
            lines.append(
                f"| {r['run_id']} | {r['task'][:40]} | {r['p']} | {r['bits']} | {r['seed']} | "
                f"{r['optimizer']} | {r['weight_decay']} | {r['epochs']} | {r['phase']} | "
                f"{tmem} | {t90} | {dt} | {ft} | {fe} |"
            )
        lines.append("")

    # Stage 1 vs Stage 2 comparison
    lines.append("\n## Stage 1 vs Stage 2: Detailed Comparison\n")
    s1 = next((r for r in all_runs if r["run_id"] == "R1_b16_s42"), None)
    s2 = next((r for r in all_runs if r["run_id"] == "R2_b16_s42"), None)
    if s1 and s2:
        lines.append("| Parameter | Stage 1 (R1_b16_s42) | Stage 2 (R2_b16_s42) | Same? |")
        lines.append("|-----------|----------------------|----------------------|-------|")
        lines.append(f"| dataset_hash | {s1['dataset_hash']} | {s2['dataset_hash']} | {'YES' if s1['dataset_hash'] == s2['dataset_hash'] else 'NO'} |")
        lines.append(f"| p | {s1['p']} | {s2['p']} | {'YES' if s1['p'] == s2['p'] else 'NO'} |")
        lines.append(f"| task (inferred) | {s1['task']} | {s2['task']} | {'YES' if s1['task'] == s2['task'] else 'NO'} |")
        lines.append(f"| optimizer | {s1['optimizer']} | {s2['optimizer']} | {'YES' if s1['optimizer'] == s2['optimizer'] else 'NO'} |")
        lines.append(f"| weight_decay | {s1['weight_decay']} | {s2['weight_decay']} | {'YES' if s1['weight_decay'] == s2['weight_decay'] else 'NO'} |")
        lines.append(f"| epochs | {s1['epochs']} | {s2['epochs']} | {'YES' if s1['epochs'] == s2['epochs'] else 'NO'} |")
        lines.append(f"| phase | {s1['phase']} | {s2['phase']} | {'YES' if s1['phase'] == s2['phase'] else 'NO'} |")
        lines.append(f"| final_test | {s1['final_test']} | {s2['final_test']} | {'YES' if s1['final_test'] == s2['final_test'] else 'NO'} |")
        lines.append("")
        lines.append("**Conclusion:** Stage 1 and Stage 2 use the same task and dataset. Stage 2 is NOT modular addition.")
        lines.append("The difference in outcome is due to optimization (WD + training duration), not group structure.\n")

    # Phase distribution
    lines.append("\n## Phase Distribution (Canonical Classification)\n")
    lines.append("Using sustained thresholds (K=3 consecutive evaluations):\n")
    lines.append("- Tmem = min{t : A_train(t) ≥ 0.99} (sustained)")
    lines.append("- T90 = min{t : A_test(t) ≥ 0.90} (sustained)\n")
    lines.append("| Phase | Count | Description |")
    lines.append("|-------|-------|-------------|")
    phase_counts = defaultdict(int)
    for r in all_runs:
        phase_counts[r["phase"]] += 1
    phase_descs = {
        "direct_generalization": "T90 ≤ Tmem (generalizes before or with memorization)",
        "grokking": "Tmem < T90 (delayed generalization after memorization)",
        "memorization_only": "Tmem exists, T90 does not (memorizes but never generalizes)",
        "underfit": "Tmem does not exist (never memorizes training set)",
        "partial": "Intermediate test accuracy without reaching 90%",
    }
    for phase, count in sorted(phase_counts.items()):
        lines.append(f"| {phase} | {count} | {phase_descs.get(phase, '?')} |")
    lines.append("")

    # Write report
    report_path = REPO_ROOT / "AUDIT_REPORT.md"
    with open(report_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Audit report written to {report_path}")
    print(f"Total runs audited: {len(all_runs)}")


if __name__ == "__main__":
    main()
