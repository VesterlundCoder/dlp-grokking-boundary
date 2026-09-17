#!/usr/bin/env python3
"""Generate all figures for 'The Grokking Boundary' paper from LUMI results."""

import json
import os
import sys
from pathlib import Path
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

RESULTS_DIR = Path(__file__).parent.parent / "results"
FIGURES_DIR = Path(__file__).parent.parent / "paper" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# Helpers
# ============================================================================

def load_stage00():
    """Load Stage 00 (MLP replication) summaries."""
    data = []
    d = RESULTS_DIR / "stage00"
    if not d.exists():
        return data
    for run_dir in sorted(d.iterdir()):
        sf = run_dir / "summary.json"
        if sf.exists():
            with open(sf) as f:
                s = json.load(f)
            # Parse name: R0_b{bits}_rep{r}_bit{b}
            parts = run_dir.name.split("_")
            bits = int(parts[1][1:])
            rep = int(parts[2][3:])
            bit = int(parts[3][3:])
            final = s.get("final", {})
            data.append({
                "bits": bits, "rep": rep, "bit": bit,
                "final_test": final.get("test_exact", 0),
                "phase": s.get("phase", "?"),
                "t90": s.get("t90"),
                "params": s.get("actual_params", 0),
            })
    return data


def load_stage00b():
    """Load Stage 00b (theorem uniform) summaries."""
    data = []
    d = RESULTS_DIR / "stage00b"
    if not d.exists():
        return data
    for run_dir in sorted(d.iterdir()):
        sf = run_dir / "summary.json"
        if sf.exists():
            with open(sf) as f:
                s = json.load(f)
            parts = run_dir.name.split("_")
            bits = int(parts[1][1:])
            seed = int(parts[2][1:])
            final = s.get("final", {})
            data.append({
                "bits": bits, "seed": seed,
                "final_test": final.get("test_exact", 0),
                "phase": s.get("phase", "?"),
                "t90": s.get("t90"),
            })
    return data


def load_stage01():
    """Load Stage 01 (architecture bridge) final metrics."""
    data = []
    d = RESULTS_DIR / "stage01"
    if not d.exists():
        return data
    for run_dir in sorted(d.iterdir()):
        mf = run_dir / "metrics.jsonl"
        if mf.exists():
            lines = mf.read_text().strip().split("\n")
            if lines:
                last = json.loads(lines[-1])
                parts = run_dir.name.split("_")
                bits = int(parts[1][1:])
                seed = int(parts[2][1:])
                data.append({
                    "bits": bits, "seed": seed,
                    "epoch": last.get("epoch", 0),
                    "train": last.get("train_exact", 0),
                    "test": last.get("test_exact", 0),
                })
    return data


def load_stage02():
    """Load Stage 02 (grokking bridge) final metrics."""
    data = []
    d = RESULTS_DIR / "stage02"
    if not d.exists():
        return data
    for run_dir in sorted(d.iterdir()):
        mf = run_dir / "metrics.jsonl"
        if mf.exists():
            lines = mf.read_text().strip().split("\n")
            if lines:
                last = json.loads(lines[-1])
                parts = run_dir.name.split("_")
                bits = int(parts[1][1:])
                seed = int(parts[2][1:])
                data.append({
                    "bits": bits, "seed": seed,
                    "epoch": last.get("epoch", 0),
                    "train": last.get("train_exact", 0),
                    "test": last.get("test_exact", 0),
                })
    return data


def load_stage03():
    """Load Stage 03 (2x2 discovery) final metrics."""
    data = []
    d = RESULTS_DIR / "stage03"
    if not d.exists():
        return data
    for run_dir in sorted(d.iterdir()):
        mf = run_dir / "metrics.jsonl"
        if mf.exists():
            lines = mf.read_text().strip().split("\n")
            if lines:
                last = json.loads(lines[-1])
                # Parse: M1_b{bits}_{task}_{seed}
                parts = run_dir.name.split("_")
                bits = int(parts[1][1:])
                task = parts[2]  # hidden or visible
                target = parts[3]  # parity or full
                seed = int(parts[4][1:])
                data.append({
                    "bits": bits, "task": task, "target": target, "seed": seed,
                    "epoch": last.get("epoch", 0),
                    "test_parity": last.get("test_parity_acc", 0),
                    "test_exact": last.get("test_exact", 0),
                    "train_parity": last.get("train_parity_acc", 0),
                })
    return data


# ============================================================================
# Figure 1: MLP replication — accuracy vs bit size
# ============================================================================

def fig_mlp_replication():
    data = load_stage00()
    if not data:
        print("No Stage 00 data, skipping fig_mlp_replication")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Left: scatter of all runs
    by_bits = defaultdict(list)
    for d in data:
        by_bits[d["bits"]].append(d["final_test"])

    bits_sorted = sorted(by_bits.keys())
    for b in bits_sorted:
        accs = by_bits[b]
        x = [b] * len(accs)
        colors = []
        for a in accs:
            if a >= 0.9:
                colors.append('#2ecc71')
            elif a >= 0.7:
                colors.append('#f39c12')
            elif a >= 0.55:
                colors.append('#e67e22')
            else:
                colors.append('#e74c3c')
        ax1.scatter(x, accs, c=colors, s=80, zorder=5, edgecolors='black', linewidth=0.5)

    # Mean line
    means = [np.mean(by_bits[b]) for b in bits_sorted]
    ax1.plot(bits_sorted, means, 'ko-', markersize=8, linewidth=2, label='Mean')
    ax1.axhline(y=0.9, color='gray', linestyle='--', alpha=0.5, label='Grokking threshold')
    ax1.axhline(y=0.5, color='gray', linestyle=':', alpha=0.3)
    ax1.set_xlabel('Bit size (b)')
    ax1.set_ylabel('Final test accuracy')
    ax1.set_title('MLP Replication: DLP Parity Bit Learning')
    ax1.set_xticks(bits_sorted)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Right: phase distribution
    phase_colors = {
        'direct_generalization': '#2ecc71',
        'grokking': '#3498db',
        'partial_generalization': '#f39c12',
        'memorization_only': '#e74c3c',
    }
    for b in bits_sorted:
        runs = by_bits[b]
        phases = [d["phase"] for d in data if d["bits"] == b]
        phase_counts = defaultdict(int)
        for p in phases:
            phase_counts[p] += 1
        bottom = 0
        for phase, count in phase_counts.items():
            ax2.bar(b, count, bottom=bottom, color=phase_colors.get(phase, 'gray'),
                    edgecolor='black', linewidth=0.5, label=phase if b == bits_sorted[0] else '')
            bottom += count

    ax2.set_xlabel('Bit size (b)')
    ax2.set_ylabel('Number of runs (15 per bit size)')
    ax2.set_title('Phase Distribution by Bit Size')
    ax2.set_xticks(bits_sorted)
    # Custom legend
    handles = [mpatches.Patch(color=c, label=p) for p, c in phase_colors.items()]
    ax2.legend(handles=handles, loc='upper right')
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig_mlp_replication.pdf", dpi=150, bbox_inches='tight')
    plt.savefig(FIGURES_DIR / "fig_mlp_replication.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Generated fig_mlp_replication.pdf")


# ============================================================================
# Figure 2: Theorem uniform (Stage 00b)
# ============================================================================

def fig_theorem_uniform():
    data = load_stage00b()
    if not data:
        print("No Stage 00b data, skipping fig_theorem_uniform")
        return

    fig, ax = plt.subplots(figsize=(8, 5))

    by_bits = defaultdict(list)
    for d in data:
        by_bits[d["bits"]].append(d["final_test"])

    bits_sorted = sorted(by_bits.keys())
    for b in bits_sorted:
        accs = by_bits[b]
        ax.scatter([b] * len(accs), accs, s=150, zorder=5, edgecolors='black', linewidth=0.5,
                   c=['#2ecc71' if a >= 0.9 else '#e74c3c' for a in accs])

    means = [np.mean(by_bits[b]) for b in bits_sorted]
    ax.plot(bits_sorted, means, 'ko-', markersize=8, linewidth=2, label='Mean')
    ax.axhline(y=0.9, color='gray', linestyle='--', alpha=0.5, label='Grokking threshold')
    ax.axhline(y=0.5, color='gray', linestyle=':', alpha=0.3)
    ax.set_xlabel('Bit size (b)')
    ax.set_ylabel('Final test accuracy')
    ax.set_title('Theorem Uniform Distribution: MLP DLP Parity')
    ax.set_xticks(bits_sorted)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig_theorem_uniform.pdf", dpi=150, bbox_inches='tight')
    plt.savefig(FIGURES_DIR / "fig_theorem_uniform.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Generated fig_theorem_uniform.pdf")


# ============================================================================
# Figure 3: Architecture bridge (Stage 01) — Transformer on prime-order DLP
# ============================================================================

def fig_architecture_bridge():
    data = load_stage01()
    if not data:
        print("No Stage 01 data, skipping fig_architecture_bridge")
        return

    fig, ax = plt.subplots(figsize=(8, 5))

    by_bits = defaultdict(list)
    for d in data:
        by_bits[d["bits"]].append(d["test"])

    bits_sorted = sorted(by_bits.keys())
    for b in bits_sorted:
        accs = by_bits[b]
        ax.scatter([b] * len(accs), accs, s=150, zorder=5, edgecolors='black', linewidth=0.5,
                   c=['#e74c3c'] * len(accs))

    means = [np.mean(by_bits[b]) for b in bits_sorted]
    ax.plot(bits_sorted, means, 'ko-', markersize=8, linewidth=2, label='Mean')
    ax.axhline(y=0.5, color='gray', linestyle=':', alpha=0.3, label='Chance level')
    ax.set_xlabel('Bit size (b)')
    ax.set_ylabel('Final test accuracy (2000 epochs)')
    ax.set_title('Architecture Bridge: Bit-Tokenizer Transformer on Prime-Order DLP')
    ax.set_xticks(bits_sorted)
    ax.set_ylim(0.4, 0.6)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig_architecture_bridge.pdf", dpi=150, bbox_inches='tight')
    plt.savefig(FIGURES_DIR / "fig_architecture_bridge.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Generated fig_architecture_bridge.pdf")


# ============================================================================
# Figure 4: Grokking bridge (Stage 02) — Transformer on additive group
# ============================================================================

def fig_grokking_bridge():
    data = load_stage02()
    if not data:
        print("No Stage 02 data, skipping fig_grokking_bridge")
        return

    fig, ax = plt.subplots(figsize=(8, 5))

    by_bits = defaultdict(list)
    for d in data:
        by_bits[d["bits"]].append(d["test"])

    bits_sorted = sorted(by_bits.keys())
    for b in bits_sorted:
        accs = by_bits[b]
        colors = ['#2ecc71' if a >= 0.9 else '#e74c3c' for a in accs]
        ax.scatter([b] * len(accs), accs, s=150, zorder=5, edgecolors='black', linewidth=0.5, c=colors)

    means = [np.mean(by_bits[b]) for b in bits_sorted]
    ax.plot(bits_sorted, means, 'ko-', markersize=8, linewidth=2, label='Mean')
    ax.axhline(y=0.9, color='gray', linestyle='--', alpha=0.5, label='Grokking threshold')
    ax.axhline(y=0.5, color='gray', linestyle=':', alpha=0.3, label='Chance level')
    ax.set_xlabel('Bit size (b)')
    ax.set_ylabel('Final test accuracy (100k epochs)')
    ax.set_title('Grokking Bridge: Bit-Tokenizer Transformer on Additive Group')
    ax.set_xticks(bits_sorted)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig_grokking_bridge.pdf", dpi=150, bbox_inches='tight')
    plt.savefig(FIGURES_DIR / "fig_grokking_bridge.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Generated fig_grokking_bridge.pdf")


# ============================================================================
# Figure 5: 2x2 discovery (Stage 03) — heatmap
# ============================================================================

def fig_2x2_discovery():
    data = load_stage03()
    if not data:
        print("No Stage 03 data, skipping fig_2x2_discovery")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Left: parity accuracy by bit size and task
    for task in ['hidden', 'visible']:
        accs_by_bits = defaultdict(list)
        for d in data:
            if d["target"] == "parity":
                accs_by_bits[d["bits"]].append(d["test_parity"])
        bits_sorted = sorted(accs_by_bits.keys())
        means = [np.mean(accs_by_bits[b]) for b in bits_sorted]
        stds = [np.std(accs_by_bits[b]) for b in bits_sorted]
        label = f'{task} base'
        color = '#3498db' if task == 'hidden' else '#e74c3c'
        ax1.errorbar(bits_sorted, means, yerr=stds, fmt='o-', color=color, label=label, capsize=5, markersize=8)

    ax1.axhline(y=0.5, color='gray', linestyle=':', alpha=0.3, label='Chance level')
    ax1.set_xlabel('Bit size (b)')
    ax1.set_ylabel('Test parity accuracy (200k epochs)')
    ax1.set_title('2×2 Discovery: Parity Task Accuracy')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Right: full-log accuracy by bit size and task
    for task in ['hidden', 'visible']:
        accs_by_bits = defaultdict(list)
        for d in data:
            if d["target"] == "full":
                accs_by_bits[d["bits"]].append(d["test_exact"])
        bits_sorted = sorted(accs_by_bits.keys())
        means = [np.mean(accs_by_bits[b]) for b in bits_sorted]
        stds = [np.std(accs_by_bits[b]) for b in bits_sorted]
        label = f'{task} base'
        color = '#3498db' if task == 'hidden' else '#e74c3c'
        ax2.errorbar(bits_sorted, means, yerr=stds, fmt='o-', color=color, label=label, capsize=5, markersize=8)

    ax2.axhline(y=0.5, color='gray', linestyle=':', alpha=0.3, label='Chance level')
    ax2.set_xlabel('Bit size (b)')
    ax2.set_ylabel('Test exact accuracy (200k epochs)')
    ax2.set_title('2×2 Discovery: Full-Log Task Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig_2x2_discovery.pdf", dpi=150, bbox_inches='tight')
    plt.savefig(FIGURES_DIR / "fig_2x2_discovery.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Generated fig_2x2_discovery.pdf")


# ============================================================================
# Figure 6: The grokking boundary — combined view
# ============================================================================

def fig_grokking_boundary():
    """The key figure: shows the boundary between grokking and non-grokking."""
    s00 = load_stage00()
    s00b = load_stage00b()
    s01 = load_stage01()
    s02 = load_stage02()

    fig, ax = plt.subplots(figsize=(10, 6))

    # MLP replication (Stage 00)
    if s00:
        by_bits = defaultdict(list)
        for d in s00:
            by_bits[d["bits"]].append(d["final_test"])
        bits_sorted = sorted(by_bits.keys())
        means = [np.mean(by_bits[b]) for b in bits_sorted]
        ax.plot(bits_sorted, means, 's-', color='#2ecc71', markersize=10, linewidth=2, label='MLP (DLP parity, natural dist.)')

    # Theorem uniform (Stage 00b)
    if s00b:
        by_bits = defaultdict(list)
        for d in s00b:
            by_bits[d["bits"]].append(d["final_test"])
        bits_sorted = sorted(by_bits.keys())
        means = [np.mean(by_bits[b]) for b in bits_sorted]
        ax.plot(bits_sorted, means, 'D-', color='#27ae60', markersize=10, linewidth=2, label='MLP (DLP parity, uniform dist.)')

    # Architecture bridge (Stage 01) — Transformer on prime-order DLP
    if s01:
        by_bits = defaultdict(list)
        for d in s01:
            by_bits[d["bits"]].append(d["test"])
        bits_sorted = sorted(by_bits.keys())
        means = [np.mean(by_bits[b]) for b in bits_sorted]
        ax.plot(bits_sorted, means, '^-', color='#e74c3c', markersize=10, linewidth=2, label='Transformer (prime-order DLP)')

    # Grokking bridge (Stage 02) — Transformer on additive group
    if s02:
        by_bits = defaultdict(list)
        for d in s02:
            by_bits[d["bits"]].append(d["test"])
        bits_sorted = sorted(by_bits.keys())
        means = [np.mean(by_bits[b]) for b in bits_sorted]
        ax.plot(bits_sorted, means, 'v-', color='#3498db', markersize=10, linewidth=2, label='Transformer (additive group)')

    ax.axhline(y=0.9, color='gray', linestyle='--', alpha=0.5, label='Grokking threshold')
    ax.axhline(y=0.5, color='gray', linestyle=':', alpha=0.3, label='Chance level')

    # Shade the grokking boundary
    ax.axvspan(16, 18, alpha=0.1, color='gray', label='Grokking boundary')

    ax.set_xlabel('Bit size (b)', fontsize=12)
    ax.set_ylabel('Final test accuracy', fontsize=12)
    ax.set_title('The Grokking Boundary: Group Structure Determines Learnability', fontsize=13)
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.4, 1.05)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig_grokking_boundary.pdf", dpi=150, bbox_inches='tight')
    plt.savefig(FIGURES_DIR / "fig_grokking_boundary.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Generated fig_grokking_boundary.pdf")


# ============================================================================

if __name__ == '__main__':
    fig_mlp_replication()
    fig_theorem_uniform()
    fig_architecture_bridge()
    fig_grokking_bridge()
    fig_2x2_discovery()
    fig_grokking_boundary()
    print(f'\nAll figures generated in {FIGURES_DIR}')
