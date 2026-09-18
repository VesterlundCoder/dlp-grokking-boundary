"""Unified training loop supporting both Paper 1 and Paper 2 optimization regimes.

Paper 1 optimization: AdamW with progressive WD ramp (wd_start=0.05, wd_step=0.05,
  wd_ramp_interval=1000, lr_drop_factor=0.1). WD ramps up after memorization.
Paper 2 optimization: Adam/AdamW with fixed weight decay (0 or 0.3), no ramp.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from .models import GrokkingTransformer
from .representations import IntegerTokenizer, BitTokenizer


@dataclass
class TrainConfig:
    """Configuration for a single training run."""
    # Model
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 2
    dropout: float = 0.0

    # Training
    epochs: int = 500_000
    lr: float = 1e-3
    batch_size: int = 0  # 0 = full batch
    eval_interval: int = 100
    checkpoint_interval: int = 10_000
    early_stop_patience: int = 100
    early_stop_threshold: float = 0.99

    # Probe/resume protocol (Paper 2)
    probe_steps: int = 0       # 0 = no probe; >0 = pause after this many steps
    resume_from_probe: bool = False  # resume from probe checkpoint
    probe_checkpoint_path: str = ""  # path to probe checkpoint for resume

    # Optimization regime
    optimizer: str = "adamw"  # "adam" or "adamw"
    weight_decay: float = 0.0
    # Paper 1 progressive WD
    progressive_wd: bool = False
    wd_start: float = 0.05
    wd_step: float = 0.05
    wd_ramp_interval: int = 1000
    lr_drop_factor: float = 0.1

    # Metadata
    seed: int = 42
    run_id: str = ""
    output_dir: str = "results"

    # Device
    device: str = ""  # auto-detect if empty


class TokenDataset(Dataset):
    """Dataset of tokenized (base, target) -> x pairs."""

    def __init__(self, bases, targets, labels, tokenizer):
        self.bases = bases
        self.targets = targets
        self.labels = labels
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.bases)

    def __getitem__(self, idx):
        tokens = self.tokenizer.encode(self.bases[idx], self.targets[idx])
        label = self.tokenizer.label_to_token(self.labels[idx]) if hasattr(self.tokenizer, 'label_to_token') else self.labels[idx]
        return torch.tensor(tokens, dtype=torch.long), torch.tensor(label, dtype=torch.long)


def compute_accuracy(model, dataset, tokenizer, device, batch_size=512):
    """Compute accuracy on a dataset."""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for i in range(0, len(dataset), batch_size):
            bases = dataset.bases[i:i+batch_size]
            targets = dataset.targets[i:i+batch_size]
            labels = dataset.labels[i:i+batch_size]
            tokens = tokenizer.encode_batch(bases, targets).to(device)
            labels_t = torch.tensor(
                [tokenizer.label_to_token(l) if hasattr(tokenizer, 'label_to_token') else l for l in labels],
                dtype=torch.long, device=device,
            )
            logits = model(tokens)
            preds = logits[:, -1, :].argmax(dim=-1)
            correct += (preds == labels_t).sum().item()
            total += len(labels)
    return correct / total if total > 0 else 0.0


def train(
    config: TrainConfig,
    train_dataset: TokenDataset,
    test_dataset: TokenDataset,
    tokenizer,
    modulus: int,
) -> dict:
    """Train a GrokkingTransformer on the given datasets.

    Supports probe/resume protocol:
        - If probe_steps > 0: train for probe_steps, save checkpoint, return.
        - If resume_from_probe: load probe checkpoint, continue training.

    Returns a dict with final metrics and the path to the metrics file.
    """
    # Setup
    device = config.device or ("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device)

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    # Model
    vocab_size = tokenizer.vocab_size
    max_len = tokenizer.max_len
    model = GrokkingTransformer(
        vocab_size=vocab_size,
        d_model=config.d_model,
        n_heads=config.n_heads,
        n_layers=config.n_layers,
        max_len=max_len,
        dropout=config.dropout,
    ).to(device)

    n_params = model.count_parameters()
    n_core = model.core_parameters()

    # Resume from probe checkpoint if specified
    start_epoch = 0
    if config.resume_from_probe and config.probe_checkpoint_path:
        ckpt = torch.load(config.probe_checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        start_epoch = ckpt.get("epoch", config.probe_steps)
        print(f"[{config.run_id}] Resumed from probe at epoch {start_epoch}")

    # Optimizer
    if config.progressive_wd:
        current_wd = config.wd_start
    else:
        current_wd = config.weight_decay

    if config.optimizer == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=current_wd)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=current_wd)

    if config.resume_from_probe and config.probe_checkpoint_path:
        ckpt = torch.load(config.probe_checkpoint_path, map_location=device, weights_only=False)
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])

    criterion = nn.CrossEntropyLoss()

    # Output directory
    output_dir = Path(config.output_dir) / config.run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.jsonl"
    config_path = output_dir / "config.json"

    # Save config
    with open(config_path, "w") as f:
        json.dump(config.__dict__, f, indent=2)

    # Training data
    train_tokens = tokenizer.encode_batch(train_dataset.bases, train_dataset.targets).to(device)
    train_labels = torch.tensor(
        [tokenizer.label_to_token(l) if hasattr(tokenizer, 'label_to_token') else l for l in train_dataset.labels],
        dtype=torch.long, device=device,
    )

    # Metrics
    metrics = []
    best_test = 0.0
    best_epoch = 0
    no_improve = 0
    mem_epoch = None
    current_lr = config.lr

    t0 = time.time()

    end_epoch = config.epochs if not config.probe_steps else config.probe_steps

    for epoch in range(start_epoch, end_epoch):
        model.train()

        # Forward
        logits = model(train_tokens)
        loss = criterion(logits[:, -1, :], train_labels)

        # Backward
        optimizer.zero_grad()
        loss.backward()

        # Gradient norm (before clipping, before step)
        grad_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                grad_norm += p.grad.data.norm(2).item() ** 2
        grad_norm = grad_norm ** 0.5

        optimizer.step()

        # Weight norm
        weight_norm = 0.0
        for p in model.parameters():
            weight_norm += p.data.norm(2).item() ** 2
        weight_norm = weight_norm ** 0.5

        # Progressive WD ramp (Paper 1 style)
        if config.progressive_wd and mem_epoch is not None:
            ramps_since_mem = (epoch - mem_epoch) // config.wd_ramp_interval
            target_wd = min(config.wd_start + (ramps_since_mem + 1) * config.wd_step, 0.3)
            if target_wd != current_wd:
                current_wd = target_wd
                for pg in optimizer.param_groups:
                    pg["weight_decay"] = current_wd

        # Evaluation
        if (epoch + 1) % config.eval_interval == 0 or epoch == 0 or epoch == end_epoch - 1:
            train_acc = compute_accuracy(model, train_dataset, tokenizer, device)
            test_acc = compute_accuracy(model, test_dataset, tokenizer, device)

            if train_acc >= 0.99 and mem_epoch is None:
                mem_epoch = epoch

            if test_acc > best_test:
                best_test = test_acc
                best_epoch = epoch
                no_improve = 0
            else:
                no_improve += 1

            elapsed = time.time() - t0
            metric = {
                "epoch": epoch + 1,
                "train_acc": train_acc,
                "test_acc": test_acc,
                "loss": loss.item(),
                "grad_norm": grad_norm,
                "weight_norm": weight_norm,
                "wd": current_wd,
                "lr": current_lr,
                "elapsed_s": elapsed,
            }
            metrics.append(metric)

            with open(metrics_path, "a") as f:
                f.write(json.dumps(metric) + "\n")

            # Early stopping (only in full training, not probe)
            if not config.probe_steps and best_test >= config.early_stop_threshold and no_improve >= config.early_stop_patience:
                break

            # Print progress
            if (epoch + 1) % (config.eval_interval * 10) == 0 or epoch == 0 or epoch == end_epoch - 1:
                print(f"[{config.run_id}] Epoch {epoch+1:6d} | train={train_acc:.4f} test={test_acc:.4f} "
                      f"loss={loss.item():.4f} grad={grad_norm:.4f} wnorm={weight_norm:.2f} "
                      f"wd={current_wd:.4f} t={elapsed:.1f}s", flush=True)

    # Save probe checkpoint if in probe mode
    if config.probe_steps:
        probe_ckpt_path = output_dir / "probe_checkpoint.pt"
        torch.save({
            "epoch": end_epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": config.__dict__,
            "metrics": metrics,
        }, probe_ckpt_path)
        print(f"[{config.run_id}] Probe checkpoint saved: {probe_ckpt_path}")

    # Save final checkpoint
    final_ckpt_path = output_dir / "final_checkpoint.pt"
    torch.save({
        "epoch": end_epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": config.__dict__,
        "metrics": metrics,
    }, final_ckpt_path)

    # Final summary
    final = metrics[-1] if metrics else {}
    summary = {
        "run_id": config.run_id,
        "n_params": n_params,
        "n_core_params": n_core,
        "best_test": best_test,
        "best_epoch": best_epoch,
        "final_train": final.get("train_acc", 0),
        "final_test": final.get("test_acc", 0),
        "total_epochs": final.get("epoch", 0),
        "total_time_s": final.get("elapsed_s", 0),
        "is_probe": bool(config.probe_steps),
        "mem_epoch": mem_epoch,
    }
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    return summary
