"""
hyperparameter_search2.py
=========================

Loops over a list of model/training configurations, trains each one for
a (short-ish) number of epochs, and records the best validation accuracy
achieved. Results are printed as a ranked table and saved to
'search_results.csv' so you can compare runs afterwards.

Put this file next to your existing train.py (same folder structure:
models/transformer.py, dataset/pytorch_dataset.py,
preprocessing/prepare_dataset.py). Replace models/transformer.py with the
updated version that accepts `use_mask` if you want to compare
mask vs. no-mask.

Usage:
    python hyperparameter_search2.py
"""

import csv
import json
import time
import itertools

import torch
import torch.nn as nn

from models.transformer import ISLRTransformer
from dataset.pytorch_dataset import create_dataloaders
from preprocessing.prepare_dataset import get_valid_glosses


# ============================================================
# 0. SEARCH SETTINGS
# ============================================================

SEARCH_EPOCHS = 25          # round 1 winners were still climbing at epoch 15, so give
                             # round 2 more room to actually peak before comparing
BATCH_SIZE = 32
MAX_LENGTH = 30
MIN_FRAMES = 3
ROOT = "processed_landmarks"

RESULTS_CSV = "search_results.csv"
BEST_MODEL_PREFIX = "islr_search_best"


# ============================================================
# 1. DEFINE THE CONFIGURATIONS TO TRY
# ============================================================
#
# Each dict is one experiment. Any key you leave out falls back to the
# DEFAULTS below, so you only need to specify what changes.
#
# This is a hand-picked list rather than a full grid, because a full
# grid over every parameter explodes very fast (see the itertools
# version further down if you want that instead).

DEFAULTS = {
    "num_layers": 4,
    "hidden_dim": 256,
    "num_heads": 8,
    "dropout": 0.3,
    "lr": 1e-4,
    "weight_decay": 1e-5,
    "scheduler": "step",     # "step" | "cosine" | "none"
    "use_mask": True,
}

CONFIGS = [
    # ROUND 2 -- combining the winning ingredients from round 1:
    # higher_lr_3e-4 (80.88%) and low_dropout_0.1 (79.61%) were both
    # still climbing at epoch 15 with a healthy train/val gap.
    # larger_hidden_384 (22.74%) overfit hard at dropout=0.3/lr=1e-4 --
    # worth re-testing now that lr/dropout are tuned better.
    # shallower_2layers and high_dropout_0.5 are dropped (clearly bad).

    {"name": "r2_lr3e4_drop0.1", "lr": 3e-4, "dropout": 0.1},
    {"name": "r2_lr3e4_drop0.1_nosched", "lr": 3e-4, "dropout": 0.1, "scheduler": "none"},
    {"name": "r2_lr3e4_drop0.1_cosine", "lr": 3e-4, "dropout": 0.1, "scheduler": "cosine"},
    {"name": "r2_lr3e4_drop0.15_6layers", "lr": 3e-4, "dropout": 0.15, "num_layers": 6},
    {"name": "r2_lr3e4_drop0.15_hidden384", "lr": 3e-4, "dropout": 0.15, "hidden_dim": 384},
    {"name": "r2_lr3e4_drop0.1_nomask", "lr": 3e-4, "dropout": 0.1, "use_mask": False},
    {"name": "r2_lr2e4_drop0.1", "lr": 2e-4, "dropout": 0.1},  # a bit between baseline and 3e-4
]


# ------------------------------------------------------------
# OPTIONAL: full grid search instead of the hand-picked list.
# WARNING: this multiplies out fast (below = 2*2*2*2 = 16 runs).
# Uncomment to use it, and comment out CONFIGS above.
# ------------------------------------------------------------
#
# grid = {
#     "num_layers": [2, 4],
#     "hidden_dim": [128, 256],
#     "scheduler": ["step", "cosine"],
#     "use_mask": [True, False],
# }
# keys = list(grid.keys())
# CONFIGS = []
# for combo in itertools.product(*grid.values()):
#     cfg = dict(zip(keys, combo))
#     cfg["name"] = "_".join(f"{k}={v}" for k, v in cfg.items())
#     CONFIGS.append(cfg)


# ============================================================
# 2. LOAD DATA (once, shared across all configs)
# ============================================================

print("Φορτώνουμε τα δεδομένα...")

train_valid = get_valid_glosses("data/train_greek_iso.csv", ROOT, MIN_FRAMES)
val_valid = get_valid_glosses("data/val_greek_iso.csv", ROOT, MIN_FRAMES)
test_valid = get_valid_glosses("data/test_greek_iso.csv", ROOT, MIN_FRAMES)

with open("dataset/dictionary.json", "r", encoding="utf-8") as f:
    dictionary = json.load(f)

train_loader, val_loader, test_loader = create_dataloaders(
    train_valid,
    val_valid,
    test_valid,
    dictionary,
    max_length=MAX_LENGTH,
    batch_size=BATCH_SIZE
)

num_classes = len(dictionary)

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Χρησιμοποιείται: {device}")


# ============================================================
# 3. TRAIN/EVAL LOOP FOR ONE CONFIG
# ============================================================

def build_scheduler(kind, optimizer):
    if kind == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    if kind == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=SEARCH_EPOCHS)
    if kind == "none":
        return None
    raise ValueError(f"Unknown scheduler kind: {kind}")


def run_config(cfg):
    """Trains one configuration for SEARCH_EPOCHS epochs and returns a
    results dict with the best validation accuracy reached."""

    # merge defaults + this config
    params = {**DEFAULTS, **cfg}
    name = params.get("name", "unnamed")

    print(f"\n{'=' * 60}")
    print(f"CONFIG: {name}")
    print(json.dumps({k: v for k, v in params.items() if k != "name"}, indent=2))
    print("=" * 60)

    model = ISLRTransformer(
        input_dim=292,
        hidden_dim=params["hidden_dim"],
        num_heads=params["num_heads"],
        num_layers=params["num_layers"],
        num_classes=num_classes,
        max_length=MAX_LENGTH,
        dropout=params["dropout"]
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=params["lr"],
        weight_decay=params["weight_decay"]
    )

    scheduler = build_scheduler(params["scheduler"], optimizer)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    use_mask = params["use_mask"]

    best_val_accuracy = 0.0
    best_epoch = -1
    start_time = time.time()

    for epoch in range(SEARCH_EPOCHS):

        # ---------------- TRAIN ----------------
        model.train()
        total_train_loss = 0.0
        correct_train = 0
        total_train = 0

        for batch_keypoints, batch_labels, batch_lengths in train_loader:
            batch_keypoints = batch_keypoints.to(device)
            batch_labels = batch_labels.to(device)
            batch_lengths = batch_lengths.to(device)

            optimizer.zero_grad()

            logits = model(batch_keypoints, batch_lengths, use_mask=use_mask)
            loss = criterion(logits, batch_labels)

            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            predictions = logits.argmax(dim=1)
            correct_train += (predictions == batch_labels).sum().item()
            total_train += batch_labels.size(0)

        train_accuracy = correct_train / total_train
        avg_train_loss = total_train_loss / len(train_loader)

        # ---------------- VALIDATE ----------------
        model.eval()
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for batch_keypoints, batch_labels, batch_lengths in val_loader:
                batch_keypoints = batch_keypoints.to(device)
                batch_labels = batch_labels.to(device)
                batch_lengths = batch_lengths.to(device)

                logits = model(batch_keypoints, batch_lengths, use_mask=use_mask)
                predictions = logits.argmax(dim=1)

                correct_val += (predictions == batch_labels).sum().item()
                total_val += batch_labels.size(0)

        val_accuracy = correct_val / total_val

        if scheduler is not None:
            scheduler.step()

        is_best = val_accuracy > best_val_accuracy
        if is_best:
            best_val_accuracy = val_accuracy
            best_epoch = epoch + 1
            torch.save(model.state_dict(), f"{BEST_MODEL_PREFIX}_{name}.pt")

        marker = " <- best so far" if is_best else ""
        print(
            f"  Epoch {epoch + 1}/{SEARCH_EPOCHS} -- "
            f"Train loss: {avg_train_loss:.4f}, "
            f"Train acc: {train_accuracy:.2%}, "
            f"Val acc: {val_accuracy:.2%}{marker}"
        )

    elapsed = time.time() - start_time

    result = {
        "name": name,
        "best_val_accuracy": best_val_accuracy,
        "best_epoch": best_epoch,
        "train_time_sec": round(elapsed, 1),
        **{k: v for k, v in params.items() if k != "name"}
    }

    print(f"-> Best val acc for '{name}': {best_val_accuracy:.2%} (epoch {best_epoch})")

    return result


# ============================================================
# 4. RUN THE SEARCH
# ============================================================

all_results = []

for cfg in CONFIGS:
    result = run_config(cfg)
    all_results.append(result)

    # save results incrementally, so a crash doesn't lose everything
    fieldnames = list(all_results[0].keys())
    with open(RESULTS_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_results)


# ============================================================
# 5. PRINT RANKED SUMMARY
# ============================================================

all_results.sort(key=lambda r: r["best_val_accuracy"], reverse=True)

print("\n" + "=" * 60)
print("RANKED RESULTS (best -> worst)")
print("=" * 60)

for rank, r in enumerate(all_results, start=1):
    print(
        f"{rank:2d}. {r['name']:<25} "
        f"val_acc={r['best_val_accuracy']:.2%}  "
        f"(epoch {r['best_epoch']}, {r['train_time_sec']}s)"
    )

print(f"\nFull results saved to: {RESULTS_CSV}")
print(f"Best config overall: {all_results[0]['name']} "
      f"with {all_results[0]['best_val_accuracy']:.2%} val accuracy")
print(f"Its weights were saved to: {BEST_MODEL_PREFIX}_{all_results[0]['name']}.pt")