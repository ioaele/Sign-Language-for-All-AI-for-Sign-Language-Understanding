# Run from the project root with: python3 -m models.configurations_test
# One-at-a-time hyperparameter sweep using the basic dataloader (no augmentation).
# Every config changes ONE setting from the baseline, so you can see its effect in isolation.
# Only validation accuracy is used. The test set is never touched.

import csv
import json
import time
from pathlib import Path

import torch
import torch.nn as nn

from dataset.dataset import create_dataloaders
from preprocessing.get_glosses import get_glosses
from models.transformer import ISLR_Transformer

# ---------------- settings ----------------
EPOCHS = 50
EARLY_STOP = 10
BATCH_SIZE = 32
SEED = 42

BASELINE = dict(
    max_length=40,
    lr=1e-4,
    optimizer="adam",
    num_heads=4,
    pooling_method="mean",
    encoder_layers=2,
    dim_feedforward=512,
    dropout=0.1,
)

# values to try for each parameter (the baseline value is skipped automatically)
SEARCH_SPACE = {
    "max_length": [20, 30, 40, 50, 60],
    "lr": [1e-5, 5e-5, 1e-4, 5e-4, 1e-3],
    "optimizer": ["adam", "adamw", "rmsprop", "sgd"],
    "num_heads": [1, 2, 4, 8],             # d_model=128 must be divisible by this
    "pooling_method": ["mean", "max", "attention"],
    "encoder_layers": [1, 2, 3, 4],
    "dim_feedforward": [128, 256, 512, 1024],
    "dropout": [0.0, 0.1, 0.2, 0.3, 0.5],
}

# ---------------- paths ----------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = PROJECT_ROOT / "processed_landmarks"
TRAIN_CSV = PROJECT_ROOT / "data" / "train_greek_iso.csv"
VAL_CSV = PROJECT_ROOT / "data" / "val_greek_iso.csv"
TEST_CSV = PROJECT_ROOT / "data" / "test_greek_iso.csv"
DICTIONARY_PATH = PROJECT_ROOT / "dataset" / "dictionary.json"
RESULTS_CSV = PROJECT_ROOT / "configurations_test.csv"
BEST_MODEL_PATH = PROJECT_ROOT / "configurations_test_best_model.pt"

    DEVICE = torch.device("cpu")
print("Device:", DEVICE)


def make_optimizer(name, params, lr):
    if name == "adam":
        return torch.optim.Adam(params, lr=lr)
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr)
    if name == "rmsprop":
        return torch.optim.RMSprop(params, lr=lr)
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=0.9)
    raise ValueError(f"Unknown optimizer: {name}")


def run_epoch(model, loader, loss_function, device, optimizer=None):
    training = optimizer is not None
    model.train() if training else model.eval()
    total_loss, correct, total = 0.0, 0, 0

    with torch.set_grad_enabled(training):
        for landmarks, labels, not_padded_frames in loader:
            landmarks = landmarks.to(device)
            labels = labels.to(device)
            not_padded_frames = not_padded_frames.to(device)

            logits = model(landmarks, not_padded_frames)
            loss = loss_function(logits, labels)

            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            n = labels.size(0)
            total_loss += loss.item() * n
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += n

    return total_loss / total, correct / total


def run_config(cfg, train_set, val_set, test_set, dictionary):
    # max_length changes the data, so the loaders are rebuilt for every config
    train_loader, val_loader, _ = create_dataloaders(
        train_set, val_set, test_set, dictionary,
        max_length=cfg["max_length"], batch_size=BATCH_SIZE,
    )

    torch.manual_seed(SEED)
    model = ISLR_Transformer(
        num_landmarks=292,
        max_length=cfg["max_length"],
        num_glosses=len(dictionary),
        num_heads=cfg["num_heads"],
        encoder_layers=cfg["encoder_layers"],
        dim_feedforwardnetwork=cfg["dim_feedforward"],
        percentage_zero_dropout=cfg["dropout"],
        pooling_method=cfg["pooling_method"],
    ).to(DEVICE)

    loss_function = nn.CrossEntropyLoss()
    optimizer = make_optimizer(cfg["optimizer"], model.parameters(), cfg["lr"])

    best_val_acc, best_epoch, best_state, no_improve = 0.0, 0, None, 0
    train_acc_at_best = 0.0

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = run_epoch(
            model, train_loader, loss_function, DEVICE, optimizer
        )
        val_loss, val_acc = run_epoch(
            model, val_loader, loss_function, DEVICE
        )

        print(
            f"   Epoch [{epoch}/{EPOCHS}] | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_acc:.2%} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.2%}"
        )
        if val_acc > best_val_acc:
            best_val_acc, best_epoch, train_acc_at_best = val_acc, epoch, train_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= EARLY_STOP:
                break

    return best_val_acc, train_acc_at_best, best_epoch, epoch, best_state


def build_configs():
    """Baseline first, then one config per changed value (duplicates of baseline skipped)."""
    configs = [("baseline", "-", dict(BASELINE))]
    for param, values in SEARCH_SPACE.items():
        for value in values:
            if value == BASELINE[param]:
                continue
            cfg = dict(BASELINE)
            cfg[param] = value
            configs.append((param, value, cfg))
    return configs


def main():
    with open(DICTIONARY_PATH, "r", encoding="utf-8") as f:
        dictionary = json.load(f)

    train_set = get_glosses(TRAIN_CSV, DATA_ROOT)
    val_set = get_glosses(VAL_CSV, DATA_ROOT)
    test_set = get_glosses(TEST_CSV, DATA_ROOT)  # only needed by create_dataloaders, never evaluated

    configs = build_configs()
    print(f"Running {len(configs)} configs\n")

    fieldnames = ["changed_param", "value", *BASELINE.keys(),
                  "best_val_acc", "train_acc_at_best", "best_epoch", "epochs_run", "minutes"]
    results = []
    overall_best = 0.0

    with open(RESULTS_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for i, (param, value, cfg) in enumerate(configs, 1):
            print(f"[{i}/{len(configs)}] {param} = {value}")
            start = time.time()
            try:
                val_acc, train_acc, best_epoch, epochs_run, state = run_config(
                    cfg, train_set, val_set, test_set, dictionary
                )
            except Exception as e:  # e.g. one bad config should not kill the whole sweep
                print(f"   FAILED: {e}\n")
                continue
            minutes = (time.time() - start) / 60

            print(f"   val acc {val_acc:.2%} (train {train_acc:.2%}) "
                  f"best epoch {best_epoch}/{epochs_run}, {minutes:.1f} min\n")

            row = {"changed_param": param, "value": value, **cfg,
                   "best_val_acc": round(val_acc, 4), "train_acc_at_best": round(train_acc, 4),
                   "best_epoch": best_epoch, "epochs_run": epochs_run, "minutes": round(minutes, 1)}
            writer.writerow(row)
            f.flush()
            results.append(row)

            if val_acc > overall_best:
                overall_best = val_acc
                torch.save({"config": cfg, "state_dict": state}, BEST_MODEL_PATH)

    print("=" * 60)
    print("Top 5 configs by validation accuracy:")
    for r in sorted(results, key=lambda r: r["best_val_acc"], reverse=True)[:5]:
        print(f"  {r['best_val_acc']:.2%}  changed {r['changed_param']}={r['value']}")
    print(f"\nFull results: {RESULTS_CSV}")


if __name__ == "__main__":
    main()
