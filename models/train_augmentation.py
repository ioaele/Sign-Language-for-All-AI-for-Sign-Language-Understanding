import json
import time
import math

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from models.transformer import ISLRTransformer
from dataset.GSLDataset_augmentation import create_dataloaders
from preprocessing.prepare_dataset import get_valid_glosses


#configurations

NUM_EPOCHS = 70          # max epochs
WARMUP_EPOCHS = 5        # 5 prota epochs - to LR anevainei siga siga 
PATIENCE = 15            # an den beltiothei to val accuracy gia 15 epoxes, stamataei
BATCH_SIZE = 32          # posa videos mazi se kathe "vima" ekpaideusis
MAX_LENGTH = 30          # max frames ana video
MIN_FRAMES = 3           # agnoise videos me ligotera apo 3 frames
ROOT = "processed_landmarks"

BEST_MODEL_PATH = "islr_v2_best.pt"  # pou apothikeuetai to kalytero montelo

NUM_LAYERS = 4
HIDDEN_DIM = 256
NUM_HEADS = 8
DROPOUT = 0.1
LR = 2e-4                # arxikos rithmos mathisis (learning rate)
WEIGHT_DECAY = 1e-5      # mikri "poinh" gia na min ginontai ta varh poly megala
USE_MASK = True
POOLING = "class_query"  # "class_query" (neo) i "mean" (palio, gia sigkrisi)


# ============================================================
# 1. FORTONOUME TA DEDOMENA (train set exei augmentation)
# ============================================================

print("Φορτώνουμε τα δεδομένα...")

train_valid = get_valid_glosses("data/train_greek_iso.csv", ROOT, MIN_FRAMES)
val_valid = get_valid_glosses("data/val_greek_iso.csv", ROOT, MIN_FRAMES)
test_valid = get_valid_glosses("data/test_greek_iso.csv", ROOT, MIN_FRAMES)

with open("dataset/dictionary.json", "r", encoding="utf-8") as f:
    dictionary = json.load(f)

id_to_label = {v: k for k, v in dictionary.items()}

train_loader, val_loader, test_loader = create_dataloaders(
    train_valid,
    val_valid,
    test_valid,
    dictionary,
    max_length=MAX_LENGTH,
    batch_size=BATCH_SIZE
)

num_classes = len(dictionary)

# an yparxei Mac GPU (mps) allios CPU
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

print(f"Χρησιμοποιείται: {device}")
print(f"Αριθμός κλάσεων: {num_classes}")
print(f"Train/Val/Test μεγέθη: {len(train_valid)}/{len(val_valid)}/{len(test_valid)}")



# FTIAXNOUME TO MONTELO / OPTIMIZER / SCHEDULER / LOSS
model = ISLRTransformer(
    input_dim=292,
    hidden_dim=HIDDEN_DIM,
    num_heads=NUM_HEADS,
    num_layers=NUM_LAYERS,
    num_classes=num_classes,
    max_length=MAX_LENGTH,
    dropout=DROPOUT,
    pooling=POOLING
).to(device)

# o optimizer einai auto pou allazei ta varh tou montelou vasei tou loss
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LR,
    weight_decay=WEIGHT_DECAY
)


def lr_lambda(epoch):
    """
    Kontrolarei to LR (posso "megala vimata" kanei to montelo otan
    mathainei) se kathe epoxi:
    - protes WARMUP_EPOCHS epoxes: to LR anevainei siga siga apo 0 sto
      kanoniko (gia na min "petaxtei" apotoma sto ksekinima)
    - meta: peftei omala (cosine curve) mexri to telos
    """

    if epoch < WARMUP_EPOCHS:
        return (epoch + 1) / WARMUP_EPOCHS

    progress = (epoch - WARMUP_EPOCHS) / max(1, (NUM_EPOCHS - WARMUP_EPOCHS))
    return 0.5 * (1 + math.cos(math.pi * progress))


scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

# to "loss" metraei poso lathos exei to montelo. label_smoothing
# kanei to montelo na min einai UPERBOLIKA sigouro gia tis apantiseis tou
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)


#TRAINING LOOP (me early stopping)
history = {"train_loss": [], "train_acc": [], "val_acc": []}

best_val_accuracy = 0.0
best_epoch = -1
epochs_without_improvement = 0

start_time = time.time()

for epoch in range(NUM_EPOCHS):

    #train
    model.train()
    total_train_loss = 0.0
    correct_train = 0
    total_train = 0
    skipped_batches = 0

    for batch_keypoints, batch_labels, batch_lengths in train_loader:
        batch_keypoints = batch_keypoints.to(device)
        batch_labels = batch_labels.to(device)
        batch_lengths = batch_lengths.to(device)

        optimizer.zero_grad()  # midenizoume ta palia "gradients" prin to neo vima

        logits = model(batch_keypoints, batch_lengths, use_mask=USE_MASK)
        loss = criterion(logits, batch_labels)

        # asfaleia: an to loss vgei NaN/Inf (kati "espase" arithmitika),
        # agnoisame AUTO to batch, gia na min xalasei olo to montelo
        if not torch.isfinite(loss):
            skipped_batches += 1
            optimizer.zero_grad()
            continue

        loss.backward()  # ipologizei pos na allaxoun ta varh (gradients)

        # asfaleia #2: elegxoume kai ta idia ta gradients gia NaN/Inf
        # (mporei to loss na einai OK alla ta gradients oxi)
        grad_ok = all(
            torch.isfinite(p.grad).all()
            for p in model.parameters()
            if p.grad is not None
        )
        if not grad_ok:
            skipped_batches += 1
            optimizer.zero_grad()
            continue

        # asfaleia #3: "koftoume" ta gradients an ginoun poly megala
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()  # edo ginetai i pragmatiki enimerosi ton varon

        total_train_loss += loss.item()
        predictions = logits.argmax(dim=1)
        correct_train += (predictions == batch_labels).sum().item()
        total_train += batch_labels.size(0)

    train_accuracy = correct_train / total_train if total_train > 0 else 0.0
    avg_train_loss = total_train_loss / max(1, (len(train_loader) - skipped_batches))

    # ---------------- VALIDATE (elegxos, xoris na mathainei) ----------------
    model.eval()
    correct_val = 0
    total_val = 0

    with torch.no_grad():  # den xreiazomaste gradients edo -- pio grigoro
        for batch_keypoints, batch_labels, batch_lengths in val_loader:
            batch_keypoints = batch_keypoints.to(device)
            batch_labels = batch_labels.to(device)
            batch_lengths = batch_lengths.to(device)

            logits = model(batch_keypoints, batch_lengths, use_mask=USE_MASK)
            predictions = logits.argmax(dim=1)

            correct_val += (predictions == batch_labels).sum().item()
            total_val += batch_labels.size(0)

    val_accuracy = correct_val / total_val

    scheduler.step()  # proxoraei to LR sto epomeno vima tou schedule

    history["train_loss"].append(avg_train_loss)
    history["train_acc"].append(train_accuracy)
    history["val_acc"].append(val_accuracy)

    # an auti i epoxi einai i kalyteri mexri stigmis, apothikeuoume to montelo
    is_best = val_accuracy > best_val_accuracy
    if is_best:
        best_val_accuracy = val_accuracy
        best_epoch = epoch + 1
        epochs_without_improvement = 0
        torch.save(model.state_dict(), BEST_MODEL_PATH)
    else:
        epochs_without_improvement += 1

    current_lr = optimizer.param_groups[0]["lr"]
    marker = " ← ΝΕΟ ΚΑΛΥΤΕΡΟ" if is_best else ""

    print(
        f"Epoch {epoch + 1}/{NUM_EPOCHS} -- "
        f"LR: {current_lr:.8f}, "
        f"Train loss: {avg_train_loss:.4f}, "
        f"Train acc: {train_accuracy:.2%}, "
        f"Val acc: {val_accuracy:.2%}"
        f"{marker}"
    )

    # early stopping: an den beltionomaste gia poly kairo, stamataei
    if epochs_without_improvement >= PATIENCE:
        print(f"\nΔεν βελτιώθηκε η val accuracy για {PATIENCE} epochs. Σταματάμε στο epoch {epoch + 1}.")
        break

training_time = time.time() - start_time

print(f"\nΚαλύτερο val accuracy: {best_val_accuracy:.2%} (epoch {best_epoch})")
print(f"Συνολικός χρόνος εκπαίδευσης: {training_time / 60:.1f} λεπτά")


# ============================================================
# 4. GRAFIKI PARASTASI TIS EKPAIDEUSIS
# ============================================================

epochs_range = range(1, len(history["train_acc"]) + 1)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

axes[0].plot(epochs_range, [a * 100 for a in history["train_acc"]], label="Train accuracy")
axes[0].plot(epochs_range, [a * 100 for a in history["val_acc"]], label="Validation accuracy")
axes[0].axvline(best_epoch, color="gray", linestyle="--", alpha=0.6, label=f"Best epoch ({best_epoch})")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Accuracy (%)")
axes[0].set_title("Training / Validation Accuracy (v2)")
axes[0].legend()
axes[0].grid(alpha=0.3)

axes[1].plot(epochs_range, history["train_loss"], color="tab:red", label="Train loss")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Loss")
axes[1].set_title("Training Loss (v2)")
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig("training_curves_v2.png", dpi=150)
plt.close()
print("Αποθηκεύτηκε: training_curves_v2.png")


# ============================================================
# 5. FORTONOUME TO KALYTERO MONTELO KAI TO DOKIMAZOUME STO TEST SET
# ============================================================

model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=device))
model.eval()

all_preds = []
all_labels = []

with torch.no_grad():
    for batch_keypoints, batch_labels, batch_lengths in test_loader:
        batch_keypoints = batch_keypoints.to(device)
        batch_lengths = batch_lengths.to(device)

        logits = model(batch_keypoints, batch_lengths, use_mask=USE_MASK)
        predictions = logits.argmax(dim=1)

        all_preds.extend(predictions.cpu().numpy().tolist())
        all_labels.extend(batch_labels.numpy().tolist())

all_preds = np.array(all_preds)
all_labels = np.array(all_labels)

test_accuracy = (all_preds == all_labels).mean()

print(f"\n{'=' * 60}")
print(f"TEST SET ACCURACY (v2): {test_accuracy:.2%}")
print(f"{'=' * 60}")


# ============================================================
# 6. ANAFORA ANA KATIGORIA (classification report)
# ============================================================

try:
    from sklearn.metrics import classification_report, confusion_matrix
    import csv

    present_classes = sorted(set(all_labels.tolist()) | set(all_preds.tolist()))
    target_names = [id_to_label[c] for c in present_classes]

    report = classification_report(
        all_labels, all_preds,
        labels=present_classes, target_names=target_names,
        digits=4, zero_division=0
    )

    with open("test_classification_report_v2.txt", "w", encoding="utf-8") as f:
        f.write(f"Test accuracy: {test_accuracy:.4f}\n\n")
        f.write(report)

    # accuracy gia KATHE noima xexoristo (poia noimata mathainei kala/asxima)
    per_class_rows = []
    for c in present_classes:
        mask = all_labels == c
        n = mask.sum()
        acc = (all_preds[mask] == c).mean() if n > 0 else float("nan")
        per_class_rows.append({
            "class_id": c, "gloss": id_to_label[c],
            "n_test_samples": int(n),
            "accuracy": round(float(acc), 4) if n > 0 else ""
        })

    per_class_rows.sort(key=lambda r: (r["accuracy"] if r["accuracy"] != "" else 2.0))

    with open("per_class_accuracy_v2.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["class_id", "gloss", "n_test_samples", "accuracy"])
        writer.writeheader()
        writer.writerows(per_class_rows)

    # confusion matrix: dixnei POU mperdeuetai to montelo (p.x. noima A -> nomizei B)
    cm = confusion_matrix(all_labels, all_preds, labels=present_classes)

    plt.figure(figsize=(10, 10))
    plt.imshow(cm, cmap="viridis")
    plt.title(f"Confusion Matrix v2 ({len(present_classes)} classes) -- Test Set")
    plt.xlabel("Predicted class")
    plt.ylabel("True class")
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig("confusion_matrix_v2.png", dpi=150)
    plt.close()

    # top 30 pio "mperdemenoi" zeugi noimaton
    confused_pairs = []
    for i, true_c in enumerate(present_classes):
        for j, pred_c in enumerate(present_classes):
            if i != j and cm[i, j] > 0:
                confused_pairs.append((cm[i, j], id_to_label[true_c], id_to_label[pred_c]))
    confused_pairs.sort(reverse=True)

    with open("top_confused_pairs_v2.txt", "w", encoding="utf-8") as f:
        f.write("Top 30 most confused (true -> predicted) gloss pairs on the test set (v2):\n\n")
        for count, true_label, pred_label in confused_pairs[:30]:
            f.write(f"{count:4d}x   {true_label}  ->  {pred_label}\n")

    print("Αποθηκεύτηκαν: classification report / per-class accuracy / confusion matrix / top confused pairs")

except ImportError:
    print("[Προειδοποίηση] Λείπει το scikit-learn -- παραλείπεται η αναλυτική αναφορά.")


# ============================================================
# 7. TELIKI PERILIPSI SE JSON
# ============================================================

summary = {
    "hyperparameters": {
        "num_layers": NUM_LAYERS, "hidden_dim": HIDDEN_DIM, "num_heads": NUM_HEADS,
        "dropout": DROPOUT, "lr": LR, "weight_decay": WEIGHT_DECAY,
        "scheduler": "warmup+cosine", "warmup_epochs": WARMUP_EPOCHS,
        "use_mask": USE_MASK, "pooling": POOLING, "augmentation": True,
        "batch_size": BATCH_SIZE, "max_length": MAX_LENGTH,
    },
    "training": {
        "epochs_run": len(history["train_acc"]), "epochs_budget": NUM_EPOCHS,
        "best_epoch": best_epoch, "best_val_accuracy": round(best_val_accuracy, 4),
        "training_time_minutes": round(training_time / 60, 1),
    },
    "test_accuracy": round(float(test_accuracy), 4),
    "num_classes": num_classes,
    "dataset_sizes": {"train": len(train_valid), "val": len(val_valid), "test": len(test_valid)},
    "compare_to_v1": {
        "v1_test_accuracy": 0.8420,
        "note": "v1 = mean pooling, no augmentation, plain cosine (no warmup)"
    }
}

with open("final_results_summary_v2.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)

print("\nΑποθηκεύτηκε: final_results_summary_v2.json")
print("Τέλος.")