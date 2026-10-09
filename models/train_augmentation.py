# cd "/Users/ioaele/Desktop/thesis/Sign-Language-for-All-AI-for-Sign-Language-Understanding/code"
# python3 -m models.train_augmentation
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
from dataset.dataset_augmentation import create_dataloaders
from preprocessing.prepare_dataset import get_valid_glosses


#configurations
NUM_EPOCHS = 70 #loop for running the model
WARMUP_EPOCHS = 5   #the model at the first WARMUP_EPOCHS mathainei arga 
PATIENCE = 15  #an perasoun PATIENCE epochs xoris kalyteri epidosi stamataei 
BATCH_SIZE = 32 #posa glosses mpenoun sto modelo
MAX_LENGTH = 30   #max frame length 
MIN_FRAMES = 3   #min frame length 
ROOT = "processed_landmarks"  

BEST_MODEL_PATH = "islr_augmentation.pt" 

NUM_LAYERS = 4 #transformer layers
HIDDEN_DIM = 256  # transform the input numbers/landmarks to hidden_dim 
NUM_HEADS = 8  #attention heads to prosexei se diaforetikes sxeseis
DROPOUT = 0.1  # pososto pou "svinei" tyxaia to montelo kata to training - to avoid overfitting
LR = 2e-4  #learning rate: poso "megalo vima" kanei to montelo se kathe diorthosi - mikro = pio argo alla pio stathero mathisi
WEIGHT_DECAY = 1e-5  #to avoid overfitting: mikri "poinh" pou empodizei ta varh na ginoun poly megala
USE_MASK = True    #to montelo agnoei ta padding frames (den ta ypologizei san pragmatika)
POOLING = "class_query"  #pooling method (from paper https://ieeexplore.ieee.org/document/9707552)


train_valid = get_valid_glosses("data/train_greek_iso.csv", ROOT, MIN_FRAMES)
val_valid = get_valid_glosses("data/val_greek_iso.csv", ROOT, MIN_FRAMES)
test_valid = get_valid_glosses("data/test_greek_iso.csv", ROOT, MIN_FRAMES)

with open("dataset/dictionary.json", "r", encoding="utf-8") as f:
    dictionary = json.load(f)

id_to_label = {v: k for k, v in dictionary.items()}

#pernoume ta dedomena mas input ana batch_size
train_loader, val_loader, test_loader = create_dataloaders(train_valid,val_valid,test_valid,dictionary,max_length=MAX_LENGTH,batch_size=BATCH_SIZE)

#posa pithana translations exoume 
num_classes = len(dictionary)

#in what device (gpu or cpu) will the model run
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

#kaloume to transformer
model = ISLRTransformer(input_dim=292,hidden_dim=HIDDEN_DIM,num_heads=NUM_HEADS,num_layers=NUM_LAYERS,
                        num_classes=num_classes,max_length=MAX_LENGTH,dropout=DROPOUT,pooling=POOLING).to(device)

#optimizer: allazei ta varh tou montelou vasei tou loss
optimizer = torch.optim.Adam(model.parameters(),lr=LR,weight_decay=WEIGHT_DECAY)


def lr_schedule(epoch):
#managing learning rate (lr)
    # - prota WARMUP_EPOCHS: to LR anevainei siga siga apo 0 sto
    #   kanoniko (gia na min "petaxtei" apotoma sto ksekinima)
    # - meta: peftei omala (cosine curve) mexri to telos
  
    if epoch < WARMUP_EPOCHS:
        return (epoch + 1) / WARMUP_EPOCHS

    progress = (epoch - WARMUP_EPOCHS) / max(1, (NUM_EPOCHS - WARMUP_EPOCHS))
    return 0.5 * (1 + math.cos(math.pi * progress))


scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_schedule)

#to "loss" metraei poso lathos exei to montelo. label_smoothing
#kanei to montelo na min einai ipervolika sigouro gia tis apantiseis tou
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

        score = model(batch_keypoints, batch_lengths, use_mask=USE_MASK)
        loss = criterion(score, batch_labels)

 #an to loss vgei NaN/Inf ignore it 
        if not torch.isfinite(loss):
            skipped_batches += 1
            optimizer.zero_grad()
            continue

        loss.backward()  # ipologizei pos na allaxoun ta varh (gradients)
        #Το gradient ενός βάρους σου λέει: "αν αυξήσω λίγο αυτό το βάρος, το loss θα πάει πάνω ή κάτω, και πόσο απότομα;"
        #Το backward() κάνει backpropagation — δηλαδή υπολογίζει, για κάθε βάρος στο μοντέλο, πόσο "ευθύνεται" για το λάθος (loss).

        #elegxoume kai ta idia ta gradients gia NaN/Inf
        grad_ok = all(
            torch.isfinite(p.grad).all()
            for p in model.parameters()
            if p.grad is not None
        )
        if not grad_ok:
            skipped_batches += 1
            optimizer.zero_grad()
            continue

        # cut the an ginoun poly megala
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()  #i pragmatiki enimerosi ton varon
        #παίρνει αυτά τα gradients και κινεί κάθε βάρος λίγο προς την κατεύθυνση που μειώνει το loss, 
        #πολλαπλασιασμένο με το learning rate (πόσο μεγάλο είναι το "βήμα").

        total_train_loss += loss.item()
        predictions = score.argmax(dim=1)
        correct_train += (predictions == batch_labels).sum().item()
        total_train += batch_labels.size(0)

    train_accuracy = correct_train / total_train if total_train > 0 else 0.0
    avg_train_loss = total_train_loss / max(1, (len(train_loader) - skipped_batches))

#validation
    model.eval()
    correct_val = 0
    total_val = 0
#same logic as train without backward/optimizer
# #apla metra posa provlepei swsta sto validation set
    with torch.no_grad():  
        for batch_keypoints, batch_labels, batch_lengths in val_loader:
            batch_keypoints = batch_keypoints.to(device)
            batch_labels = batch_labels.to(device)
            batch_lengths = batch_lengths.to(device)

            score = model(batch_keypoints, batch_lengths, use_mask=USE_MASK)
            predictions = score.argmax(dim=1)

            correct_val += (predictions == batch_labels).sum().item()
            total_val += batch_labels.size(0)

    val_accuracy = correct_val / total_val

    scheduler.step()  # proxoraei to LR sto epomeno vima tou schedule

    history["train_loss"].append(avg_train_loss)
    history["train_acc"].append(train_accuracy)
    history["val_acc"].append(val_accuracy)

    #an auto to epoch einai i kalyteri mexri stigmis apothikeuoume to montelo
    #save statistics
    is_best = val_accuracy > best_val_accuracy
    if is_best:
        best_val_accuracy = val_accuracy
        best_epoch = epoch + 1
        epochs_without_improvement = 0
        torch.save(model.state_dict(), BEST_MODEL_PATH)
    else:
        epochs_without_improvement += 1

    current_lr = optimizer.param_groups[0]["lr"]

    print(
        f"Epoch {epoch + 1}/{NUM_EPOCHS}: "
        f"LR: {current_lr:.8f}, "
        f"Train loss: {avg_train_loss:.4f}, "
        f"Train acc: {train_accuracy:.2%}, "
        f"Val acc: {val_accuracy:.2%}"
    )

    # early stopping: an den veltionete to val accuracy mexri PATIENCE epochs stop
    if epochs_without_improvement >= PATIENCE:
        print(f"\nThe val accuracy was not improved for {PATIENCE} epochs. Stopping at epoch {epoch + 1}.")
        break

training_time = time.time() - start_time

print(f"\Best val accuracy: {best_val_accuracy:.2%} (epoch {best_epoch})")
print(f"Total training time: {training_time / 60:.1f} minutes")



#GRAFIKI PARASTASI TIS EKPAIDEUSIS

epochs_range = range(1, len(history["train_acc"]) + 1)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

axes[0].plot(epochs_range, [a * 100 for a in history["train_acc"]], label="Train accuracy")
axes[0].plot(epochs_range, [a * 100 for a in history["val_acc"]], label="Validation accuracy")
axes[0].axvline(best_epoch, color="gray", linestyle="--", alpha=0.6, label=f"Best epoch ({best_epoch})")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Accuracy (%)")
axes[0].set_title("Training / Validation Accuracy")
axes[0].legend()
axes[0].grid(alpha=0.3)

axes[1].plot(epochs_range, history["train_loss"], color="tab:red", label="Train loss")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Loss")
axes[1].set_title("Training Loss")
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig("training_curves_augmentation.png", dpi=150)
plt.close()
print("saved: training_curves_augmentation.png")



# FORTONOUME TO KALYTERO MONTELO KAI TO DOKIMAZOUME STO TEST SET
model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=device))
model.eval()

all_preds = []
all_labels = []

with torch.no_grad():
    for batch_keypoints, batch_labels, batch_lengths in test_loader:
        batch_keypoints = batch_keypoints.to(device)
        batch_lengths = batch_lengths.to(device)

        score = model(batch_keypoints, batch_lengths, use_mask=USE_MASK)
        predictions = score.argmax(dim=1)

        all_preds.extend(predictions.cpu().numpy().tolist())
        all_labels.extend(batch_labels.numpy().tolist())

all_preds = np.array(all_preds)
all_labels = np.array(all_labels)

test_accuracy = (all_preds == all_labels).mean()


print(f"TEST SET ACCURACY: {test_accuracy:.2%}")

# ANAFORA ANA KATIGORIA (classification report)
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

    with open("test_classification_augmentation_report.txt", "w", encoding="utf-8") as f:
        f.write(f"Test accuracy: {test_accuracy:.4f}\n\n")
        f.write(report)

    # accuracy gia kathe gloss xexoristo (poia glosses mathainei kala/asxima)
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

    with open("per_class_accuracy_augmentation.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["class_id", "gloss", "n_test_samples", "accuracy"])
        writer.writeheader()
        writer.writerows(per_class_rows)

    # confusion matrix: dixnei pou mperdeuetai to montelo
    cm = confusion_matrix(all_labels, all_preds, labels=present_classes)

    plt.figure(figsize=(10, 10))
    plt.imshow(cm, cmap="viridis")
    plt.title(f"Confusion Matrix ({len(present_classes)} classes) -- Test Set")
    plt.xlabel("Predicted class")
    plt.ylabel("True class")
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig("confusion_matrix_augmentation.png", dpi=150)
    plt.close()

    # top 30 pio "mperdemena" zeugi glosses
    confused_pairs = []
    for i, true_c in enumerate(present_classes):
        for j, pred_c in enumerate(present_classes):
            if i != j and cm[i, j] > 0:
                confused_pairs.append((cm[i, j], id_to_label[true_c], id_to_label[pred_c]))
    confused_pairs.sort(reverse=True)

    with open("top_confused_pairs_augmentation.txt", "w", encoding="utf-8") as f:
        f.write("Top 30 most confused (true -> predicted) gloss pairs on the test set:\n\n")
        for count, true_label, pred_label in confused_pairs[:30]:
            f.write(f"{count:4d}x   {true_label}  ->  {pred_label}\n")

except ImportError:
  print("sklearn missing, skipping report")
