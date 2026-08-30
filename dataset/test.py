#claude test that the pytorch classes are working

# python3 -m dataset.test

import json
from dataset.pytorch_dataset import create_dataloaders
from preprocessing.prepare_dataset import get_valid_glosses

# --- Βήμα 1: φόρτωσε ό,τι χρειάζεται για να φτιάξεις τα splits ---
root = "processed_landmarks"
MIN_FRAMES = 3

train_valid = get_valid_glosses("data/train_greek_iso.csv",root,MIN_FRAMES)
val_valid  = get_valid_glosses("data/val_greek_iso.csv",root,MIN_FRAMES)
test_valid = get_valid_glosses("data/test_greek_iso.csv",root,MIN_FRAMES)

# --- Βήμα 2: φόρτωσε το label_to_id dictionary ---
with open("dataset/dictionary.json", "r", encoding="utf-8") as f:
    dictionary = json.load(f)

# --- Βήμα 3: φτιάξε τα DataLoaders ---
train_loader, val_loader, test_loader = create_dataloaders(
    train_valid, val_valid, test_valid, dictionary, max_length=30, batch_size=32
)

# --- Βήμα 4: ΕΛΕΓΧΟΙ ---

print(f"Train batches: {len(train_loader)}")
print(f"Val batches:   {len(val_loader)}")
print(f"Test batches:  {len(test_loader)}")

# πάρε ΕΝΑ batch από το train_loader, δες τι επιστρέφει
batch_keypoints, batch_labels = next(iter(train_loader))

print(f"\nShape ενός batch keypoints: {batch_keypoints.shape}")
print(f"Shape ενός batch labels: {batch_labels.shape}")

print(f"\nΤύπος δεδομένων keypoints: {batch_keypoints.dtype}")
print(f"Τύπος δεδομένων labels: {batch_labels.dtype}")

print(f"\nΠαράδειγμα label_id ενός δείγματος: {batch_labels[0].item()}")

# sanity check -- η τιμή min/max των keypoints πρέπει να είναι λογική
# (μικροί αριθμοί, όχι τεράστιοι, αφού είναι κανονικοποιημένα)
print(f"\nMin value στο batch: {batch_keypoints.min().item():.2f}")
print(f"Max value στο batch: {batch_keypoints.max().item():.2f}")