import torch
import torch.nn as nn 
from transformer import ISLR_Transformer  


model = ISLR_Transformer()

# Δημιουργούμε προσωρινά τυχαία δεδομένα για τη δοκιμή
# x = torch.randn(32, 40, 292)

# # Περνάμε τα δεδομένα μέσα από το μοντέλο
# y = model(x)

# print("Input shape:", x.shape)
# print("Output shape:", y.shape)
# print("Positional encoding shape:",model.positional_encoding.pe.shape)

# x = torch.randn(4, 40, 292)
# not_padding_frames = torch.tensor([25, 32, 18, 40])

# y = model(x, not_padding_frames)

# print("Input shape:", x.shape)
# print("Output shape:", y.shape)
# model = ISLR_Transformer()

# x = torch.randn(4, 40, 292)

# valid_lengths = torch.tensor([25, 32, 18, 40])

# logits = model(x, valid_lengths)

# print("Input shape:", x.shape)
# print("Output shape:", logits.shape)
# print("Predicted class IDs:", logits.argmax(dim=1))



# configs = [
#     {"d_model": 128, "num_heads": 4, "encoder_layers": 2},
#     {"d_model": 128, "num_heads": 4, "encoder_layers": 4},
#     {"d_model": 256, "num_heads": 8, "encoder_layers": 2},
# ]

# for config in configs:
#     model = ISLR_Transformer(**config)
#     print(config)
#     print("Parameters:", sum(
#         p.numel() for p in model.parameters()
#         if p.requires_grad
#     ))

# import json
# import pandas as pd
# with open("dataset/dictionary.json", "r", encoding="utf-8") as f:
#     gloss_to_id = json.load(f)

# print("Number of glosses:", len(gloss_to_id))
# print("Minimum ID:", min(gloss_to_id.values()))
# print("Maximum ID:", max(gloss_to_id.values()))
# print("ΓΕΙΑ:", gloss_to_id["ΓΕΙΑ"])


# assert len(gloss_to_id) == 305
# assert sorted(gloss_to_id.values()) == list(range(305))

# print("Dictionary is valid!")



# csv_files = {
#     "train": "data/train_greek_iso_merged.csv",
#     "val": "data/val_greek_iso_merged.csv",
#     "test": "data/test_greek_iso_merged.csv"
# }

# for split, csv_path in csv_files.items():

#     df = pd.read_csv(
#         csv_path,
#         sep="|",
#         header=None,
#         names=["path", "label"]
#     )

#     df["label"] = df["label"].str.strip()

#     missing_glosses = sorted(
#         set(df["label"]) - set(gloss_to_id)
#     )

#     print(f"\n{split.upper()}")
#     print("Number of samples:", len(df))
#     print("Unique glosses:", df["label"].nunique())
#     print("Glosses missing from dictionary:", missing_glosses)



import json
from preprocessing.prepare_dataset import get_valid_glosses
from dataset.dataset import create_dataloaders

with open("dataset/dictionary.json", "r", encoding="utf-8") as f:
    dictionary = json.load(f)

print("Number of glosses:", len(dictionary))
print("ID of ΓΕΙΑ:", dictionary["ΓΕΙΑ"])

import pandas as pd

train_valid = load_glosses("data/train_greek_iso_merged.csv")
val_valid = load_glosses("data/val_greek_iso_merged.csv")
test_valid = load_glosses("data/test_greek_iso_merged.csv")



print("Train samples:", len(train_valid))
print("Validation samples:", len(val_valid))
print("Test samples:", len(test_valid))

print("First training sample:", train_valid[0])


train_loader, val_loader, test_loader = create_dataloaders(
    train_valid,
    val_valid,
    test_valid,
    dictionary,
    max_length=40,
    batch_size=32
)

landmarks, labels, lengths = next(iter(train_loader))

print("Landmarks:", landmarks.shape)
print("Labels:", labels.shape)
print("Lengths:", lengths.shape)