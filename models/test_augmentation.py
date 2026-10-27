#python3 -m models.test_model_sanity
# Checks that the model's output for a sample does NOT depend on the other samples in the batch,
# on the order of the batch, or on the content of padded frames. Needs no training.
import json
import torch
from pathlib import Path

from dataset.dataset import create_dataloaders
from preprocessing.get_glosses import get_glosses
from models.transformer import ISLR_Transformer

MAX_LENGTH = 40
BATCH_SIZE = 32

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = PROJECT_ROOT / "processed_landmarks"
TRAIN_CSV = PROJECT_ROOT / "data" / "train_greek_iso.csv"
VAL_CSV = PROJECT_ROOT / "data" / "val_greek_iso.csv"
TEST_CSV = PROJECT_ROOT / "data" / "test_greek_iso.csv"
DICTIONARY_PATH = PROJECT_ROOT / "dataset" / "dictionary.json"
MODEL_PATH = PROJECT_ROOT / "islr_transformer_best.pt"

DEVICE = (torch.device("mps") if torch.backends.mps.is_available()
          else torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))

with open(DICTIONARY_PATH, "r", encoding="utf-8") as f:
    dictionary = json.load(f)

train_set = get_glosses(TRAIN_CSV, DATA_ROOT)
val_set = get_glosses(VAL_CSV, DATA_ROOT)
test_set = get_glosses(TEST_CSV, DATA_ROOT)
train_loader, val_loader, _ = create_dataloaders(
    train_set, val_set, test_set, dictionary, max_length=MAX_LENGTH, batch_size=BATCH_SIZE)

lm, lab, npd = next(iter(train_loader))
lm, lab, npd = lm.to(DEVICE), lab.to(DEVICE), npd.to(DEVICE)


def build(trained):
    torch.manual_seed(0)
    m = ISLR_Transformer(num_landmarks=292, max_length=MAX_LENGTH,
                         num_glosses=len(dictionary)).to(DEVICE)
    if trained:
        m.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    m.eval()
    return m


models = [("random init", build(False))]
if MODEL_PATH.exists():
    models.append(("baseline checkpoint", build(True)))

for name, model in models:
    print("\n" + "=" * 78)
    print(f"MODEL: {name}   (tolerance for 'same' = 1e-3 on logits)")
    print("=" * 78)
    with torch.no_grad():
        full = model(lm, npd)

        # A. each sample alone vs inside the batch
        single = torch.cat([model(lm[i:i + 1], npd[i:i + 1]) for i in range(lm.size(0))])
        diff_a = (full - single).abs().max().item()
        agree_a = (full.argmax(1) == single.argmax(1)).float().mean().item()
        print(f"A. batch vs alone:          max |logit diff| = {diff_a:.5f}, argmax agreement = {agree_a:.1%}")

        # B. permute the order of the samples inside the batch
        perm = torch.randperm(lm.size(0), device=DEVICE)
        permuted = model(lm[perm], npd[perm])
        diff_b = (full[perm] - permuted).abs().max().item()
        print(f"B. batch order permuted:    max |logit diff| = {diff_b:.5f}")

        # C. replace the content of padded frames with noise (should not matter)
        lm_noise = lm.clone()
        for i in range(lm.size(0)):
            n = int(npd[i].item())
            if n < lm.size(1):
                lm_noise[i, n:] = torch.randn_like(lm_noise[i, n:])
        noisy = model(lm_noise, npd)
        diff_c = (full - noisy).abs().max().item()
        print(f"C. padded frames -> noise:  max |logit diff| = {diff_c:.5f}")

        # D. does the number of real frames matter? (same data, wrong frame count)
        wrong = model(lm, torch.full_like(npd, lm.size(1)))
        diff_d = (full - wrong).abs().max().item()
        print(f"D. not_padded -> all 40:    max |logit diff| = {diff_d:.5f}  (should be > 0: mask is used)")

print("""
HOW TO READ
- A or B clearly > 1e-3  -> the model mixes information ACROSS SAMPLES in a batch. That is a bug
  (typical cause: nn.TransformerEncoder created without batch_first=True, so attention runs over the
  batch dimension instead of the frames). It would explain why eval results depend on which samples
  happen to be in a batch.
- C clearly > 1e-3       -> padded frames influence the output: the padding mask is wrong or ignored
  (e.g. mask built the wrong way round, or not passed to the encoder).
- D == 0                 -> not_padded_frames is not used at all.
- All of A, B, C ~ 0 and D > 0 -> the model code is fine; the instability is a training-dynamics issue.
""")