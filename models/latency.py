#chat code
import time
import statistics
import json
import torch
from pathlib import Path

from dataset.dataset import create_dataloaders
from preprocessing.get_glosses import get_glosses
from models.transformer import ISLR_Transformer


# =========================
# CONFIGURATION
# =========================

PROJECT_ROOT = Path(__file__).resolve().parent.parent 

MODEL_PATH = PROJECT_ROOT / "model_files" / "islr_transformer_aug_best_config27.pt"

DATA_ROOT = PROJECT_ROOT / "processed_landmarks"
TEST_CSV = PROJECT_ROOT / "data" / "test_greek_iso.csv"
DICTIONARY_PATH = PROJECT_ROOT / "dataset" / "dictionary.json"

MAX_LENGTH = 30
BATCH_SIZE = 32

# Change to "mps" if you want to test on Apple Silicon GPU
DEVICE = torch.device("cpu")

WARMUP_RUNS = 20
MEASURED_RUNS = 100


# =========================
# LOAD DATA AND MODEL
# =========================

print("Device:", DEVICE)

with open(DICTIONARY_PATH, "r", encoding="utf-8") as f:
    dictionary = json.load(f)

test_set = get_glosses(TEST_CSV, DATA_ROOT)

# Reuse your existing data-loading pipeline
_, _, test_loader = create_dataloaders(
    test_set,
    test_set,
    test_set,
    dictionary,
    max_length=MAX_LENGTH,
    batch_size=BATCH_SIZE
)

model = ISLR_Transformer(
    num_landmarks=292,
    max_length=MAX_LENGTH,
    num_glosses=len(dictionary)
).to(DEVICE)

# Load trained weights; no training takes place
state_dict = torch.load(
    MODEL_PATH,
    map_location=DEVICE,
    weights_only=True
)
model.load_state_dict(state_dict)
model.eval()

print("Model loaded:", MODEL_PATH.name)


# =========================
# SYNCHRONIZATION
# =========================

def synchronize():
    if DEVICE.type == "mps":
        torch.mps.synchronize()
    elif DEVICE.type == "cuda":
        torch.cuda.synchronize(DEVICE)


# =========================
# PREPARE ONE TEST SAMPLE
# =========================

landmarks, labels, not_padded_frames = next(iter(test_loader))

# Batch size 1: measure one prediction at a time
landmarks = landmarks[0:1].to(DEVICE)
not_padded_frames = not_padded_frames[0:1].to(DEVICE)

print("Input shape:", landmarks.shape)


# =========================
# WARM-UP
# =========================

with torch.inference_mode():
    for _ in range(WARMUP_RUNS):
        _ = model(landmarks, not_padded_frames)

synchronize()


# =========================
# MEASURE INFERENCE LATENCY
# =========================

latencies_ms = []

with torch.inference_mode():
    for _ in range(MEASURED_RUNS):
        synchronize()
        start = time.perf_counter()

        logits = model(landmarks, not_padded_frames)

        synchronize()
        end = time.perf_counter()

        latencies_ms.append((end - start) * 1000)


# =========================
# RESULTS
# =========================

mean_latency = statistics.mean(latencies_ms)
median_latency = statistics.median(latencies_ms)

print("\n========== LATENCY RESULTS ==========")
print(f"Model: {MODEL_PATH.name}")
print(f"Device: {DEVICE}")
print(f"Warm-up runs: {WARMUP_RUNS}")
print(f"Measured runs: {MEASURED_RUNS}")
print(f"Mean latency: {mean_latency:.3f} ms")
print(f"Median latency: {median_latency:.3f} ms")
print(f"Minimum latency: {min(latencies_ms):.3f} ms")
print(f"Maximum latency: {max(latencies_ms):.3f} ms")
print(f"Throughput estimate: {1000 / mean_latency:.2f} predictions/sec")
print("=====================================")
