#chatgpt code
import os
import glob
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# SETTINGS
# ============================================================

DATA_DIR = "processed_landmarks"

MAX_LENGTHS = [
    20, 25, 30, 35, 40, 45,
    50, 60, 70, 80, 90, 100,
    120, 150
]

FPS = 30


# ============================================================
# 1. LOAD SEQUENCE LENGTHS
# ============================================================

npy_files = glob.glob(
    os.path.join(DATA_DIR, "**", "*.npy"),
    recursive=True
)

print(f"Found {len(npy_files)} .npy files")

sequence_lengths = []

for file in npy_files:

    try:
        data = np.load(file, allow_pickle=True)

        num_frames = data.shape[0]

        sequence_lengths.append(num_frames)

    except Exception as e:
        print(f"Could not read {file}: {e}")


sequence_lengths = np.array(sequence_lengths)

print(f"Successfully analyzed {len(sequence_lengths)} sequences")


# ============================================================
# 2. CALCULATE METRICS
# ============================================================

total_sequences = len(sequence_lengths)
total_original_frames = np.sum(sequence_lengths)

results = []


for max_length in MAX_LENGTHS:

    # --------------------------------------------------------
    # Number of truncated sequences
    # --------------------------------------------------------

    truncated = np.sum(sequence_lengths > max_length)

    truncated_percentage = (
        truncated / total_sequences
    ) * 100


    # --------------------------------------------------------
    # Number of sequences with NO truncation
    # --------------------------------------------------------

    retained_sequences = np.sum(
        sequence_lengths <= max_length
    )

    retained_sequence_percentage = (
        retained_sequences / total_sequences
    ) * 100


    # --------------------------------------------------------
    # Frames kept after truncation
    # --------------------------------------------------------

    frames_kept = np.minimum(
        sequence_lengths,
        max_length
    )

    total_frames_kept = np.sum(frames_kept)

    information_retained = (
        total_frames_kept /
        total_original_frames
    ) * 100


    # --------------------------------------------------------
    # Padding
    # --------------------------------------------------------

    padding_frames = np.maximum(
        max_length - sequence_lengths,
        0
    )

    total_padding = np.sum(padding_frames)

    total_input_frames = (
        total_sequences * max_length
    )

    padding_percentage = (
        total_padding /
        total_input_frames
    ) * 100


    # --------------------------------------------------------
    # Average padding per sequence
    # --------------------------------------------------------

    average_padding = np.mean(
        padding_frames
    )


    # --------------------------------------------------------
    # Average sequence length
    # --------------------------------------------------------

    average_length = np.mean(
        sequence_lengths
    )


    results.append({
        "max_length": max_length,
        "duration": max_length / FPS,
        "truncated_%": truncated_percentage,
        "fully_retained_%": retained_sequence_percentage,
        "information_retained_%": information_retained,
        "padding_%": padding_percentage,
        "average_padding": average_padding
    })


# ============================================================
# 3. PRINT RESULTS
# ============================================================

print("\n")
print("=" * 100)
print("MAX LENGTH ANALYSIS")
print("=" * 100)

print(
    f"{'Max':>6} "
    f"{'Time':>8} "
    f"{'Trunc %':>10} "
    f"{'Full %':>10} "
    f"{'Info %':>10} "
    f"{'Padding %':>12} "
    f"{'Avg Pad':>10}"
)

print("-" * 100)


for r in results:

    print(
        f"{r['max_length']:>6} "
        f"{r['duration']:>7.2f}s "
        f"{r['truncated_%']:>9.2f}% "
        f"{r['fully_retained_%']:>9.2f}% "
        f"{r['information_retained_%']:>9.2f}% "
        f"{r['padding_%']:>11.2f}% "
        f"{r['average_padding']:>9.2f}"
    )


# ============================================================
# 4. GRAPH: TRUNCATION VS PADDING
# ============================================================

max_lengths = [
    r["max_length"] for r in results
]

truncation = [
    r["truncated_%"] for r in results
]

padding = [
    r["padding_%"] for r in results
]


plt.figure(figsize=(10, 6))

plt.plot(
    max_lengths,
    truncation,
    marker="o",
    label="Truncated sequences (%)"
)

plt.plot(
    max_lengths,
    padding,
    marker="o",
    label="Padding (%)"
)

plt.xlabel("Maximum sequence length (frames)")
plt.ylabel("Percentage (%)")

plt.title(
    "Trade-off Between Truncation and Padding"
)

plt.legend()

plt.grid(True, alpha=0.3)

plt.xticks(max_lengths)

plt.tight_layout()

plt.savefig(
    "max_length_truncation_vs_padding.png",
    dpi=300
)

plt.show()


# ============================================================
# 5. GRAPH: INFORMATION RETENTION
# ============================================================

information_retained = [
    r["information_retained_%"]
    for r in results
]


plt.figure(figsize=(10, 6))

plt.plot(
    max_lengths,
    information_retained,
    marker="o"
)

plt.xlabel("Maximum sequence length (frames)")
plt.ylabel("Original information retained (%)")

plt.title(
    "Information Retained for Different Maximum Sequence Lengths"
)

plt.grid(True, alpha=0.3)

plt.xticks(max_lengths)

plt.tight_layout()

plt.savefig(
    "max_length_information_retained.png",
    dpi=300
)

plt.show()


# ============================================================
# 6. GRAPH: PADDING ONLY
# ============================================================

plt.figure(figsize=(10, 6))

plt.plot(
    max_lengths,
    padding,
    marker="o"
)

plt.xlabel("Maximum sequence length (frames)")
plt.ylabel("Padding (%)")

plt.title(
    "Padding Percentage for Different Maximum Sequence Lengths"
)

plt.grid(True, alpha=0.3)

plt.xticks(max_lengths)

plt.tight_layout()

plt.savefig(
    "max_length_padding.png",
    dpi=300
)

plt.show()