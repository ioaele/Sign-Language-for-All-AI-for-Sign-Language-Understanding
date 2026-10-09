#move to code to run
#python3 3frames_eda.py
#chatgpt code
import os
import csv
import numpy as np
from tqdm import tqdm


# --------------------------------------------------
# Read CSV
# --------------------------------------------------

def get_glosses(csv_path):
    glosses = []

    with open(csv_path, encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="|")

        for row in reader:
            if len(row) >= 2:
                path = row[0].strip()
                label = row[1].strip()

                glosses.append((path, label))

    return glosses


# --------------------------------------------------
# Check minimum number of frames
# --------------------------------------------------

def get_valid_glosses(csv_path, min_frames=3):

    glosses = get_glosses(csv_path)
    valid = []

    for path, label in tqdm(
        glosses,
        desc=f"Checking {os.path.basename(csv_path)}"
    ):

        # The CSV contains:
        # police5_signer3_rep4_glosses/glosses0047
        #
        # Actual file is:
        # processed_landmarks/police5_signer3_rep4_glosses/glosses0047.npy

        npy_path = os.path.join(
            "processed_landmarks",
            path + ".npy"
        )

        try:
            landmarks = np.load(npy_path)

        except Exception as e:
            print(f"\nERROR loading: {npy_path}")
            print(e)
            continue

        # Number of frames
        num_frames = landmarks.shape[0]

        # Keep samples with >= 3 frames
        if num_frames >= min_frames:
            valid.append((path, label))

    # --------------------------------------------------
    # Statistics
    # --------------------------------------------------

    total = len(glosses)
    remaining = len(valid)
    eliminated = total - remaining

    print("\n" + "=" * 60)
    print(f"File: {os.path.basename(csv_path)}")
    print(f"Total gloss instances: {total}")
    print(f"Remaining (>= {min_frames} frames): {remaining}")
    print(f"Eliminated (< {min_frames} frames): {eliminated}")

    if total > 0:
        print(f"Remaining percentage: {remaining / total * 100:.2f}%")
        print(f"Eliminated percentage: {eliminated / total * 100:.2f}%")

    print("=" * 60)

    return valid


# --------------------------------------------------
# MAIN
# --------------------------------------------------

if __name__ == "__main__":

    MIN_FRAMES = 3

    csv_files = [
        "data/train_greek_iso_merged.csv",
        "data/val_greek_iso_merged.csv",
        "data/test_greek_iso_merged.csv"
    ]

    for csv_path in csv_files:

        if os.path.exists(csv_path):

            get_valid_glosses(
                csv_path,
                min_frames=MIN_FRAMES
            )

        else:
            print(f"\nCSV file not found: {csv_path}")
