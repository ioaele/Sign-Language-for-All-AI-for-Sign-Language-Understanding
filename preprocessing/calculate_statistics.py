# claude code to see the statistics max min duration,mean etc of the glosses
# to see what the same duration of all them must be.
# they all need to have the same duration because the transformer/deep 
# model always must have the same type of input
import os
import numpy as np

processed_root = "processed_landmarks"

all_lengths = []

for session_folder in os.listdir(processed_root):
    session_path = os.path.join(processed_root, session_folder)
    if not os.path.isdir(session_path):
        continue

    for gloss_file in os.listdir(session_path):
        if gloss_file.endswith(".npy"):
            keypoints = np.load(os.path.join(session_path, gloss_file))
            all_lengths.append(keypoints.shape[0])  # το πρώτο νούμερο = πλήθος frames

all_lengths = np.array(all_lengths)

print(f"Σύνολο glosses: {len(all_lengths)}")
print(f"Mean: {all_lengths.mean():.1f}")
print(f"Median: {np.median(all_lengths):.1f}")
print(f"Min: {all_lengths.min()}")
print(f"Max: {all_lengths.max()}")
print(f"95th percentile: {np.percentile(all_lengths, 95):.1f}")