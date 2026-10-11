import os
import csv
import numpy as np
from tqdm import tqdm

#metatropi twn csv files se lista path,label 
def get_glosses(csv_path):
    glosses = []
    with open(csv_path, encoding="utf-8") as f:
        read = csv.reader(f, delimiter="|")
        for row in read:
            folder_path = row[0].strip()   #RELATIVE PATH
            label = row[1].strip()        
            glosses.append((folder_path, label))
    return glosses


#vasi tou eda polla glosses exoyn <3 frames opote filtraroume ta dedomena mas gia na pernoun >=3frames glosses
# den evala statika min frames=3 gia na einai pio dynamic kai flexible
def get_valid_glosses(csv_path, root, min_frames):
    valid = []

    glosses = get_glosses(csv_path)

    for i, (npy_path, label) in enumerate(tqdm(glosses, desc=f"Validating {os.path.basename(csv_path)}")):

        path = os.path.join(root, npy_path + ".npy")

        try:
            landmarks = np.load(path)

        except Exception as e: #an arxeio adio (0B) do not crash and continue 
            continue

        if landmarks.shape[0] >= min_frames: #to shape sou epistrefei to metadata tou npy
#             # (number of frames,dots/landmarks/point in each frame,how many dimensions is each landmark)
            valid.append((npy_path, label))

    return valid


#GIA NA MPORO NA KALW LOAD TA .NPY FILES I NEED THE FULL PATH NOT THE REGULAR THAT IS IN THE CSV FILES



def relative_to_full_gloss_paths(glosses, root):
    return [(str(root / f"{relative_path}.npy"), label)for relative_path, label in glosses]
