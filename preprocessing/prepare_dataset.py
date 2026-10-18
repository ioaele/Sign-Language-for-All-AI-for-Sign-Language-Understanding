import os
import csv
import numpy as np
from tqdm import tqdm

#metatropi twn csv files se lista path,label
#den filtraroume glosses me <3 frames
#GIA NA MPORO NA KALW LOAD TA .NPY FILES I NEED THE FULL PATH NOT THE REGULAR THAT IS IN THE CSV FILES
def get_glosses(csv_path, root):
    glosses = []

    with open(csv_path, encoding="utf-8") as f:
        read = csv.reader(f, delimiter="|")

        for row in read:
            folder_path = row[0].strip()   #RELATIVE PATH
            label = row[1].strip()

            #metatropi relative path se full path gia na mporo na kano load ta .npy arxeia
            full_path = str(root / f"{folder_path}.npy")

            glosses.append((full_path, label))

    return glosses