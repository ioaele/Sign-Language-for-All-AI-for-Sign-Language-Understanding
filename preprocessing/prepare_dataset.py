#claude script
import os
import csv
import numpy as np

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

#prepei na filtraroume apo to csv file poia glosses exoume emeis (epeksergasmena/normalized)
def get_available_glosses(glosses,root):
    available = []
    for folder_path, label in glosses:
        npy_path = os.path.join(root, folder_path + ".npy") #TO ACTUAL PATH OXI TO RELATIVE PATH POU IPARXEI STO CSV
        if os.path.exists(npy_path):
            available.append((npy_path, label))
    return available


#epidi kapoia glosses exoun 1 kai 2 frames pou den einai efikto realistika 
# na exoume to gloss kai na vroume me to media pipe to soma klp ta afairoume apo ta deigmata mas
# den evala statika min frames=3 gia na einai pio dynamic kai flexible
def get_valid_glosses(csv_path,root,min_frames):
    valid = []
    
    all=get_glosses(csv_path)
    available=get_available_glosses(all,root)
    
 
    for npy_path, label in available:
        landmarks = np.load(npy_path)          
        if landmarks.shape[0] >= min_frames: #to shape sou epistrefei to metadata tou npy
            # (number of frames,dots/landmarks/point in each frame,how many dimensions is each landmark)
            valid.append((npy_path, label))

    return valid

