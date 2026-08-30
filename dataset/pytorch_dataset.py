#https://docs.pytorch.org/tutorials/beginner/basics/data_tutorial.html
# changing to match my project anti gia images emeis exoume keypoints idi apo to normalization twvn frames

import os
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class GSLDataset(Dataset):
    
    def __init__(self, glosses, dictionary, max_length=30):
        self.glosses = glosses # lista (path,label) poy ftiaksame apo to get_available_glosses
        self.dictionary = dictionary  #lista label-id
        self.max_length = max_length      # megisto length gia to stathero input
        
    def __len__(self):
        return len(self.glosses) # returns the number of glosses we have
    
    def __getitem__(self, idx):
        npy_path, label = self.glosses[idx]  #idx gloss path

        landmarks=np.load(npy_path)  #load its landmarks
        landmarks = self.change_length(npy_path)   #change its number of frames to be at max_length

        label_id = self.dictionary[label] #get the id from its label in the dictionary

        return torch.tensor(landmarks, dtype=torch.float32), label_id
     #for a specific index return the landmarks of that gloss and its label id
     
# afou to deep model prepei na exei to idio input type each time prepei na kanoume ta glosses mas idiou mikous
# eite padding to make them bigger eite truncate to make the shorter (in length)
# giafto kai eprepe na doume to meso length ton glosses (posa frames exoun) me thn klassi
# calculate_staticts k apo oti idame:
# Total glosses: 33162
# Mean: 16.9
# Median: 16.0
# Min: 3
# Max: 137
# 95th percentile: 29.0 afto simeni oti mono to 5% einai panw apo 29 frames
# ara to meso length (max_length) ipologizw na einai peripou 30 afou etsi tha kano truncate mono to 5% pou einai poli pio simantiko apo to padding (xanis nohma)

    def change_length(self,npy_path):
            max_length=self.max_length
            landmarks = np.load(npy_path)
            length=landmarks.shape[0]
            # apo to (number of frames,dots/landmarks/point in each frame,how many dimensions is each landmark)
            # we get the first argument
            
            if length>max_length : #concatenate keep only the first max_length frames
                return landmarks[:max_length]
            
            elif length<max_length: #padding
                padding_length = max_length - length # see hoe much padding we need
                padding = np.zeros((padding_length, landmarks.shape[1], landmarks.shape[2])) 
                # create npy table with zeros with the size of padding_length and with the same landmark size and dimensions 
                return np.concatenate([landmarks, padding], axis=0) #enose tous dio "pinakes", axis=0
                    #to add them according to the frames length (proto argument)
            
            else: # 30 length keep it as it is 
                return landmarks
            
      
def create_dataloaders(train_valid, val_valid, test_valid, dictionary, max_length=30, batch_size=32):
# batch_size = posa dedomena tha vlepei/epeksergazete to montelo mazi/taftoxrona

    # ftiaxnoume 3 ksexorista Dataset objects. ena gia to kathe train/val/test
    # to kathena "kseri" mono ta dika toy dedomena alla ola tous xrisimopoioun to idio dictionary 
    train_dataset = GSLDataset(train_valid, dictionary, max_length)
    val_dataset   = GSLDataset(val_valid, dictionary, max_length)
    test_dataset  = GSLDataset(test_valid, dictionary, max_length)

    # DataLoader gia to TRAINING:
    # shuffle=True: anakateuei ti seira ton deigmaton se kathe epoch
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    # return the 3 dataLoaders 
    return train_loader, val_loader, test_loader