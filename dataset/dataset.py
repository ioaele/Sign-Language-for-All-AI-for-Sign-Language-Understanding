#https://docs.pytorch.org/tutorials/beginner/basics/data_tutorial.html
# changing to match my project anti gia images emeis exoume keypoints idi apo to normalization twvn frames

import os
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class GSLDataset(Dataset):
    
    def __init__(self, glosses, dictionary, max_length=40):
        self.glosses = glosses # lista (path,label) poy ftiaksame apo to get_available_glosses
        self.dictionary = dictionary  #lista label-id
        self.max_length = max_length      # megisto length gia to stathero input
        
    def __len__(self):
        return len(self.glosses) # returns the number of glosses we have

   
    def __getitem__(self, index):
        npy_path, label = self.glosses[index]  #index gloss path

        landmarks = np.load(npy_path)  # [frames, landmarks, 2]

        landmarks, not_padding_frames = self.change_length(landmarks)   #change its number of frames to be at max_length

        #BHMA 1
        landmarks = landmarks.reshape(self.max_length, -1) #(32,40,146,2) -> (32,40,292)

        
        label_id = self.dictionary[label] #get the id from its label in the dictionary

        return torch.tensor(landmarks, dtype=torch.float32), label_id, not_padding_frames
     #for a specific index return the landmarks of that gloss, its label id and its real number of frames
     
     #Ένα tensor είναι ένας πολυδιάστατος πίνακας αριθμών. 
     # Στο PyTorch, τα tensors είναι η βασική δομή δεδομένων που χρησιμοποιούμε για 
     # να αποθηκεύουμε inputs, labels, παραμέτρους του μοντέλου και αποτελέσματα υπολογισμών.
     
     
# afou to deep model prepei na exei to idio input type each time prepei na kanoume ta glosses mas idiou mikous
# eite padding to make them bigger eite truncate to make the shorter (in length)
# giafto kai eprepe na doume to meso length ton glosses (posa frames exoun) me thn klassi
# calculate_staticts k apo oti idame:
#AFTER EDA I HAVE TO THE CONCLUSION THAT MAYBE 40 MAX_LENGTH IS MORE OPTIMAL THAN 30.
    def change_length(self, landmarks):
        max_length = self.max_length
        length = landmarks.shape[0]
        # apo to (number of frames,dots/landmarks/point in each frame,how many dimensions is each landmark)
        # we get the first argument

        not_padding_frames = min(length, max_length) #posa frames einai pragmatika frames kai oxi padding
        
        if length > max_length:  # concatenate keep only the first max_length frames
            return landmarks[:max_length], not_padding_frames

        elif length < max_length:  # padding
            padding_length = max_length - length  # see how much padding we need
            padding = np.zeros((padding_length, landmarks.shape[1], landmarks.shape[2]),dtype=landmarks.dtype)            # create npy table with zeros with the size of padding_length and with the same landmark size and dimensions
            padded = np.concatenate([landmarks, padding], axis=0)  # enose tous dio "pinakes", axis=0
                # to add them according to the frames length (proto argument)
            return padded, not_padding_frames

        else:  # max_length keep it as it is
            return landmarks, not_padding_frames
            
      
def create_dataloaders(train_valid, val_valid, test_valid, dictionary, max_length=40, batch_size=32):
# batch_size = posa dedomena tha vlepei/epeksergazete to montelo mazi/taftoxrona
#APO (40,146,2) -> (32,40,146,2) !!!
    # ftiaxnoume 3 ksexorista Dataset objects. ena gia to kathe train/val/test
    # to kathena "kseri" mono ta dika toy dedomena alla ola tous xrisimopoioun to idio dictionary 
    train_dataset = GSLDataset(train_valid, dictionary, max_length)
    val_dataset   = GSLDataset(val_valid, dictionary, max_length)
    test_dataset  = GSLDataset(test_valid, dictionary, max_length)

    # DataLoader gia to TRAINING:
    # shuffle=True: anakateuei ti seira ton deigmaton se kathe epoch APARAITHTO GIA MONO TO TRAINING SET
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # return the 3 dataLoaders 
    return train_loader, val_loader, test_loader