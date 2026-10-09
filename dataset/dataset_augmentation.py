#https://docs.pytorch.org/tutorials/beginner/basics/data_tutorial.html
# changing to match my project anti gia images emeis exoume keypoints idi apo to normalization twvn frames

import os
import random
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np


#augmentation functions ONLY FOR TRAINING based on this paper https://ieeexplore.ieee.org/document/9707552
#change them a bit because i use a different normalization method

def rotate_landmarks(landmarks, max_angle_degree=13):
    
    # rotate each landmark by [-13,+13] moires girw apo to center oloklirou  tou gloss ((x,y) apo frames * landmarks per frame)
    # i do not rotate based on the center of the frame as the paper as i used a different normalization method and the center 
    # of the frame is not necessarily at [0.5,0.5] afou kano vasi tous omous kai etsi to frame den menei [0,1] × [0,1] frame
    angle = np.radians(random.uniform(-max_angle_degree, max_angle_degree)) 
    
    cos = np.cos(angle) 
    sin = np.sin(angle)

    # apo [frames, landmarks, 2] se [all frames landmarks,2] kai pernoume ton meso oro tous gia to center tou gloss
    center = landmarks.reshape(-1, 2).mean(axis=0) # axis 0 mean / number of frames
    cx, cy = center

    x = landmarks[..., 0] #all x in (30, 146)
    y = landmarks[..., 1] #all y in (30, 146)

    x_rot = (x - cx)*cos-(y - cy)*sin+cx #new rotated x
    y_rot = (y - cy)*cos+(x - cx)*sin+cy #new rotated y

    
    rotated = np.stack([x_rot, y_rot], axis=-1) # back to (frames, points, 2)

    return rotated


def scale_landmarks(landmarks, scale_range=(0.92, 1.08)):
# randomly changes the width and height (scales) of the gloss me vasi to kentro tou (menei stathero) gia mi metakinithei olokliro to sign 

    scale_x = random.uniform(*scale_range) 
    scale_y = random.uniform(*scale_range)

    # apo [frames, landmarks, 2] se [all frames landmarks,2] kai pernoume ton meso oro tous gia to center tou gloss
    center = landmarks.reshape(-1, 2).mean(axis=0) # axis 0 mean / number of frames to calculate center

    centered = landmarks - center #prosorina metaferoume ta landmarks me vasi to center p ginete (0,0) gia efkolia 
    centered[..., 0] *= scale_x
    centered[..., 1] *= scale_y

    return centered + center # epanaferoume ta landmarks stin arxiki tous thesi vasi to center


def add_noise(landmarks, sigma=0.005):

# Mikres tyxaies allages sta landmark coordinates gia na prosomoiwsoume to MediaPipe detector jitter giati kathe idia mporei na min exei ta idia landmarks according to mediapipe detection
# Etsi to model ginetai pio anthektiko se mikres variations ton landmarks kai mporei na meiwsei to overfitting.

    noise = np.random.normal(loc=0.0, scale=sigma, size=landmarks.shape)
    return landmarks + noise


def augmentation(landmarks):
    #applies each augmentation independently with 50% probability in a fixed order

    landmarks = landmarks.astype(np.float32).copy() #metatrepoume ta landmarks se float32 and copy them so we do not change the original npy array
    if random.random() < 0.5:
        landmarks = rotate_landmarks(landmarks)
    if random.random() < 0.5:
        landmarks = scale_landmarks(landmarks)
    if random.random() < 0.5:
        landmarks = add_noise(landmarks)

    return landmarks


class GSLDataset(Dataset):

    def __init__(self, glosses, dictionary, max_length=30, augmentation=False):
        self.glosses = glosses # lista (path,label) poy ftiaksame apo to get_available_glosses
        self.dictionary = dictionary  #lista label-id
        self.max_length = max_length      # megisto length gia to stathero input
        self.augmentation = augmentation            # True mono gia to training set

    def __len__(self):
        return len(self.glosses) # returns the number of glosses we have


    def __getitem__(self, idx):
        npy_path, label = self.glosses[idx]  #idx gloss path

    
        landmarks = np.load(npy_path)  # [frames, landmarks, 2]

        if self.augmentation:
            landmarks = augmentation(landmarks)

        landmarks, valid_length = self.change_length(landmarks)   #change its number of frames to be at max_length

        label_id = self.dictionary[label] #get the id from its label in the dictionary

        return torch.tensor(landmarks, dtype=torch.float32), label_id, valid_length
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

    def change_length(self, landmarks):
        max_length = self.max_length
        length = landmarks.shape[0]
        # apo to (number of frames,dots/landmarks/point in each frame,how many dimensions is each landmark)
        # we get the first argument

        valid_length = min(length, max_length)
        
        if length > max_length:  # concatenate keep only the first max_length frames
            return landmarks[:max_length], valid_length

        elif length < max_length:  # padding
            padding_length = max_length - length  # see how much padding we need
            padding = np.zeros((padding_length, landmarks.shape[1], landmarks.shape[2]),dtype=landmarks.dtype)            # create npy table with zeros with the size of padding_length and with the same landmark size and dimensions
            padded = np.concatenate([landmarks, padding], axis=0)  # enose tous dio "pinakes", axis=0
                # to add them according to the frames length (proto argument)
            return padded, valid_length

        else:  # 30 length keep it as it is
            return landmarks, valid_length


def create_dataloaders(train_valid, val_valid, test_valid, dictionary, max_length=30,
                        batch_size=32):
# batch_size = posa dedomena tha vlepei/epeksergazete to montelo mazi/taftoxrona

    # ftiaxnoume 3 ksexorista Dataset objects. ena gia to kathe train/val/test
    # to kathena "kseri" mono ta dika toy dedomena alla ola tous xrisimopoioun to idio dictionary
    #augmentation=True only for training set
    train_dataset = GSLDataset(train_valid, dictionary, max_length, augmentation=True)
    val_dataset   = GSLDataset(val_valid, dictionary, max_length, augmentation=False)
    test_dataset  = GSLDataset(test_valid, dictionary, max_length, augmentation=False)

    # DataLoader gia to TRAINING:
    # shuffle=True: anakateuei ti seira ton deigmaton se kathe epoch
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # return the 3 dataLoaders
    return train_loader, val_loader, test_loader

