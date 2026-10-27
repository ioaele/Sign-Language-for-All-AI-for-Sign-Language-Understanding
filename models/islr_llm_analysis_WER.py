#python3 -m models.islr_llm
import json
import torch
import numpy as np
from pathlib import Path
import csv
from jiwer import wer

from preprocessing.get_glosses import get_glosses
from models.transformer import ISLR_Transformer
from torch.utils.data import DataLoader
from dataset.dataset import GSLDataset
#AUGMENTATION ONLY DURING TRAINING

import os
from dotenv import load_dotenv
from google import genai



#necessary paths 
PROJECT_ROOT = Path(__file__).resolve().parent.parent #code folder
MODEL_PATH = (PROJECT_ROOT/ "model_files"/ "islr_transformer_aug_best_config33_seed2026_40_BESTANDFINAL.pt")
DICTIONARY_PATH = PROJECT_ROOT / "dataset" / "dictionary.json"
DATA_ROOT = PROJECT_ROOT / "processed_landmarks"
VIDEO_FOLDERS = [
    DATA_ROOT / "health1_signer3_rep3_glosses",
    DATA_ROOT / "health1_signer3_rep4_glosses",
    DATA_ROOT / "health1_signer3_rep5_glosses",
    DATA_ROOT / "health2_signer3_rep3_glosses",
    DATA_ROOT / "health2_signer3_rep4_glosses",
    DATA_ROOT / "health2_signer3_rep5_glosses",
    DATA_ROOT / "health3_signer3_rep3_glosses",
    DATA_ROOT / "health3_signer3_rep4_glosses",
    DATA_ROOT / "health3_signer3_rep5_glosses",
    DATA_ROOT / "health4_signer3_rep3_glosses",
    DATA_ROOT / "health4_signer3_rep4_glosses",
    DATA_ROOT / "health4_signer3_rep5_glosses",
    DATA_ROOT / "health5_signer3_rep3_glosses",
    DATA_ROOT / "health5_signer3_rep4_glosses",
    DATA_ROOT / "health5_signer3_rep5_glosses",

    DATA_ROOT / "kep1_signer3_rep3_glosses",
    DATA_ROOT / "kep1_signer3_rep4_glosses",
    DATA_ROOT / "kep1_signer3_rep5_glosses",
    DATA_ROOT / "kep2_signer3_rep3_glosses",
    DATA_ROOT / "kep2_signer3_rep4_glosses",
    DATA_ROOT / "kep2_signer3_rep5_glosses",
    DATA_ROOT / "kep3_signer3_rep3_glosses",
    DATA_ROOT / "kep3_signer3_rep4_glosses",
    DATA_ROOT / "kep3_signer3_rep5_glosses",
    DATA_ROOT / "kep4_signer3_rep3_glosses",
    DATA_ROOT / "kep4_signer3_rep4_glosses",
    DATA_ROOT / "kep4_signer3_rep5_glosses",
    DATA_ROOT / "kep5_signer3_rep3_glosses",
    DATA_ROOT / "kep5_signer3_rep4_glosses",
    DATA_ROOT / "kep5_signer3_rep5_glosses",

    DATA_ROOT / "police1_signer3_rep3_glosses",
    DATA_ROOT / "police1_signer3_rep4_glosses",
    DATA_ROOT / "police1_signer3_rep5_glosses",
    DATA_ROOT / "police2_signer3_rep3_glosses",
    DATA_ROOT / "police2_signer3_rep4_glosses",
    DATA_ROOT / "police2_signer3_rep5_glosses",
    DATA_ROOT / "police3_signer3_rep3_glosses",
    DATA_ROOT / "police3_signer3_rep4_glosses",
    DATA_ROOT / "police3_signer3_rep5_glosses",
    DATA_ROOT / "police4_signer3_rep3_glosses",
    DATA_ROOT / "police4_signer3_rep4_glosses",
    DATA_ROOT / "police4_signer3_rep5_glosses",
    DATA_ROOT / "police5_signer3_rep3_glosses",
    DATA_ROOT / "police5_signer3_rep4_glosses",
    DATA_ROOT / "police5_signer3_rep5_glosses",
]

sum=0

for VIDEO_FOLDER in VIDEO_FOLDERS:
    #sorting the npy files CHAT COMMAND
    npy_files = sorted(VIDEO_FOLDER.glob("*.npy"),key=lambda path: int(path.stem.replace("glosses", "")))

    MAX_LENGTH = 40

    DEVICE = torch.device("cpu")

    with open(DICTIONARY_PATH,"r",encoding="utf-8") as f: #loading the dictionary
        dictionary = json.load(f)
        
    #dimiourgo ton transformer me ta configurations tou trained model
    model = ISLR_Transformer(num_landmarks=292,d_model=128,num_heads=4,encoder_layers=2,max_length=MAX_LENGTH,num_glosses=len(dictionary),dim_feedforwardnetwork=512,
    percentage_zero_dropout=0.2,pooling_method="attention").to(DEVICE)


    #loading ta trained vari: einai oi arithmitikes times pou emathe o transformer kata to train
    trained_weights = torch.load( MODEL_PATH,map_location=DEVICE,weights_only=True)

    #input the trained weights into the model 
    model.load_state_dict(trained_weights)

    model.eval() #mode evaluation

    #PROSTOPARON. META THA TO KANO NA PERNEI OPOIODIPOTE VIDEO ME PREPROCESSING ETC
    dataset = GSLDataset(glosses=npy_files,dictionary=None,max_length=MAX_LENGTH)
    loader = DataLoader(dataset,batch_size=1,shuffle=False)
    #BATCH SIZE=1 AFOU MONO 1 VIDEO METAFRAZOUME THN FORA

    #id -> gloss
    id_to_gloss = {int(value): key for key, value in dictionary.items()}
    predicted_glosses = []

    #no gradients
    with torch.no_grad():
        
        for landmarks, not_padded_frames in loader:
            
            #input:40 frames x 292 landmarks output: 306 classes me kathe ena ena skor
            logits = model(landmarks.to(DEVICE), not_padded_frames.to(DEVICE))

            #getting the max score (the predicted gloss)
            predicted_id = logits.argmax(dim=1).item()
            predicted_gloss = id_to_gloss[predicted_id]

            predicted_glosses.append(predicted_gloss)

    predicted_gloss_sequence = " ".join(predicted_glosses)

    # print("Predicted gloss sequence:\n", predicted_gloss_sequence)

    #ADDED PART!!!! 
    #TRANSFORMER EVALUATION
    #WORD ERROR RATE 
    #ARITHMOS LATHON/ARITHMOS GLOSSES 
    #ARITHMOS LATHON= ARITHMOS LATHOS PREDICTION, ARITHMOS POU TO MODELO PROVLEPEI EPIPLEON LEKSIS POU DEN IPARXOYN STO GLOSS SEQUENCE, ARITHMOS POU TO MONTELO DEN PROVLEPSE TIPOTE GIA TO SIGKEKRIMENO GLOSS
    #reference:https://www.llamaindex.ai/glossary/what-is-word-error-rate

    TEST_CSV = PROJECT_ROOT / "data" / "test_greek_iso.csv"

    test_glosses = get_glosses(TEST_CSV, DATA_ROOT)

    #dictionary of file name and gloss
    #px "glosses0001":"ΓΕΙΑ"
    file_gloss_dictionary= {Path(full_path).stem: label
        for full_path, label in test_glosses
        if Path(full_path).parent == VIDEO_FOLDER
    }

    #.stem=παιρνει το ονομα του αρχειου χωρις την καταληξη του (.npy)
    #getting the actual glosses of the video to compare
    #px ["ΓΕΙΑ","ΕΡΦΕΡΦ",...]
    actual_glosses = [file_gloss_dictionary[path.stem] for path in npy_files]

    actual_glosses_sequence = " ".join(actual_glosses) #ta enonoume se mia protasi 

    # print("Actual gloss sequence:\n", actual_glosses_sequence)

    transformer_wer = wer(actual_glosses_sequence,predicted_gloss_sequence) #ipologizete to wer

    print(f"Transformer WER:{transformer_wer * 100:.2f}%")
    
    sum+=transformer_wer*100
    
    
print(f"MEAN WER:{sum/len(VIDEO_FOLDERS):.2f}%")
