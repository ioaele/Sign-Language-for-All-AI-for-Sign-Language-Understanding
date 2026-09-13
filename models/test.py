"""
inference.py
=============

Pairnei to IDI training tou montelou (to .pt arxeio pou apothikeftike
apo to train_v2.py) kai to xrisimopoiei gia na kanei PROVLEPSI (inference)
se ENA OLOKLIRO FAKELO me glosses (glosses0000.npy, glosses0001.npy, ...)
-- ola ta arxeia enos video, ME TI SEIRA.

To apotelesma einai mia LISTA apo glosses (p.x. "ΑΣΤΥΝΟΜΙΑ ΕΓΩ ΨΑΧΝΩ"),
etoimi na tin stileis meta se ena LLM (GPT/Claude) gia na ftiaxei
kanoniki, gramatika sosti protasi.

Xrisi:
    python inference.py
"""

import json
import numpy as np
import torch

from models.transformer import ISLRTransformer


# ============================================================
# 0. RYTHMISEIS -- idia me auta pou xrisimopoiises sto training
# ============================================================

MODEL_PATH = "islr_v2_best.pt"      # to .pt pou apothikeftike apo to train_v2.py
DICTIONARY_PATH = "dataset/dictionary.json"

HIDDEN_DIM = 256
NUM_HEADS = 8
NUM_LAYERS = 4
DROPOUT = 0.1                        # den peirazei sto inference (to eval() to kleinei etsi ki allios)
MAX_LENGTH = 30
POOLING = "class_query"


# ============================================================
# 1. FORTONOUME TO DICTIONARY (id -> leksi/gloss)
# ============================================================

with open(DICTIONARY_PATH, "r", encoding="utf-8") as f:
    dictionary = json.load(f)          # p.x. {"ΤΡΩΩ": 42, "ΨΩΜΙ": 7, ...}

id_to_label = {v: k for k, v in dictionary.items()}   # antistrofo: {42: "ΤΡΩΩ", ...}
num_classes = len(dictionary)


# ============================================================
# 2. FORTONOUME TO MONTELO (IDIA arxitektoniki, apla xoris training)
# ============================================================

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

model = ISLRTransformer(
    input_dim=292,
    hidden_dim=HIDDEN_DIM,
    num_heads=NUM_HEADS,
    num_layers=NUM_LAYERS,
    num_classes=num_classes,
    max_length=MAX_LENGTH,
    dropout=DROPOUT,
    pooling=POOLING
).to(device)

# fortonoume ta MATHIMENA varh (oxi tyxaia pia -- auta pou ekpaideftikan)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))

# .eval() = "modalitita provlepsis" -- kleinei to dropout, den
# allazei tipota, apla xrisimopoiei to montelo opos einai
model.eval()


# ============================================================
# 3. SYNARTISI: ENA video landmarks -> mia leksi/gloss
# ============================================================

def predict_gloss(landmarks_tensor, num_real_frames):
    """
    landmarks_tensor: tensor sximatos [frames, landmarks, coords]
                       (ENA MONO video, oxi batch)
    num_real_frames:  posa apo auta ta frames einai PRAGMATIKA
                       (ta ypoloipa mexri MAX_LENGTH einai padding)

    epistrefei: (gloss_string, confidence_0_to_1)
    """

    # to montelo perimenei BATCH (protes diastasi = posa videos mazi).
    # emeis exoume MONO 1 video, ara prosthetoume mia "fake" diastasi
    # batch=1 sto jekinima (unsqueeze(0))
    x = landmarks_tensor.unsqueeze(0).to(device)          # [1, frames, landmarks, coords]
    lengths = torch.tensor([num_real_frames]).to(device)   # [1]

    with torch.no_grad():   # den kanoume training, den xreiazomaste gradients
        logits = model(x, lengths, use_mask=True)           # [1, num_classes]

        # metatrepoume ta 310 "skor" se PITHANOTITES pou athroizoun sto 1
        probabilities = torch.softmax(logits, dim=1)

        # to pio psilo skor = i provlepsi tou montelou
        confidence, predicted_id = probabilities.max(dim=1)

    gloss = id_to_label[predicted_id.item()]
    return gloss, confidence.item()


# ============================================================
# 4. FORTOSI ENOS PRAGMATIKOU .npy ARXEIOU
# ============================================================
# AKRIVOS I IDIA logiki me to change_length() tou GSLDataset sou
# (dataset/GSLDataset_augmentation.py) -- to shape einai
# [frames, landmarks, 2] (MONO x,y -- OXI 4 coords, auto to eixa
# lathos prin). Den ksanagrafoume kainouria logiki, apla tin
# xrisimopoioume edo gia ENA deigma anti gia olo to dataset.

def load_landmarks_from_npy(npy_path, max_length=MAX_LENGTH):
    """
    Fortonei ENA .npy arxeio landmarks (p.x. apo
    processed_landmarks/health1_signer1_rep2_glosses/kapoio_gloss.npy)
    kai to etoimazei gia to montelo, AKRIVOS opos to change_length()
    tou GSLDataset:

    1. To .npy exei to PRAGMATIko mikos tou video (p.x. 18 frames),
       sxima [frames, landmarks, 2]
    2. An exei PERISSOTERA frames apo max_length -> ta kovoume (truncate)
    3. An exei LIGOTERA -> gemizoume (pad) me midenika sto telos
    4. Krataei to PRAGMATIKO (valid) mikos xexoristo, gia na kserei
       to montelo poia frames einai padding
    """

    landmarks = np.load(npy_path)   # sxima: [frames, landmarks, 2]

    length = landmarks.shape[0]
    valid_length = min(length, max_length)

    if length > max_length:
        landmarks = landmarks[:max_length]
    elif length < max_length:
        padding_length = max_length - length
        padding = np.zeros(
            (padding_length, landmarks.shape[1], landmarks.shape[2]),
            dtype=landmarks.dtype
        )
        landmarks = np.concatenate([landmarks, padding], axis=0)
    # an length == max_length, den kanoume tipota, to kratame etsi

    landmarks_tensor = torch.tensor(landmarks, dtype=torch.float32)

    return landmarks_tensor, valid_length


# ============================================================
# 5. STELNOUME TA GLOSSES STO LLM (Gemini -- DOREAN) GIA PROTASI
# ============================================================
# Xreiazete: pip install google-genai
# Kai to API key sou san environment variable (dorean, xoris karta):
#
#   1. Pigaine sto https://aistudio.google.com/apikey
#   2. "Create API key"
#   3. export GEMINI_API_KEY="..."   (Mac/Linux)
#      setx GEMINI_API_KEY "..."     (Windows)
import os
from groq import Groq

llm_client = Groq(
    api_key=os.environ.get("GROQ_API_KEY")
)


def glosses_to_sentence(gloss_sentence, model="openai/gpt-oss-120b"):
    """
    Pairnei ta glosses (p.x. "ΑΣΤΥΝΟΜΙΑ ΕΓΩ ΨΑΧΝΩ") kai ta stelnei
    sto Groq, zitontas mia SOSTI, gramatika protasi sta Ellinika.

    gloss_sentence: string me ta glosses me ti sosti seira, xorismena
                    me kena (auto pou vgazei to proigoumeno vima)
    model:          poio Groq montelo na xrisimopoiisei

    epistrefei: string -- i teliki protasi
    """

    prompt = (
        "Αυτά είναι glosses (λέξεις-κλειδιά) από βίντεο Ελληνικής "
        "Νοηματικής Γλώσσας, με τη σειρά που εμφανίστηκαν:\n\n"
        f"{gloss_sentence}\n\n"
        "Γράψε ΜΟΝΟ μία σωστή, γραμματικά ολοκληρωμένη πρόταση στα "
        "Ελληνικά που αποδίδει το νόημά τους. Μην προσθέσεις τίποτα "
        "άλλο εκτός από την πρόταση."
    )

    response = llm_client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    sentence = response.choices[0].message.content.strip()

    return sentence

# ============================================================
# 6. PARADEIGMA XRISIS -- OLOKLIRO to pipeline, apo ta .npy mexri protasi
# ============================================================

if __name__ == "__main__":

    # ---- EDO: valto path enos FAKELOU me polla glosses ----
    # Kathe fakelos (p.x. "processed_landmarks/police4_signer2_rep2_glosses/")
    # periexei POLLA .npy arxeia -- ENA gia KATHE leksi/gloss tou video,
    # onomasmena me SEIRA: glosses0000.npy, glosses0001.npy, k.o.k.
    #
    # Trexoume to montelo se KATHE arxeio, ME TI SEIRA (0000, 0001, ...),
    # kai mazevoume ta apotelesmata se mia lista -- afti i lista einai
    # to "gloss sentence" pou tha steileis meta sto LLM.

    import glob

    FOLDER_PATH = "processed_landmarks/police1_signer1_rep1_glosses"

    # sorted() -- gia na eimaste SIGOUROI oti ta pairnoume me ti sosti
    # seira (0000, 0001, 0002, ...) kai oxi tixaia
    npy_files = sorted(glob.glob(f"{FOLDER_PATH}/glosses*.npy"))

    predicted_glosses = []

    for npy_path in npy_files:
        landmarks_tensor, num_real_frames = load_landmarks_from_npy(npy_path)
        gloss, confidence = predict_gloss(landmarks_tensor, num_real_frames)

        predicted_glosses.append(gloss)

        print(f"{npy_path}  ->  {gloss}  (σιγουριά: {confidence:.2%})")

    # to teliko "gloss sentence" -- afto tha steileis sto LLM
    gloss_sentence = " ".join(predicted_glosses)

    print(f"\nΣειρά glosses: {gloss_sentence}")

    # stelnoume ta glosses sto LLM gia na ftiaksei sosti protasi
    final_sentence = glosses_to_sentence(gloss_sentence)
    print(f"Τελική πρόταση: {final_sentence}")