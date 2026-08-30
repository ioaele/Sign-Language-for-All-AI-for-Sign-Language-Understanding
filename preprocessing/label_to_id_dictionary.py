# o ipologistis den katalamvainei apo leksis gia na mpori px na antistoixisi to kathe gloss sto antistoixo label toy
# ara prepei oles aftes oi unique leksis pou exoume na metatrapoun se arithmous kai etsi tha exoume ena leksiko label -> id
# apo to train dataset mas!!
from prepare_dataset import get_valid_glosses
import json

def label_to_id():
    root = "processed_landmarks"  
    label_to_id={}
    # Χρησιμοποιείς λίστα από tuples (dictionary = [("ΓΕΙΑ", 0), ("ΕΓΩ", 1), ...]) αντί για πραγματικό Python dictionary ({"ΓΕΙΑ": 0, "ΕΓΩ": 1, ...}). Λειτουργικά και τα δύο "λεξικά" έχουν την ίδια πληροφορία, αλλά:
# Με λίστα, για να βρεις το id μιας συγκεκριμένης λέξης (π.χ. "ΓΕΙΑ") θα πρέπει να ψάξεις μέσα στη λίστα ένα-ένα μέχρι να τη βρεις (αργό αν έχεις 310 στοιχεία, ασήμαντο εδώ, αλλά κακή συνήθεια)
# Με πραγματικό dictionary, η αναζήτηση είναι άμεση: label_to_id["ΓΕΙΑ"] σου δίνει κατευθείαν το id, χωρίς αναζήτηση
    MIN_FRAMES = 3   

    train_glosses = get_valid_glosses("data/train_greek_iso.csv",root,MIN_FRAMES)

    i=0

    unique_labels = sorted(set(label for npy_path, label in train_glosses))

    for label in unique_labels:
        label_to_id[label] = i
        i=i+1
            
            
    # print (label_to_id)
    return label_to_id


if __name__ == "__main__":
    dictionary = label_to_id()

#htan aparaithto na to apothikevso kapou gia na min xriazete kathe fora na to ksana ftiaxnw gia na to prospelasw
    with open("dictionary.json", "w", encoding="utf-8") as f:
        json.dump(dictionary, f, ensure_ascii=False, indent=2)

    print(f"Label-to-id dictionary saved in label_to_id.json")

