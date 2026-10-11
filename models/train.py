#python3 -m models.train
import json
import torch
import torch.nn as nn
from pathlib import Path

from dataset.dataset import create_dataloaders
from dataset.dataset_augmentation import create_dataloaders_aug
from preprocessing.prepare_dataset import get_valid_glosses,relative_to_full_gloss_paths
from models.transformer import ISLR_Transformer


MAX_LENGTH = 40 #max frame length
BATCH_SIZE = 32 #how many samples the model processes in each iteration before updating its weights
EPOCHS = 50 #epanalipsis training dataset
LEARNING_RATE = 0.0001 # leei poso grigora the model updates its weights
#TO LOSS LEEI POSO LATHOS IMASTE APO TO SOSTO LABEL. OSO PIO MIKRO TOSO TO KALITERO (KALITERI PROVLEPSI)

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent #pernei to parent folder apo to path tou current file (t train.py)

DATA_ROOT = PROJECT_ROOT / "processed_landmarks"

TRAIN_CSV = PROJECT_ROOT / "data" / "train_greek_iso_merged.csv"
VAL_CSV = PROJECT_ROOT / "data" / "val_greek_iso_merged.csv"
TEST_CSV = PROJECT_ROOT / "data" / "test_greek_iso_merged.csv"

DICTIONARY_PATH = PROJECT_ROOT / "dataset" / "dictionary.json"

MIN_FRAMES = 3 #temporary untli solving the storage problem

# Select the device the model is going to run on
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps") #mac apple silicon
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

print("Device:", DEVICE)

    
    
#Load the gloss dictionary
with open(DICTIONARY_PATH, "r", encoding="utf-8") as f:
    dictionary = json.load(f)

#get the valid glosses for each dataset
train_set = get_valid_glosses(TRAIN_CSV, DATA_ROOT, MIN_FRAMES)
val_set = get_valid_glosses(VAL_CSV, DATA_ROOT, MIN_FRAMES)
test_set = get_valid_glosses(TEST_CSV, DATA_ROOT, MIN_FRAMES)

#relative to full paths
train_set = relative_to_full_gloss_paths(train_set, DATA_ROOT)
val_set= relative_to_full_gloss_paths(val_set, DATA_ROOT)
test_set= relative_to_full_gloss_paths(test_set, DATA_ROOT)


#dimiourgw ta dataloaders
train_loader, val_loader, test_loader = create_dataloaders(train_set,val_set,test_set,dictionary,
    max_length=MAX_LENGTH,
    batch_size=BATCH_SIZE,
)

#dimiourgw ta dataloaders
train_loader_aug, val_loader_aug, test_loader_aug = create_dataloaders_aug(train_set,val_set,test_set,dictionary,
    max_length=MAX_LENGTH,
    batch_size=BATCH_SIZE,
)

#initialing the transformer to the device we chose
model = ISLR_Transformer(num_landmarks=292,max_length=MAX_LENGTH,num_glosses=len(dictionary),).to(DEVICE)

#epilegoume to loss function:
# Loss Function: Measuring Error
# After making a prediction, the model needs to know how close it was to the correct answer.

# That’s where the loss function comes in.

# It computes a numerical score that reflects the difference between the model’s prediction and the true label. Common loss functions:

# Cross-Entropy Loss for classification EPILEGOUME AYTO
# Mean Squared Error (MSE) for regression
# Negative Log-Likelihood for language models
#https://medium.com/@yssudarsan/training-a-deep-learning-model-bcd29a74304f

loss_function = nn.CrossEntropyLoss() #to cross entropi loss metra poso apexei 
#h provlepsi tou montelou apo thn sosth kathgoria 

#epilegoume ton optimizer
#optimizer: different optimizers affect how weights are updated
# Common choices include Adam, RMSProp, AdaDelta, and SGD. 
# Experiment to see which optimizer works best for your task, 
# as each has unique advantages depending on the dataset and problem type.
#WE START WITH THE ADAM
optimizer = torch.optim.Adam(model.parameters(),lr=LEARNING_RATE) 
#model.parameters(): ta varoi kai oi parametroi p tha mathei to modelo


#ARA SO FAR WE HAVE: (1)Transformer model: Dexetai landmarks kai paragei ta logits gia ta glosses
#(2) CrossEntropyLoss: Sigkrinei ta logits me tis sostes etiketes kai ipologizei to loss
#(3) Adam optimizer: Xrisimopoiei tis kliseis tou loss gia na enimerosei ta varoi



def train(model, loader, loss_function, optimizer, device):
    model.train() #train mode

    #arxikopoihsh metavlitwn
    total_loss = 0.0
    correct = 0
    total = 0

    for landmarks, labels, not_padded_frames in loader:
        #metaferoume ta sto device p ipologizontai ola
        landmarks = landmarks.to(device)
        labels = labels.to(device)
        not_padded_frames = not_padded_frames.to(device)

        
        optimizer.zero_grad() #midenozoume ola ta gradients p exoume ipologisei prin
        #gradients(kliseis): einai enas arithmos gia kathe weight poy lalei 
        #"an afksiso ligo afto to weight, to loss tha anevei h tha pesei kai poso grigora"

        #calling the transformer kai mas epistrefei thn vathmologia(logits) gia kathe gloss sto dictionary
        logits = model(landmarks, not_padded_frames)
        
        #ipologizei to loss: poso apexoun oi provlepsis apo ta sosta labels tou batch 
        loss = loss_function(logits, labels)

        #ipologizei ta gradients tou loss os gia kathe weight tou modelou (ipologizei poso epireazei kathe parametros to sfalma)
        #CLAUDE EXPLANATION CAUSE I DO NOT UNDERSTAND 100% WHAT GRADIENTS DO 
        # Η αναλογία με τη συνταγή
        # Φτιάχνεις μια σούπα και βγαίνει πολύ αλμυρή. Το λάθος (το loss) είναι ότι η γεύση απέχει από αυτή που ήθελες. Η σούπα έχει πολλά υλικά: αλάτι, νερό, κρεμμύδι, μπαχαρικά.
        # Το backward() είναι σαν να ρωτάς για κάθε υλικό ξεχωριστά: «αν άλλαζα λίγο την ποσότητά σου, πόσο θα άλλαζε η γεύση;»
        # Το αλάτι έχει τεράστια επίδραση: λίγο λιγότερο αλάτι θα διόρθωνε πολύ το λάθος.
        # Το κρεμμύδι έχει μικρή επίδραση.
        # Το νερό σχεδόν καθόλου.

        # Αυτή η επίδραση, ως αριθμός για κάθε υλικό, είναι η κλίση. Μεγάλη κλίση σημαίνει ότι το υλικό φταίει πολύ. Μικρή κλίση σημαίνει ότι φταίει λίγο.

        # Στο μοντέλο
        # Τα «υλικά» είναι τα weights. Μετά το forward pass και τον 
        # υπολογισμό του loss, το loss.backward():
        # Ξεκινά από το loss και πηγαίνει προς τα πίσω μέσα από τα layers,
        # από την έξοδο προς την είσοδο (γι' αυτό λέγεται backward).
        # Σε κάθε weight που συναντά, υπολογίζει πόσο επηρέασε το τελικό λάθος.
        # Αποθηκεύει αυτόν τον αριθμό δίπλα στο weight.

        # Το σημαντικό: δεν αλλάζει κανένα weight. Απλώς γράφει ένα 
        # «δελτίο ευθύνης» για το καθένα. Η αλλαγή γίνεται μετά, από το 
        # optimizer.step(), που διαβάζει αυτά τα δελτία.

        # Γιατί «προς τα πίσω»
        # Το λάθος γεννιέται στο τέλος, στην πρόβλεψη. 
        # Η πρόβλεψη όμως εξαρτάται από ό,τι έγινε στο προηγούμενο layer, 
        # που εξαρτάται από το προηγούμενο, και ούτω καθεξής. 
        # Για να βρεις ποιο weight στην αρχή φταίει, πρέπει να ακολουθήσεις 
        # την αλυσίδα αιτίου και αποτελέσματος ανάποδα. Η μέθοδος αυτή λέγεται 
        # backpropagation και χρησιμοποιεί τον κανόνα της αλυσίδας από τα 
        # μαθηματικά, ώστε να μην χρειάζεται να δοκιμάσει κάθε weight χωριστά.
        loss.backward()
        
        #efarmmozei ton tipo  w = w − learning rate × gradients se ola ta weights
        optimizer.step()

        batch_size = labels.size(0)

        total_loss += loss.item() * batch_size #sum of loss (to loss einai sinithos o mo tou batch
        #ara poll/zw me batch_size gia na parw to actual loss tou kathe batch)
        #TO .ITEM METATREPEI ENA ANTIKEIMENO PYTORCH SE APLO ARITHMO PY.
        
        #max_logits: h provlepsi (afti me thn megaliteri vathmologia)
        predictions = logits.argmax(dim=1)
        
        correct += (predictions == labels).sum().item() #if correct prediction correct++
        
        total += batch_size #sinolika digmata

    average_loss = total_loss / total #mo loss olwn twn digmaton enos epoch
    accuracy = correct / total #% ton digmaton pou htan predicted sosta 

    return average_loss, accuracy



def evaluation(model, loader, loss_function, device):
    model.eval() #validation mode

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad(): #einai aparatito (apparently) giati sto evaluation den xriazomaste na ipologisoume ta gradients
        #dioti den enimeronoume ta varoi
        for landmarks, labels, not_padded_frames in loader:
            landmarks = landmarks.to(device)
            labels = labels.to(device)
            not_padded_frames = not_padded_frames.to(device)

            logits = model(landmarks, not_padded_frames)
            loss = loss_function(logits, labels)
            #den enimeromou ta varoi giati theloume na metrisome thn apodosi tou idi ekpaidevmenou modelou
            #oxi na to ekpaidefsoume ksana
            
            #CLAUDE EXPLANATION
            #Η αναλογία με το διαγώνισμα
            # Φαντάσου έναν μαθητή που διαβάζει για εξετάσεις.
            # Training: λύνει ασκήσεις, κοιτάζει τις λύσεις, καταλαβαίνει 
            # τα λάθη του και διορθώνει τον τρόπο που σκέφτεται. Εδώ μαθαίνει.
            # Evaluation: δίνει το διαγώνισμα. Απαντά και παίρνει βαθμό, αλλά δεν 
            # επιτρέπεται να αλλάξει τον τρόπο σκέψης του μέσα στο διαγώνισμα.
            # Αν τον άφηνες να διορθώνει τις απαντήσεις του βλέποντας τη λύση την 
            # ώρα του διαγωνίσματος, ο βαθμός δεν θα έδειχνε τι ξέρει πραγματικά. 
            # Θα έδειχνε πόσο καλά αντέγραψε.

            # Γιατί όχι ενημέρωση των weights
            # Αν αλλάζαμε τα weights στα δεδομένα του evaluation, 
            # το μοντέλο θα «έβλεπε» τις απαντήσεις και θα προσαρμοζόταν σε αυτές. 
            # Τότε το evaluation δεν θα μας έλεγε πώς θα τα πάει σε καινούργια δεδομένα 
            # που δεν έχει ξαναδεί, και αυτό είναι ακριβώς που θέλουμε να μάθουμε.
            
            batch_size = labels.size(0)

            total_loss += loss.item() * batch_size

            predictions = logits.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += batch_size

    average_loss = total_loss / total
    accuracy = correct / total

    return average_loss, accuracy




MODEL_PATH = PROJECT_ROOT / "islr_transformer_best.pt"
MODEL_PATH_AUG= PROJECT_ROOT / "islr_transformer_aug_best.pt"

#TRAIN AND TEST
best_validation_accuracy = 0.0
best_validation_aug_accuracy=0.0
early_stop=5
i=0
print("TRAIN/VAL")
for epoch in range(1, EPOCHS + 1): #epanalamvanoume tin diadikasia epoch fores
  
    train_loss, train_accuracy = train(model, train_loader, loss_function, optimizer, DEVICE)
    train_loss_aug,train_accuracy_aug=train(model,train_loader_aug,loss_function,optimizer,DEVICE)
    
    val_loss, val_accuracy = evaluation(model, val_loader, loss_function, DEVICE)
    val_loss_aug, val_accuracy_aug = evaluation(model, val_loader_aug, loss_function, DEVICE)

    print(f"Epoch [{epoch}/{EPOCHS}]")
        
    print(
        f"Train Loss: {train_loss:.4f} | "
        f"Train Accuracy: {train_accuracy:.2%} | "
        f"Val Loss: {val_loss:.4f} | "
        f"Val Accuracy: {val_accuracy:.2%}"
    )
    
    
    print(
        f"Train Loss Aug: {train_loss_aug:.4f} | "
        f"Train Accuracy Aug: {train_accuracy_aug:.2%} | "
        f"Val Loss Aug: {val_loss_aug:.4f} | "
        f"Val Accuracy Aug: {val_accuracy_aug:.2%}"
    )
    

    if val_accuracy > best_validation_accuracy: 
        best_validation_accuracy = val_accuracy #max accuracy
        i=0
        torch.save(model.state_dict(), MODEL_PATH)

        print(f"Best model saved")
        print(f"Validation accuracy: {best_validation_accuracy:.2%}")

    else:
        i = i + 1
        
            
    if val_accuracy_aug> best_validation_aug_accuracy: 
        best_validation_aug_accuracy = val_accuracy_aug #max accuracy
        torch.save(model.state_dict(), MODEL_PATH_AUG)

        print(f"Best Augmentation model saved")
        print(f"Validation aug accuracy: {best_validation_aug_accuracy:.2%}\n")
    

    if i >= early_stop:
        print(f"No improvement for {early_stop} epochs")
        print(f"Stopping {epoch}/{EPOCHS} epoch")
        break
        
       

#TEST
# print("TEST")
# #loading the best model
# model.load_state_dict(
#     torch.load(MODEL_PATH, map_location=DEVICE)
# )

# #evaluating on test set
# test_loss, test_accuracy = evaluation(model, test_loader, loss_function, DEVICE)

# print(f"Best Validation Accuracy: {best_val_accuracy:.2%}")
# print(f"Test Loss: {test_loss:.4f}")
# print(f"Test Accuracy: {test_accuracy:.2%}")
