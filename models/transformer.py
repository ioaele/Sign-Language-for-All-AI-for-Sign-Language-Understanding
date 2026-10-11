import torch
import torch.nn as nn 
import math

#3 Positional encoding
#xroniki seira (thesi) tou kathe frame
#apo to arthro https://medium.com/@xiaxiami/understanding-positional-encoding-in-transformer-and-large-language-models-58f1d9a713ed
#xrisimopoio to Sinusoidal Positional Encoding p einai pio konta sta dedomena pou eisixthike apo to paper attention is all you need
#kalo episis ine kai to learned positional encoding.(comparison later)
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_length): #constructor
        super().__init__()

        #to dianisma tis thesis 
        pe = torch.zeros(max_length, d_model)

       #gia kathe frame kai gia kathe zevgari stilwn (i,i+1)
        for pos in range(max_length):
            for i in range(0, d_model, 2):
                angle = pos / (10000 ** (i / d_model)) #tipos
                pe[pos, i] = math.sin(angle)       #artia stili:sin
                pe[pos, i + 1] = math.cos(angle)   #peritti stili:cos


        pe=pe.unsqueeze(0) #prostheto mia nea diastasi sthna arxi gia to batch size gia na mporei na gini h prosthesi tou dianismatos thesis
        #me to arxiko frame_embeddings pinaka-(32,40,128) (40,128)->(1,40,128)
        
        #apothikevoume ton pinaka ws stathero (oxi parametro, den mathenete klp)
        #mesa sto buffer tou montelou wste na metakinite mazi toy sti gpu (an exoume)
        self.register_buffer("pe", pe) #ensoma

    def forward(self, frame_embeddings):
        return frame_embeddings + self.pe[:, :frame_embeddings.size(1), :] #prostitei ton pinaka thesewn sto x (landmarks 128 after embedding) san extra pliroforia pou leei ti thesi einai to kathe frame
    #still (32,40,128) mexri thn grammi tou max_length 





class ISLR_Transformer(nn.Module): #nn.Module einai h vasiki klasi ths pytorch gia nevronika diktia
    
    def __init__(
        self,
        num_landmarks=292,
        d_model=128,
        num_heads=4,
        encoder_layers=2,
        max_length=40,
        num_glosses=305,
        dim_feedforwardnetwork=512,
        percentage_zero_dropout=0.1
    ): #constructor: an den doso parametrous arxikopoia aftomata
        
        super().__init__() #arxikopoiei to nn.Module
        
        self.num_landmarks = num_landmarks
        self.d_model = d_model
        self.max_length = max_length
        self.num_glosses = num_glosses
        
        
        #2
        #input embedding (linear)
        self.input_projection = nn.Linear(num_landmarks, d_model) #(32,40,292)->(32,40,128)
        #chat explaination: Για κάθε frame, το μοντέλο παίρνει τους 292 αριθμούς και υπολογίζει 128 νέους αριθμούς, 
        #χρησιμοποιώντας παραμέτρους που μαθαίνει κατά την εκπαίδευση.

        #3 Positional encoding
        #xroniki seira (thesi) tou kathe frame
        self.positional_encoding = PositionalEncoding(d_model=d_model,max_length=max_length)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model,nhead=num_heads,
        dim_feedforward=dim_feedforwardnetwork,dropout=percentage_zero_dropout,batch_first=True)
        
        #dimiourgoume ena encoder layer
        #d_model=posa landmarks exei kathe input frame
        #nhead=number of attention heads 
        #batvh_first= ta dedomena erconati se batch size (batch_sizr,number of frames,landmarks)
        
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer,num_layers=encoder_layers, enable_nested_tensor=False) 
        #dimiourgoume encoder me encoder_layers layers
        #kathe layer epeksergazete thn exodo tou proigoumenou 
        # enable_nested_tensor=False giati xoris afto exo error.Simenei oti mia sigkekrimeni litourgia veltistopoihshs tou pytorch transformer den ipostirizete apo to mps backend tou mac
        #ara thn apenergopoioume 
        
        self.classifier = nn.Linear(d_model, num_glosses)  #6 metatrepw thn anaparastasi twn 128 xar/kwn se bathmologies (called logits) sto dictionary (305 glosses/leksis) 
        
        
        
        
        
        
        
        
        
        
        
    def forward(self, landmarks,not_padding_frames): #einai h methodos poy kathorizei pws mpainoun ta dedomena mesa sto modelo
        #self: to antikimeno tou montelou
        #landmarks:input p dinoume sto modelo diladi ta landmarks (30,40,292)
        frame_embeddings = self.input_projection(landmarks) # kalite h sinartisi ston constructor me to input mas
        frame_embeddings_position = self.positional_encoding(frame_embeddings) #pe landmarks
        
        
        #GIA NA MHN METROUN STOUS IPOLOGISMOUS TOU SELF-ATTENTION TA PADDING MASKS (FRAMES WITH (0,0) LANDMARKS) 
        #DIMIOURGOME ENA BOOLEAN PINAKA PADDING MASK GIA KATHE FRAME (32BATCH SIZE,40FRAMES EACH) OPOU AN FALSE TOTE EINAI REAL FRAMES
        #KAI TRUE EINAI PADDING (NOT TAKE INTO CONSIDERATION)
        B, T = frame_embeddings_position.size(0), frame_embeddings_position.size(1) #batch, frames

        padding_mask = torch.zeros(B, T, dtype=torch.bool, device=frame_embeddings_position.device)  #(32,40) all false
        #device: to padding mask dimiourgeite sthn idia siskevi me ta dedomena (frame_embeddings_position)

        for b in range(B): #gloss 0,1,...,31
            padding_mask[b, not_padding_frames[b]:] = True 
        #apo to telos twn real frames kai meta=true 
        #stin grammi 0,1,2 gia to kathe gloss vale true apo to px 32 (teleftaio true frame) kai meta
                
        
        # encoded_frames = self.transformer_encoder(frame_embeddings_position,src_key_padding_mask=padding_mask) #perna ta frame_embeddings_position ston encoder
        attention_padding_mask = torch.zeros_like(
            padding_mask,
            dtype=frame_embeddings_position.dtype
        )

        attention_padding_mask = attention_padding_mask.masked_fill(
            padding_mask,
            -1e4
        )

        encoded_frames = self.transformer_encoder(
            frame_embeddings_position,
            src_key_padding_mask=attention_padding_mask
        )
        
        #ME TIN KANONIKI ENTOLI (AFINONTAS TO PADDING MASK T/F KAI ISAGONTAS TO STO TRANSFORMER) KANO TO EKSIS (LISI APO CHAT)
        # What does this change do?
        # - padding_mask remains a Boolean mask: True means that a frame is padding.
        # - attention_padding_mask converts it into a floating-point mask: 0 for valid frames and -10000 for padding.
        # - The Transformer adds this mask to its attention scores, strongly discouraging it from attending to padding frames.
                
        #pooling apo (32,40,128) -> (32,128) giati theloume dianisma gia kathe video/gloss oxi gia kathe frame ksexorista
        #there are many pooling methods,the most basics are:
        # Mean pooling: Average all token embeddings into a single vector
        # CLS token pooling: Use the special [CLS] token’s embedding as the document representation
        # Max pooling: Take the maximum value for each dimension across all tokens
        # Weighted pooling: Assign different importance to different tokens (e.g., using attention weights)
        #arxika xrisimopoiw to mean pooling
        
        #midenizw tis theseis ton padding frames gia na min ta ipologisoume sto mo
        #lalei opou to mask einai true,midenise
        encoded_frames = encoded_frames.masked_fill(padding_mask.unsqueeze(-1),0.0) #prepei an kanw to padding mask (32,40)->(32,40,1) gia na mporo na kanw ton midenismo sta 128 xar/ka tou kathe frame 

        #iplogizoume to mo:to sum ginetai gia kathe gloss (32,40,128)->(32,128) sum xar/ka kathe frame
        #diairoume me to plithos ton pragmatikwn frames
        pooling_vector = encoded_frames.sum(dim=1) / not_padding_frames.unsqueeze(1)

        #prediction tou kathe video sto gloss/leksi 
        gloss_logits = self.classifier(pooling_vector)

        
        
        
        return gloss_logits
    
    

    #an grayoume model(x) h pytorch kalei aftomata tin sinartisi forward (modelo)

