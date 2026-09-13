import torch
import torch.nn as nn


class ISLRTransformer(nn.Module):

    def __init__(
        self,
        input_dim=292,      #landmarks kathe frame (landmarks x 2 (x,y))
        hidden_dim=256,     #metatropi twn landmarks 
        num_heads=8,        # posa attention heads. kathe kefali koitaei ta dedomena apo diaforetiki gonia gia na vgalei diaforetiki sxesi metaksi twn landmarks
        num_layers=4,       #transformer layers
        num_classes=310,    #dictionary size se poses leksis (id) mporei na antistixithei to gloss
        max_length=30,      # max frame length
        dropout=0.3,       # Dropout in a transformer is a regularization technique that 
        #randomly turns off a fraction of neurons or connection weights during training to prevent the model from memorizing the data.
        pooling="class_query"  #pooling
    ):
        super().__init__()

        self.pooling = pooling

    
#linear layer apo input_dim landmarks se hidder_dim landmarks 
        self.input_projection = nn.Linear(input_dim,hidden_dim)


#positional encoding: kathe frame pernei index for the model to have the sense of seira kai oxi san ena sinolo ola t frames
        self.positional_encoding = nn.Parameter(
            torch.randn(max_length, hidden_dim)
        )


#input dropout: tixaia svinei merikous arithmous kata to training gia na min stiriezetai to montelo poli se sigkekrimena landmarks
        self.input_dropout = nn.Dropout(dropout)

#transformer encoder: self-attention mechanism vlepontas oles tis sxeseis metaksi ton frames 
#num_layers: transformer layers how many times this process is repeated
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim,nhead=num_heads,dropout=dropout,batch_first=True)

        self.transformer_encoder = nn.TransformerEncoder(encoder_layer,num_layers=num_layers,enable_nested_tensor=False)


#class query pooling instead of e.g. mean polling reference: https://ieeexplore.ieee.org/document/9707552
#afto xreiazete giati kathe frame exei ena "dianisma" alla gia na exoume to teliko classification xreiazomaste ena dianisma gia olo to gloss
        if self.pooling == "class_query":

# to "erotima" mas - ena mono learnable dianysma pou tha mathei na "rotaei" ta frames poio einai simantiko
            self.class_query = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)

# etoimo attention tou pytorch. to erotima (query) tha "koitaei" ola ta frames (key/value) kai tha vgalei ena dianysma me to
# poso simantiko einai to kathe frame
            self.class_attention = nn.MultiheadAttention(embed_dim=hidden_dim,num_heads=num_heads,dropout=dropout,batch_first=True)

# kanonikopoiisi klimakas meta to attention
            self.pool_norm = nn.LayerNorm(hidden_dim)


# classifier: pairnei to teliko dianysma (256 arithmoi) kai to metatrepei se 310 arithmous. ena "skor" gia kathe pithano noima/leksi
# To pio psilo skor = i apantisi tou montelou

        self.classifier_dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim,num_classes)

    def forward(self, x, lengths, use_mask=True):
        
# x : [batch, frames, landmarks, coordinates]
# batch = number of glosses
        batch_size, seq_len, num_points, coords = x.shape

#landmarks * coordinates
        x = x.reshape(
            batch_size,
            seq_len,
            num_points * coords
        )
#input projection
        x = self.input_projection(x)
        
#index in each frame 
        x = x + self.positional_encoding[:seq_len]

#dropout
        x = self.input_dropout(x)


#padding mask. gia na kserei to modelo oti afta den einai real frames (padding) me landmarks (0,0)
#true/false: true padding false real frame
        lengths = lengths.to(x.device).unsqueeze(1)
        lengths = lengths.clamp(min=1)  # den einai olo to gloss padded (min=1)

        if use_mask: #use mask =true 
            positions = torch.arange(seq_len,device=x.device).unsqueeze(0)

            padding_mask = positions >= lengths
        else:
            padding_mask = torch.zeros(batch_size, seq_len,dtype=torch.bool,device=x.device)



        x = self.transformer_encoder(x)



        if self.pooling == "class_query":
# Pairnoume to "erotima" mas kai to "antigrafoume" gia kathe video tou batch
            query = self.class_query.expand(batch_size, -1, -1)

# To erotima "koitaei" ola ta frames tou video kai apofasizei poia einai simantika (agnooontas ta padding eksetias sto padding_mask)
# query = to "erotima" [B,1,H] | key/value = ta frames meta ton encoder [B,S,H]
# key_padding_mask: True = agnoise auto to frame (padding)
            x, _ = self.class_attention(
                query,
                x,
                x,
                key_padding_mask=padding_mask
            )
            x = x.squeeze(1)  # [B, 1, H] -> [B, H]

            # kanonikopoioume tin klimaka
            x = self.pool_norm(x)

        # else:  
        #     if use_mask:
        #         valid_mask = (~padding_mask).unsqueeze(-1)

        #   
        #         x = x.masked_fill(
        #             padding_mask.unsqueeze(-1),
        #             0.0
        #         )

        #         # apla mesos oros MONO ton pragmatikon frames
        #         x = x.sum(dim=1) / valid_mask.sum(
        #             dim=1
        #         ).clamp(min=1)
        #     else:
        #         x = x.mean(dim=1)

# dropout prin to teliko classifier

        x = self.classifier_dropout(x)

# teliko skor - ena skor gia kathe noima
        logits = self.classifier(x)

        return logits