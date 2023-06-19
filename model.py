import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


# class Transformer_model(nn.Module):
class scGREAT(nn.Module):
    def __init__(self,expression_data_shape, embed_size, num_layers, num_head, biobert_embedding_path): 
        super(scGREAT, self).__init__()

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.encoder_layer = nn.TransformerEncoderLayer(d_model=embed_size, nhead=num_head, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers) 

        self.biobert = np.load(biobert_embedding_path)[1:]
        self.biobert_embedding = nn.Embedding.from_pretrained(torch.from_numpy(self.biobert)) 
        self.position_embedding = nn.Embedding(2, embed_size)

        self.encoder512 = nn.Linear(expression_data_shape[1], 512)
        self.encoder768 = nn.Linear(512, embed_size)
        
        self.flatten = nn.Flatten()
        self.linear1024 = nn.Linear(1536, 1024) 
        self.layernorm1024 = nn.LayerNorm(1024) 
        self.batchnorm1024 = nn.BatchNorm1d(1024)

        self.linear512 = nn.Linear(1024, 512)
        self.layernorm512 = nn.LayerNorm(512) 
        self.batchnorm512 = nn.BatchNorm1d(512)

        self.linear256 = nn.Linear(512, 256)
        self.layernorm256 = nn.LayerNorm(256)
        self.batchnorm256 = nn.BatchNorm1d(256)

        self.linear2 = nn.Linear(256, 1)
        self.actf = nn.PReLU()
        self.dropout = nn.Dropout(p=0.2)
        self.softmax = nn.Softmax(dim=1)
        self.pool = nn.AvgPool1d(kernel_size=4, stride=4)  


    def forward(self, gene_pair_index,expr_embedding): 
        bs = expr_embedding.shape[0]
        position = torch.Tensor([0,1]*bs).reshape(bs,-1).to(torch.int32)
        position.to(self.device)
        p_e = self.position_embedding(position)
    
        out_expr_e = self.encoder512(expr_embedding)
        out_expr_e = F.leaky_relu(self.encoder768(out_expr_e)) 
        b_e = self.biobert_embedding(gene_pair_index)
        input_ = torch.add(out_expr_e, torch.add(b_e, p_e))  
        out = self.transformer_encoder(input_)
        out = self.flatten(out) 
        
        out = self.linear1024(out)
        out = self.dropout(out)
        out = self.actf(out)

        r = out.unsqueeze(1)
        r = self.pool(r)
        r = r.squeeze(1)

        out = self.linear512(out)
        out = self.dropout(out)
        out = self.actf(out)

        out = self.linear256(out) + r
        out = self.dropout(out)
        out = self.actf(out)

        outs = self.linear2(out)
        outs = nn.Sigmoid()(outs)

        return outs
