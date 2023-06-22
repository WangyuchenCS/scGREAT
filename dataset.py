import torch
import numpy as np
import pandas as pd

class Dataset(torch.utils.data.Dataset):
    def __init__(self,data_path,expression_data): 
        data = pd.read_csv(data_path,index_col=0,header=0)
        
        self.dataset = np.array(data.iloc[:,:2])
        label = np.array(data.iloc[:,-1])
        self.label = np.eye(2)[label]  
        self.label = label
        self.expression_data = expression_data
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, i):
        gene_pair_index = self.dataset[i] 
        gene1_expr = np.expand_dims(self.expression_data[gene_pair_index[0]],axis=0) 
        gene2_expr = np.expand_dims(self.expression_data[gene_pair_index[1]],axis=0) 
        expr_embedding = np.concatenate((gene1_expr,gene2_expr),axis=0)  
        label = self.label[i]
        return gene_pair_index,expr_embedding,label