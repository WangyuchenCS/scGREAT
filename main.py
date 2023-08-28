
import numpy as np
import pandas as pd
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
from dataset import Dataset
from model import scGREAT
from train_val import train,validate


def main(data_dir, args):
    # data_dir = 'hESC500'
    expression_data_path = data_dir + '/BL--ExpressionData.csv'
    biovect_e_path       = data_dir + '/biovect768.npy'
    train_data_path      = data_dir + '/Train_set.csv'
    val_data_path        = data_dir + '/Validation_set.csv'
    test_data_path       = data_dir + '/Test_set.csv'
    expression_data = np.array(pd.read_csv(expression_data_path,index_col=0,header=0))

    # Data Preprocessing
    standard = StandardScaler()
    scaled_df = standard.fit_transform(expression_data.T)
    expression_data = scaled_df.T
    expression_data_shape = expression_data.shape 

    train_dataset = Dataset(train_data_path, expression_data)
    val_dataset = Dataset(val_data_path, expression_data)
    test_dataset = Dataset(test_data_path, expression_data)

    # Model parameters
    Batch_size = args.batch_size
    Embed_size = args.embed_size
    Num_layers = args.num_layers
    Num_head = args.num_head
    LR = args.lr
    EPOCHS = args.epochs
    step_size = args.step_size
    gamma = args.gamma
    global schedulerflag
    schedulerflag = args.scheduler_flag

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=Batch_size,
                                               shuffle=True,
                                               drop_last=False,
                                               num_workers=0)
    
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                             batch_size=Batch_size,
                                             shuffle=True,
                                             drop_last=False)
    
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=Batch_size,
                                              shuffle=True,
                                              drop_last=False)



    T = scGREAT(expression_data_shape,Embed_size,Num_layers,Num_head,biovect_e_path)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    T = T.to(device)
    optimizer = torch.optim.Adam(T.parameters(), lr=LR)
    scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
    loss_func = nn.BCELoss() 


    for epoch in range(1, EPOCHS + 1):
        train(T, train_loader, loss_func, optimizer, epoch, scheduler, args)
        AUC_val,AUPR_val = validate(T,val_loader,loss_func)
        print('-' * 100)
        print('| end of epoch {:3d} |valid AUROC {:8.3f} | valid AUPRC {:8.3f}'.format(epoch,AUC_val,AUPR_val))
        print('-' * 100)
        AUC_test,AUPR_test = validate(T,test_loader,loss_func)
        print('| end of epoch {:3d} |test  AUROC {:8.3f} | test  AUPRC {:8.3f}'.format(epoch,AUC_test,AUPR_test))
        print('-' * 100)

        if AUC_val<0.501:
            print("AUC_val<0.501 !!")
            break

