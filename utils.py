import torch
import numpy as np
import pandas as pd
import sys
import os
import random
from datetime import datetime
from sklearn.metrics import roc_auc_score,average_precision_score


class Logger(object):
    timenow = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_name = timenow +'_T_new_save.log'
    def __init__(self, filename=log_name, stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, 'a')
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message) 
    def flush(self):
        pass


def write_data_to_file(name, data):
    with open(f"{name}.txt", "a") as f:
        f.write(data + "\n")


def seed_everything(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    if torch.cuda.is_available(): 
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True


def Evaluation(y_true, y_pred,flag=False):
    if flag:
        y_p = y_pred[:,-1]
        y_p = y_p.cpu().detach().numpy()
        y_p = y_p.flatten()
    else:
        y_p = y_pred.cpu().detach().numpy()
        y_p = y_p.flatten()
    y_t = y_true.cpu().numpy().flatten().astype(int)

    try:
        AUROC = roc_auc_score(y_true=y_t, y_score=y_p)
        AUPRC = average_precision_score(y_true=y_t,y_score=y_p)
    except Exception as e:
        AUROC = 0
        AUPRC = 0

    return AUROC, AUPRC
