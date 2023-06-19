
import torch
from utils import Evaluation

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(model,dataloader,loss_func,optimizer,epoch,scheduler,args):
    model.train()
    log_interval = 200
    total_loss = 0

    for idx, (gene_pair_index,expr_embedding,label) in enumerate(dataloader):

        expr_embedding = expr_embedding.to(torch.float32)
        label.to(device)
        gene_pair_index.to(device)
        expr_embedding.to(device)
        optimizer.zero_grad()
        predicted_label = model(gene_pair_index,expr_embedding)
        loss = loss_func(predicted_label.squeeze(), label.float())
        total_loss += loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        optimizer.step()

        if args.scheduler_flag:
            scheduler.step()

        if idx % log_interval == 0 and idx > 0:
            AUROC, AUPRC = Evaluation(y_pred=predicted_label, y_true=label) 
            print('| epoch {:3d} | {:5d} /{:5d} batches |Train loss {:8.3f} | AUROC {:8.3f} | AUPRC {:8.3f}'.format(epoch,idx,len(dataloader),loss,AUROC,AUPRC))

    print('| epoch {:3d} | total_loss {:8.3f}'.format(epoch, total_loss))



def validate(model,dataloader,loss_func):
    model.eval()
    with torch.no_grad():
        pre = []
        lb = []

        for idx, (gene_pair_index,expr_embedding,label) in enumerate(dataloader):
            expr_embedding = expr_embedding.to(torch.float32)
            gene_pair_index.to(device)
            expr_embedding.to(device)            
            predicted_label = model(gene_pair_index,expr_embedding) 

            pre.extend(predicted_label)
            label.to(device)
            lb.extend(label)

        pre = torch.vstack(pre)
        lb = torch.vstack(lb)
        
        AUROC, AUPRC = Evaluation(y_pred=pre, y_true=lb)
    return AUROC,AUPRC