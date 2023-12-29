
import pandas as pd
import numpy as np
import os
import time
import h5py
import random
from tqdm import tqdm
import subprocess
from sh import get_embedding_sh


def seed_everything(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)

seed = 42
seed_everything(seed)



def Hard_Negative_Specific_train_test_val(label_file, Gene_file, TF_file, train_set_file, val_set_file, test_set_file,
                                          ratio=0.67, p_val=0.5):
    label = pd.read_csv(label_file, index_col=0) 
    tf_set = pd.read_csv(TF_file, index_col=0)['index'].values  
    gene_set = pd.read_csv(Gene_file, index_col=0)['index'].values 

    tf = label['TF'].values  
    tf_list = np.unique(tf) 

    pos_dict = {}
    for i in tf_list:
        pos_dict[i] = []
    for i, j in label.values: 
        pos_dict[i].append(j)

    neg_dict = {}
    for i in tf_set:
        neg_dict[i] = []

    for i in tf_set:
        if i in pos_dict.keys(): 
            pos_item = pos_dict[i]
            pos_item.append(i)
            pos_dict[i] = np.setdiff1d(pos_dict[i], i) 
            
            neg_item = np.setdiff1d(gene_set, pos_item) 
            neg_dict[i].extend(neg_item)
            # pos_dict[i] = np.setdiff1d(pos_dict[i], i)
        else:
            neg_item = np.setdiff1d(gene_set, i)
            neg_dict[i].extend(neg_item)


    train_pos = {}
    val_pos = {}
    test_pos = {}
    for k in pos_dict.keys(): 
        if len(pos_dict[k]) ==1:
            p = np.random.uniform(0,1) 
            if p <= p_val:  
                train_pos[k] = pos_dict[k]
            else:
                test_pos[k] = pos_dict[k]
        elif len(pos_dict[k]) ==2:
            np.random.shuffle(pos_dict[k])
            train_pos[k] = [pos_dict[k][0]]
            test_pos[k] = [pos_dict[k][1]]
        else:
            np.random.shuffle(pos_dict[k])
            train_pos[k] = pos_dict[k][:int(len(pos_dict[k])*ratio)]
            val_pos[k] = pos_dict[k][int(len(pos_dict[k])*ratio):int(len(pos_dict[k])*(ratio+0.1))]
            test_pos[k] = pos_dict[k][int(len(pos_dict[k])*(ratio+0.1)):]


    train_neg = {}
    val_neg = {}
    test_neg = {}
    for k in pos_dict.keys():
        neg_num = len(neg_dict[k])
        np.random.shuffle(neg_dict[k])
        train_neg[k] = neg_dict[k][:int(neg_num*ratio)]
        val_neg[k] = neg_dict[k][int(neg_num*ratio):int(neg_num*(0.1+ratio))]
        test_neg[k] = neg_dict[k][int(neg_num*(0.1+ratio)):]

    train_pos_set = []
    for k in train_pos.keys():
        for val in train_pos[k]:
            train_pos_set.append([k,val])
    train_neg_set = []
    for k in train_neg.keys():
        for val in train_neg[k]:
            train_neg_set.append([k,val])
    train_set = train_pos_set + train_neg_set
    print('train pos:neg = {}:{}'.format(len(train_pos_set), len(train_neg_set)),round(len(train_pos_set)/len(train_neg_set),4))
    train_label = [1 for _ in range(len(train_pos_set))] + [0 for _ in range(len(train_neg_set))]
    train_sample = np.array(train_set)

    train = pd.DataFrame()
    train['TF'] = train_sample[:, 0]
    train['Target'] = train_sample[:, 1]
    train['Label'] = train_label
    train.to_csv(train_set_file)

    val_pos_set = []
    for k in val_pos.keys():
        for val in val_pos[k]:
            val_pos_set.append([k,val])
    val_neg_set = []
    for k in val_neg.keys():
        for val in val_neg[k]:
            val_neg_set.append([k,val])
    val_set = val_pos_set + val_neg_set
    print('val pos:neg = {}:{}'.format(len(val_pos_set), len(val_neg_set)),round(len(val_pos_set)/len(val_neg_set),4))
    val_label = [1 for _ in range(len(val_pos_set))] + [0 for _ in range(len(val_neg_set))]
    val_sample = np.array(val_set)
    val = pd.DataFrame()
    val['TF'] = val_sample[:, 0]
    val['Target'] = val_sample[:, 1]
    val['Label'] = val_label
    val.to_csv(val_set_file)


    test_pos_set = []
    for k in test_pos.keys():
        for j in test_pos[k]:
            test_pos_set.append([k,j])

    test_neg_set = []
    for k in test_neg.keys():
        for j in test_neg[k]:
            test_neg_set.append([k,j])
    test_set = test_pos_set +test_neg_set
    print('test pos:neg = {}:{}'.format(len(test_pos_set), len(test_neg_set)),round(len(test_pos_set)/len(test_neg_set),4))
    test_label = [1 for _ in range(len(test_pos_set))] + [0 for _ in range(len(test_neg_set))]
    test_sample = np.array(test_set)
    test = pd.DataFrame()
    test['TF'] = test_sample[:,0]
    test['Target'] = test_sample[:,1]
    test['Label'] = test_label
    test.to_csv(test_set_file)




def train_val_test_set(label_file,Gene_file,TF_file,train_set_file,val_set_file,test_set_file,density,p_val=0.5):
    gene_set = pd.read_csv(Gene_file, index_col=0)['index'].values
    tf_set = pd.read_csv(TF_file, index_col=0)['index'].values
    label = pd.read_csv(label_file, index_col=0)
    tf = label['TF'].values

    tf_list = np.unique(tf)
    pos_dict = {}
    for i in tf_list:
        pos_dict[i] = []
    for i, j in label.values:
        pos_dict[i].append(j)

    train_pos = {}
    val_pos = {}
    test_pos = {}

    for k in pos_dict.keys():
        if len(pos_dict[k]) <= 1:
            p = np.random.uniform(0,1)
            if p <= p_val:
                train_pos[k] = pos_dict[k]
            else:
                test_pos[k] = pos_dict[k]
        elif len(pos_dict[k]) == 2:
            train_pos[k] = [pos_dict[k][0]]
            test_pos[k] = [pos_dict[k][1]]
        else:
            np.random.shuffle(pos_dict[k])
            train_pos[k] = pos_dict[k][:len(pos_dict[k]) * 2 // 3]
            test_pos[k] = pos_dict[k][len(pos_dict[k]) * 2 // 3:]

            val_pos[k] = train_pos[k][:len(train_pos[k])//5]
            train_pos[k] = train_pos[k][len(train_pos[k])//5:]

    train_neg = {}
    for k in train_pos.keys():
        train_neg[k] = []
        for i in range(len(train_pos[k])):
            neg = np.random.choice(gene_set)
            while neg == k or neg in pos_dict[k] or neg in train_neg[k]:
                neg = np.random.choice(gene_set)
            train_neg[k].append(neg)


    train_pos_set = []
    train_neg_set = []
    for k in train_pos.keys():
        for j in train_pos[k]:
            train_pos_set.append([k, j])
    tran_pos_label = [1 for _ in range(len(train_pos_set))]

    for k in train_neg.keys():
        for j in train_neg[k]:
            train_neg_set.append([k, j])
    tran_neg_label = [0 for _ in range(len(train_neg_set))]


    train_set = train_pos_set + train_neg_set
    train_label = tran_pos_label + tran_neg_label

    train_sample = train_set.copy()
    for i, val in enumerate(train_sample):
        val.append(train_label[i])
    train = pd.DataFrame(train_sample, columns=['TF', 'Target', 'Label'])
    train.to_csv(train_set_file)


    val_pos_set = []
    for k in val_pos.keys():
        for j in val_pos[k]:
            val_pos_set.append([k, j])
    val_pos_label = [1 for _ in range(len(val_pos_set))]

    val_neg = {}
    for k in val_pos.keys():
        val_neg[k] = []
        for i in range(len(val_pos[k])):
            neg = np.random.choice(gene_set)
            while neg == k or neg in pos_dict[k] or neg in train_neg[k] or neg in val_neg[k]:
                neg = np.random.choice(gene_set)
            val_neg[k].append(neg)


    val_neg_set = []
    for k in val_neg.keys():
        for j in val_neg[k]:
            val_neg_set.append([k,j])


    val_neg_label = [0 for _ in range(len(val_neg_set))]
    val_set = val_pos_set + val_neg_set
    val_set_label = val_pos_label + val_neg_label

    val_set_a = np.array(val_set)
    val_sample = pd.DataFrame()
    val_sample['TF'] = val_set_a[:,0]
    val_sample['Target'] = val_set_a[:,1]
    val_sample['Label'] = val_set_label
    val_sample.to_csv(val_set_file)


    test_pos_set = []
    for k in test_pos.keys():
        for j in test_pos[k]:
            test_pos_set.append([k, j])

    count = 0
    for k in test_pos.keys():
        count += len(test_pos[k])
    test_neg_num = int(count // density-count)
    test_neg = {}
    for k in tf_set:
        test_neg[k] = []

    test_neg_set = []
    for i in range(test_neg_num):
        t1 = np.random.choice(tf_set)
        t2 = np.random.choice(gene_set)
        while t1 == t2 or [t1, t2] in train_set or [t1, t2] in test_pos_set or [t1, t2] in val_set or [t1,t2] in test_neg_set:
            t2 = np.random.choice(gene_set)

        test_neg_set.append([t1,t2])

    test_pos_label = [1 for _ in range(len(test_pos_set))]
    test_neg_label = [0 for _ in range(len(test_neg_set))]

    test_set = test_pos_set + test_neg_set
    test_label = test_pos_label + test_neg_label
    for i, val in enumerate(test_set):
        val.append(test_label[i])

    test_sample = pd.DataFrame(test_set, columns=['TF', 'Target', 'Label'])
    test_sample.to_csv(test_set_file)



def gen_biobert_name(net_type_i,data_type_i):
    num = [500,1000]
    data_root_path = '/Dataset/' + net_type_i + ' Dataset'
    # for data_type_i in data_type:
    for num_i in num:
        Gene2file = data_root_path + '/' + data_type_i + '/TFs+' + str(num_i) + '/Target.csv'  
        father_dir = os.getcwd() + '/' + 'data_split' + '/' + net_type_i + '/' + data_type_i + '/TFs' + '_' + str(num_i)
        if not os.path.exists(father_dir): os.makedirs(father_dir)
        bio_name_file = father_dir + '/bio_name.txt' 
        gene = pd.read_csv(Gene2file,index_col=0,header=0)
        print(len(gene['Gene']) == len(set(gene['Gene'])))
        gene['Gene'].to_csv(bio_name_file, sep='\t', index=False)
    print('gen biobert name finished')


def data_split(net_type_i,data_type_i):
    num = [500,1000]
    
    for num_i in num:
        print('num',num_i)
        
        TF2file = data_type_i+str(num_i)+'/TF.csv'  #'/BL--ExpressionData.csv'
        Gene2file = data_type_i+str(num_i)+'/Target.csv'
        label_file = data_type_i+str(num_i)+'/Label.csv'

        father_dir = data_type_i+str(num_i)
        if not os.path.exists(father_dir): os.makedirs(father_dir)

        train_set_file = father_dir + '/Train_set.csv'
        val_set_file   = father_dir + '/Validation_set.csv'
        test_set_file  = father_dir + '/Test_set.csv'
        print('train_set_filepath:',train_set_file)
        Hard_Negative_Specific_train_test_val(label_file,     Gene2file,    TF2file,
                                              train_set_file, val_set_file, test_set_file)

    print('data split finished')



def load_embedding(net_type_i,data_type_i):
    num = [500,1000]
    data_root_path = '/data_split/' + net_type_i
    for num_i in num:
        father_dir = data_root_path + '/' + data_type_i + '/TFs_'+str(num_i)
        biovec_path = father_dir + '/bio_name_emb768.h5'
        genename_path = father_dir + '/bio_name.txt'
        dic = {}
        with h5py.File(biovec_path, 'r') as f:
            with open(genename_path, 'r') as f_in:
                print("The number of keys in h5: {}".format(len(f)))
                for i, input in enumerate(f_in):
                    entity_name = input.strip()
                    embedding = f[entity_name]['embedding'][:]
                    dic[entity_name] = embedding
        biovect = np.array(list(dic.values()))
        save_path = father_dir +'/biovect768.npy'
        np.save(save_path, biovect)
    print('load embedding finished')



def data_move(net_type_i,data_type_i):
    import os
    import shutil

    source_folder = '/Benchmark Dataset/' + net_type_i + ' Dataset'
    destination_folder = 'data_split/' + net_type_i

    num = [500,1000]
    files_to_copy = ['BL--ExpressionData.csv', 'BL--network.csv', 'Label.csv', 'Target.csv', 'TF.csv']


    for num_i in num:
        for file_i in files_to_copy:
            source_file_path = source_folder + '/' + data_type_i + '/TFs+' + str(num_i) + '/' + file_i
            destination_file_path = destination_folder + '/' + data_type_i + '/TFs_' + str(num_i) + '/' + file_i
            if os.path.exists(source_file_path):
                shutil.copy2(source_file_path, destination_file_path)
                # print(f'copyfiles: {source_file_path} -> {destination_file_path}')
            else:
                print(f'The source file does not exist: {source_file_path}')
    print('data move finished')



def causual_neg(dataset):
    print('original shape',dataset.shape)
    a = dataset[dataset['Label']==1]  
    print('positive shape',a.shape)
    a['TF_new'] = a['TF']
    a['TF'] = a['Target']
    a['Target'] = a['TF_new']
    a['Label'] = 2
    a = a.drop('TF_new', axis=1)
    dataset = pd.concat([dataset,a],axis=0)
    print('final shape',dataset.shape)
    return dataset


def casual_inference(net_type_i,data_type_i): 
    num = [500,1000]
    
    for num_i in num:
        print('num',num_i)
        father_dir = data_type_i+str(num_i)
        if not os.path.exists(father_dir): os.makedirs(father_dir)


        train_set_file = father_dir + '/Train_set.csv'
        val_set_file   = father_dir + '/Validation_set.csv'
        test_set_file  = father_dir + '/Test_set.csv'
        train_set = pd.read_csv(train_set_file, header=0, index_col=0)
        val_set   = pd.read_csv(val_set_file,   header=0, index_col=0)
        test_set  = pd.read_csv(test_set_file,  header=0, index_col=0)
        
        train_set = causual_neg(train_set)
        val_set   = causual_neg(val_set)
        test_set  = causual_neg(test_set)
        train_set_file_c = father_dir + '/Train_set_c.csv'
        val_set_file_c   = father_dir + '/Validation_set_c.csv'
        test_set_file_c  = father_dir + '/Test_set_c.csv'
        
        train_set.to_csv(train_set_file_c)
        val_set.to_csv(val_set_file_c)
        test_set.to_csv(test_set_file_c)


if __name__ == '__main__':

    casual_flag = True 
    net_type = ['Specific','Non-Specific', 'STRING'] 
    data_type = ['hESC','hHEP','mDC','mESC','mHSC-E','mHSC-GM','mHSC-L'] 
    for net_type_i in net_type:
        for data_type_i in data_type:
            print(net_type_i,data_type_i)

            # step1: Generate bio_name.txt file
            print('step1:')
            gen_biobert_name(net_type_i,data_type_i) 

            # # step2: Generate bio_name_emb768.h5 file    biovec 768
            print('step2:')
            get_embedding_sh(net_type_i,data_type_i)
            # # system sleep
            # print('sleep 100s')
            time.sleep(100)

            # # step3: Generate biovect768.npy file  
            print('step3:')
            load_embedding(net_type_i,data_type_i)

            # # step4: Generate training, validation, and testing sets
            print('step4:')
            data_split(net_type_i,data_type_i)
            if casual_flag: 
                casual_inference(net_type_i,data_type_i)
            
            # # step5: move all files to the datasplit folder  optional
            print('step5:')
            data_move(net_type_i,data_type_i)

