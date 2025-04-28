from utils import *
import torch
import pandas as pd
import yaml
import warnings
from Structe_DPP_HyperGraph import HyGraph_Matrix_DPP_Structure
import torch.nn.functional as F
#import wandb
import os
import torch
import warnings
import numpy as np
from model import *
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from hypergraph_utils import generate_G_from_H
from hypergraph_utils import construct_H_with_KNN
from sklearn.metrics import roc_auc_score, f1_score
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics.pairwise import cosine_similarity as cos



warnings.filterwarnings("ignore")

with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)
config['device'] = "cuda" if torch.cuda.is_available() else "cpu"
#config['device'] = "cpu"

setup(config['seed'])


reg_loss_co = 0.0002
fold = 0

torch.set_default_dtype(torch.float32)





#for name in ["Es","GPCRs","ICs","NRs"]:
    # dataName: Luo Es GPCRs ICs NRs Zheng
for negative_ratio in [1,2,3,4,5]:
#for dr in [0.1,0.2,0.3,0.4,0.5]:
    print('negative_ratio:', negative_ratio)
    #print('dropout rate:',dr)
    #config['negative_ratio'] = negative_ratio
    for name in ['Luo','Zheng',"Es","GPCRs","ICs","NRs"]:
        node_num, drug_protein, protein_drug, dtidata,features_d,features_p,HyGraph_Drug,HyGraph_protein = load_feature(name,config['feature_list'],config['negative_ratio'])
        #[708, 1512] torch.Size([708, 1512]) torch.Size([1512, 708]) (3846, 3)
        print(node_num,drug_protein.shape,protein_drug.shape,dtidata.shape,features_d.shape,features_p.shape,HyGraph_Drug.shape,HyGraph_protein.shape)
        dti_label = torch.tensor(dtidata[:, 2:3]).to(config['device'])
        drug_protein = drug_protein.to(config['device'])
        protein_drug = protein_drug.to(config['device'])
        HyGraph_Structure_DPP = HyGraph_Matrix_DPP_Structure(dtidata, node_num[0], node_num[1],name )
        HyGraph_Structure_DPP = HyGraph_Structure_DPP.to(config['device'])

        data = dtidata
        label = dti_label

        def train(model, optim, train_index, test_index, epoch, fold):
            model.train()
            out, d, p = model(node_num, features_d, features_p, protein_drug, drug_protein, HyGraph_Drug, HyGraph_protein, train_index, data, HyGraph_Structure_DPP)
            #print(out.shape,d.shape,p.shape)
            train_acc = (out.argmax(dim=1) == label[train_index].reshape(-1).long()).sum(dtype=float) / torch.tensor(len(train_index), dtype=float)
            task1_roc = get_roc(out, label[train_index])
            reg = get_L2reg(model.parameters())
            loss = F.nll_loss(out, label[train_index].reshape(-1).long()) + reg_loss_co * reg
            #print(train_acc,loss)
            optim.zero_grad()
            loss.backward()
            optim.step()
            
            te_acc, te_task1_roc1, te_task1_pr, te_task_precision,te_task_recall,te_task1_f1 = main_test(model, d, p, test_index, epoch, fold)

            return loss.item(), train_acc, task1_roc, te_acc, te_task1_roc1, te_task1_pr, te_task_precision,te_task_recall,te_task1_f1


        def main_test(model, d, p, test_index, epoch, fold):
            model.eval()
            out = model(node_num, features_d, features_p, protein_drug, drug_protein, HyGraph_Drug, HyGraph_protein, test_index, data, HyGraph_Structure_DPP, iftrain=False, d=d, p=p)

            acc1 = (out.argmax(dim=1) == label[test_index].reshape(-1).long()).sum(dtype=float) / torch.tensor(len(test_index), dtype=float)
            task_roc = get_roc(out, label[test_index])
            task_precision,task_recall,task_pr = get_pr(out, label[test_index])
            task_f1 = get_f1score(out, label[test_index])
            # if epoch == 799:
            #     f = open(f"{fold}out.txt","w",encoding="utf-8")
            #     for o in  (out.argmax(dim=1) == label[test_index].reshape(-1)):
            #         f.write(f"{o}\n")
            #     f.close()
            return acc1, task_roc, task_pr, task_precision,task_recall,task_f1

        def main(tr, te, seed):
            results = []
            for i in range(len(tr)):
                f = open( os.path.join(config['results_dir'],f"{name}_{config['feature_list']}_{i}foldtrain.txt"), "w", encoding="utf-8")
                train_index = tr[i]
                for train_index_one in train_index:
                    f.write(f"{train_index_one}\n")
                test_index = te[i]
                f = open( os.path.join(config['results_dir'],f"{name}_{config['feature_list']}_{i}foldtest.txt"), "w", encoding="utf-8")
                for train_index_one in test_index:
                    f.write(f"{train_index_one}\n")
                #
                # if not os.path.isdir(f"{dir}"):
                #     os.makedirs(f"{dir}")

                model = HyperGCNDTI(
                    num_protein=node_num[1],
                    num_drug=node_num[0],
                    num_hidden1=config['in_size'],
                    num_hidden2=config['hidden_size'],
                    num_out=config['out_size'],
                    dropout = dr,
                    feature_list= config['feature_list']
                ).to(config['device'])
                    
                # model.load_state_dict(torch.load(f"{dir}/net{i}.pth"))
                optim = torch.optim.Adam(lr=config['lr'], weight_decay= float(config['weight_decay']), params=model.parameters())
                best_roc =0
                best_results = []
                for epoch in tqdm(range(config['epochs'])):
                    loss, train_acc, task1_roc, acc, task1_roc1, task1_pr,task1_precision,task1_recall, task1_f1 = train(model, optim, train_index, test_index, epoch, i)
                    if task1_roc1 > best_roc:
                        best_roc = task1_roc1
                        best_model_state = model.state_dict()  # Update the best model state
                        torch.save(best_model_state, os.path.join(config['save_dir'], f"{config['feature_list']}_dataset_{name}_best_model_fold_{i}_roc.pth"))
                        best_results = acc, task1_roc1, task1_pr,task1_precision,task1_recall, task1_f1
                        best_results = list(tuple(f"{value:.4f}" for value in best_results))

                results.append(best_results)

                print(f'{i} fold cv results:', f'dataset {name}, Acc is:{best_results[0]},  AUROC is:{best_results[1]}, AUPr is:{best_results[2]},\
    Precision is:{best_results[3]} , recall is :{best_results[4]},f1 is:{best_results[5]}')
            df_results = pd.DataFrame(results, columns=['Accuracy', 'AUROC', 'AUPr',"Precision","recall","f1"])
            print(f'5-fold cv averaged results:', f'dataset {name}')
            print( df_results.mean())
            df_results.to_csv(os.path.join(config['results_dir'],f"{name}_{config['feature_list']}_CV_resluts.csv"),index=False)
            

        train_indeces, test_indeces = get_cross(dtidata)
        main(train_indeces, test_indeces, config['seed'])