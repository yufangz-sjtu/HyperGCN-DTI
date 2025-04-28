
import yaml
import warnings
from utils import *
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score, auc
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import classification_report
from sklearn.utils import shuffle
import torch
import torch.nn.functional as F

warnings.filterwarnings("ignore")

with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)
config['device'] = "cuda" if torch.cuda.is_available() else "cpu"
#config['device'] = "cpu"

setup(config['seed'])


reg_loss_co = 0.0002
fold = 0

torch.set_default_dtype(torch.float32)



def ml_classifiers(name,clf):
    node_num, drug_protein, protein_drug, dtidata,features_d,features_p,HyGraph_Drug,HyGraph_protein = load_feature(name,config['feature_list'])
    #print(dtidata)
    #dti_label = F.one_hot(torch.tensor(dtidata[:, 2:3]), num_classes=2)
    #dti_label = torch.squeeze(dti_label, dim=1)
    
    dti_label = torch.tensor(dtidata[:, 2:3])
    dti_label = dti_label.squeeze() 
    #print(dti_label)
    indices_features1 = dtidata[:, 0]  # 提取第一个 Tensor 的行索引
    indices_features2 = dtidata[:, 1]  # 提取第二个 Tensor 的行索引
    selected_features1 = features_d[indices_features1]
    selected_features2 = features_p[indices_features2]
    features = torch.cat((selected_features1, selected_features2), dim=1)
    features, dti_label = shuffle(features, dti_label, random_state=42)
    #print(features.shape,dti_label.shape)  # 检查拼接后的 Tensor 形状
    y_pred = cross_val_predict(clf, features.cpu() , dti_label, cv=5)
    y_prob =  cross_val_predict(clf, features.cpu() , dti_label, cv=5, method='predict_proba')
    #print(y_prob)
    #print(y_prob[1])
    #print(y_prob.shape)
    print('训练完成!')
        
    prec_reca_f1_supp_report = classification_report(dti_label, y_pred, target_names = ['label_0', 'label_1'])
    tn, fp, fn, tp = confusion_matrix(dti_label, y_pred).ravel()
    accuracy = (tp+tn)/(tp+fp+tn+fn)
    recall = tp / (tp+fn)
    precision = tp / (tp+fp)
    f1 = 2*precision*recall / (precision+recall)
    roc_auc = roc_auc_score(dti_label, y_prob[:, 1])
    prec, reca, _ = precision_recall_curve(dti_label, y_prob[:, 1])
    aupr = auc(reca, prec)
    '--------------------------------------------打印&输出--------------------------------------------------------------'
    print(f'{name},{clf},results:')
    print('acc={:.4f}|precision={:.4f}|recall={:.4f}|f1={:.4f}|auc={:.4f}|aupr={:.4f}'.format(accuracy, precision, recall, f1, roc_auc, aupr))
    #print('tn = {}, fp = {}, fn = {}, tp = {}'.format(tn, fp, fn, tp))
    print('-----------------------------------------------------------------')

rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
gb_clf = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
svm_clf = SVC(kernel='linear', C=1, probability=True)
mlp_clf = MLPClassifier(hidden_layer_sizes=(128,), max_iter=500, random_state=42)
for clf in [rf_clf,svm_clf,mlp_clf,gb_clf]:
    #for name in ['Luo',"Es","GPCRs","ICs","NRs",'Zheng']:
    for name in ["GPCRs","ICs"]:
        ml_classifiers(name,clf)
  


  