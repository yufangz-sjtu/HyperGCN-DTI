import numpy as np
import torch
import random
from scipy import sparse
from scipy import io as sio
import scipy.spatial.distance as dist
from sklearn.metrics import auc as auc3
from sklearn.metrics.pairwise import cosine_similarity as cos
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import precision_score, recall_score, precision_recall_curve, roc_auc_score, f1_score, auc
from hypergraph_utils import generate_G_from_H
import yaml
import pandas as pd

def set_random_seed(seed=0):
    """Set random seed.
    Parameters
    ----------
    seed : int
        Random seed to use
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def setup( seed):
    set_random_seed(seed)




def generate_negative_samples(dti_matrix):
    """
    根据药物-蛋白质相互作用矩阵生成正样本和负样本索引
    
    :param dti_matrix: 药物-蛋白质相互作用矩阵 (numpy array)
    :return: 正样本索引列表和负样本索引列表
    """
    positive_index = []
    negative_index = []

    for i in range(dti_matrix.shape[0]):
        for j in range(dti_matrix.shape[1]):
            if int(dti_matrix[i][j]) == 1:
                positive_index.append([i, j])
            else:
                negative_index.append([i, j])
    
    return positive_index, negative_index


def construct_dataset(train_positive_index, whole_negative_index, negative_ratio=1):
    """
    构建包含正样本和负样本的数据集
    
    :param train_positive_index: 正样本索引列表
    :param whole_negative_index: 负样本索引列表
    :param negative_ratio: 负样本与正样本的比例，默认为1
    :return: 数据集数组，包含正负样本
    """
    # 根据给定的负样本比例，计算需要的负样本数量
    num_negative_samples = min(len(whole_negative_index), len(train_positive_index) * negative_ratio)

    # 随机选择负样本
    negative_sample_index = np.random.choice(np.arange(len(whole_negative_index)),
                                             size=num_negative_samples,
                                             replace=False)

    # 初始化数据集数组
    data_set = np.zeros((len(train_positive_index) + num_negative_samples, 3), dtype=int)
    
    # 添加正样本到数据集中
    count = 0
    for i in train_positive_index:
        data_set[count][0] = i[0]
        data_set[count][1] = i[1]
        data_set[count][2] = 1
        count += 1

    # 添加负样本到数据集中
    for idx in negative_sample_index:
        data_set[count][0] = whole_negative_index[idx][0]
        data_set[count][1] = whole_negative_index[idx][1]
        data_set[count][2] = 0
        count += 1

    return data_set

def load_disease_info(network_path, drug_disease_filename, protein_disease_name,dataset_name="Luo"):
    """
    :param network_path: 数据文件的路径
    :param drug_disease_filename: 药物-疾病关联矩阵文件名
    :param protein_disease_filename: 蛋白质-疾病关联矩阵文件名
    :param dataset_type: 数据集类型（"luo", "yamanishi", "zheng"）
    :return: 药物-疾病关联矩阵，蛋白质-疾病关联矩阵
    """
    assert dataset_name in ["Luo","Es","GPCRs","ICs","NRs", "Zheng"]  
    drug_disease = np.loadtxt(network_path + drug_disease_filename)
    protein_disease = np.loadtxt(network_path + protein_disease_name)
    drug_disease_tensor = torch.Tensor(drug_disease)
    protein_disease_tensor = torch.Tensor(protein_disease)
    return drug_disease_tensor,protein_disease_tensor
                 
def load_dataset(network_path, drug_protein_filename, train_filename=None, test_filename=None, 
                 negative_ratio=1, dataset_type="luo"):
    """
    通用数据集加载函数
    :param network_path: 数据文件的路径
    :param drug_protein_filename: 药物-蛋白质关联矩阵文件名
    :param train_filename: 训练集文件名（可选，针对 zheng 数据集）
    :param test_filename: 测试集文件名（可选，针对 zheng 数据集）
    :param negative_ratio: 负样本与正样本的比例，默认为1
    :param dataset_type: 数据集类型（"luo", "yamanishi", "zheng"）
    :return: 节点数量列表，药物-蛋白质关联矩阵，转置矩阵，数据集
    """
    # 加载药物-蛋白质关联矩阵
    drug_protein = np.loadtxt(network_path + drug_protein_filename)
    num_drug = drug_protein.shape[0]
    num_protein = drug_protein.shape[1]
    
    # 根据数据集类型加载相应的正负样本
    if dataset_type == "luo" or dataset_type == "yamanishi":
        whole_positive_index, whole_negative_index = generate_negative_samples(drug_protein)
    elif dataset_type == "zheng":
        dti_train = np.loadtxt(network_path + train_filename)
        dti_test = np.loadtxt(network_path + test_filename)
        train_positive_index, _ = generate_negative_samples(dti_train)
        test_positive_index, _ = generate_negative_samples(dti_test)
        whole_positive_index = train_positive_index + test_positive_index
        
        # 合并所有负样本
        whole_negative_index = []
        for i in range(dti_train.shape[0]):
            for j in range(dti_train.shape[1]):
                if dti_train[i][j] == 0 and dti_test[i][j] == 0:
                    whole_negative_index.append([i, j])

    # 构建数据集
    #print('negative_ratio is', negative_ratio)
    data_set = construct_dataset(whole_positive_index, whole_negative_index, negative_ratio)

    # 将数据集写入文件
    with open(network_path+'imbanlanced_'+str(negative_ratio)+"_dti_index.txt", "w", encoding="utf-8") as f:
        for i in data_set:
            f.write(f"{i[0]}\t{i[1]}\t{i[2]}\n")

    # 创建 dtiedge.txt 文件
    with open(network_path+'imbanlanced'+str(negative_ratio)+"_dtiedge.txt", "w", encoding="utf-8") as f:
        for i in range(data_set.shape[0]):
            for j in range(i, data_set.shape[0]):
                if data_set[i][0] == data_set[j][0] or data_set[i][1] == data_set[j][1]:
                    f.write(f"{i}\t{j}\n")

    node_num = [num_drug, num_protein]
    drug_protein_tensor = torch.Tensor(drug_protein)
    protein_drug_tensor = drug_protein_tensor.t()

    return node_num, drug_protein_tensor, protein_drug_tensor, data_set


def load_dataset_from_name(dataName,negative_ratio):
    assert dataName in ["Luo","Es","GPCRs","ICs","NRs", "Zheng"]
    if dataName == "Luo":
        return  load_dataset( network_path='/data/zyf/HyperGCN-DTI/data/Luo/',
                            drug_protein_filename='mat_drug_protein.txt',
                            dataset_type="luo",
                            negative_ratio=negative_ratio)

    elif dataName == "Zheng":
        return load_dataset( network_path='/data/zyf/HyperGCN-DTI/data/Zheng/',
                            drug_protein_filename='mat_drug_target_1.txt',
                            train_filename='mat_drug_target_train.txt',
                            test_filename='mat_drug_target_test.txt',
                            dataset_type="zheng",
                            negative_ratio=negative_ratio )
    else:
        return load_dataset(network_path=f'/data/zyf/HyperGCN-DTI/data/Yamanishi/{dataName}/',
                            drug_protein_filename='d_p_i.txt',
                            dataset_type="yamanishi",
                            negative_ratio=negative_ratio)
    
def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sparse.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx
import torch

def row_normalization(tensor_mx):
    """
    对输入的tensor矩阵进行行归一化操作
    Args:
        matrix (torch.Tensor): 输入的二维 tensor 矩阵 (m x n)，由 0 和 1 组成。

    Returns:
        torch.Tensor: 行归一化后的矩阵
    """
    row_sum = tensor_mx.sum(dim=1, keepdim=True)
    row_sum[row_sum == 0] = 1.0
    normalized_matrix =tensor_mx / row_sum
    return normalized_matrix



def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def construct_fgraph(features, topk):
    # Cosine similarity
    dist = cos(features)
    inds = []
    for i in range(dist.shape[0]):
        ind = np.argpartition(dist[i, :], -(topk + 1))[-(topk + 1):]
        inds.append(ind)
    edge = []
    for i, v in enumerate(inds):
        for vv in v:
            if vv == i:
                pass
            else:
                edge.append([i, vv])
    return edge


def generate_knn(data):
    topk = 3

    edge = construct_fgraph(data, topk)
    res = []

    for line in edge:
        start, end = line[0], line[1]
        if int(start) < int(end):
            res.append([start, end])
    return res


def constructure_knngraph(dateset, h1, h2, aug=False):
    feature = torch.cat((h1[dateset[:, :1]], h2[dateset[:, 1:2]]), dim=2)

    feature = feature.squeeze(1)
    fedge = np.array(generate_knn(feature.cpu().detach().numpy()))
    fedge = load_graph(np.array(fedge), dateset.shape[0])
    edg = torch.Tensor.to_dense(fedge)
    edge = edg.numpy()


    return fedge, feature


def get_set(data, split=5):
    """
    :param data: dataset and label
    :return:
    testset index and trainset index
    """
    set1 = []
    set2 = []
    skf = StratifiedKFold(n_splits=split, shuffle=True)
    for train_index, test_index in skf.split(data[:, :2], data[:, 2:3]):
        set1.append(train_index)
        set2.append(test_index)
    return set1[0].reshape(-1), set2[0].reshape(-1)


def get_cross(data, split=5):
    """
    :param data: dataset and label
    :return:
    testset index and trainset index
    """
    set1 = []
    set2 = []
    skf = StratifiedKFold(n_splits=split, shuffle=True,random_state=config['seed'])
    for train_index, test_index in skf.split(data[:, :2], data[:, 2:3]):
        set1.append(train_index)
        set2.append(test_index)
    return set1, set2


def get_roc(out, label):
    return np.nan_to_num(roc_auc_score(label.cpu(), out[:, 1:].cpu().detach().numpy()))


def get_pr(out, label):
    predictions = out.argmax(dim=1).cpu().detach().numpy()

    precision, recall, thresholds = precision_recall_curve(label.cpu(), out[:, 1:].cpu().detach().numpy())
    aupr = auc3(recall, precision)
    precision = precision_score(label.cpu().numpy(), predictions)
    recall = recall_score(label.cpu().numpy(), predictions)
    return precision, recall,aupr


def get_f1score(out, label):
    return f1_score(label.cpu(), out.argmax(dim=1).cpu().detach().numpy())


def get_L2reg(parameters):
    reg = 0
    for param in parameters:
        reg += 0.5 * (param ** 2).sum()
    return reg


def load_graph(feature_edges, n):
    fedges = np.array(list(feature_edges), dtype=np.int32).reshape(feature_edges.shape)
    fadj = sparse.coo_matrix((np.ones(fedges.shape[0]), (fedges[:, 0], fedges[:, 1])), shape=(n, n),
                             dtype=np.float32)
    fadj = fadj + fadj.T.multiply(fadj.T > fadj) - fadj.multiply(fadj.T > fadj)
    nfadj = fadj + sparse.eye(fadj.shape[0])
    nfadj = sparse_mx_to_torch_sparse_tensor(nfadj)

    return nfadj



with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

def load_feature(dataset_name,feature_list,negative_ratio=1,normalize = False):
    config['device'] = "cuda" if torch.cuda.is_available() else "cpu"
    '''1:HyperDrug & HyperProtein features
     2:sequence feature from pre-trained LLM model
    3:HyperDrug-Disease & HyperProtein-Disease features
     input:list,contains types of features ''' 
    assert len(feature_list)>0 and 1 in  feature_list
    '为了后续图架构，保证使用HyperDrug和HyperProtein特征' 
    node_num, drug_protein, protein_drug, dtidata = load_dataset_from_name(dataset_name,negative_ratio)
    drug_protein_eye = torch.cat((drug_protein, torch.eye(node_num[0])), dim=1)
    protein_drug_eye = torch.cat((protein_drug, torch.eye(node_num[1])), dim=1)
    HyGraph_Drug = generate_G_from_H(drug_protein_eye).to(config['device'])
    HyGraph_protein = generate_G_from_H(protein_drug_eye).to(config['device'])
    #print('HyGraph_Drug,HyGraph_protein:',HyGraph_Drug.shape,HyGraph_protein.shape)
    
    if  feature_list==[1,2]:
        if dataset_name == "Luo" or dataset_name == "Zheng" or dataset_name=='BindingDB':
            hd = pd.read_csv(f'/data/zyf/HyperGCN-DTI/data/{dataset_name}/drug_smiles.csv')
            hp = pd.read_csv(f'/data/zyf/HyperGCN-DTI/data/{dataset_name}/protein_seq.csv')
        else:
            hd = pd.read_csv(f'/data/zyf/HyperGCN-DTI/data/Yamanishi/{dataset_name}/{dataset_name}_drug_smiles.csv')
            hp = pd.read_csv(f'/data/zyf/HyperGCN-DTI/data/Yamanishi/{dataset_name}/{dataset_name}_protein_seq.csv')
        features_d = torch.tensor(hd.iloc[:,2:].values,dtype=torch.float32).to(config['device'])
        features_p = torch.tensor(hp.iloc[:,2:].values,dtype=torch.float32).to(config['device'])
        print('load LLM features')

   
    elif feature_list== [1,3]:
        if dataset_name == "Luo" or dataset_name == "Zheng":
            drug_disease,protein_disease = load_disease_info(f'/data/zyf/HyperGCN-DTI/data/{dataset_name}/','mat_drug_disease.txt', 'mat_protein_disease.txt',dataset_name)
        else:
            drug_disease,protein_disease = load_disease_info(f'/data/zyf/HyperGCN-DTI/data/Yamanishi/{dataset_name}/',f'{dataset_name}_mat_drug_disease.txt', 'mat_protein_disease.txt',dataset_name)
        drug_disease_eye = torch.cat((drug_disease, torch.eye(node_num[0])), dim=1)
        protein_disease_eye = torch.cat((protein_disease, torch.eye(node_num[1])), dim=1)
        HyGraph_Drug_disease = row_normalization(generate_G_from_H(drug_disease_eye)).to(config['device'])
        HyGraph_protein_disease = row_normalization(generate_G_from_H(protein_disease_eye)).to(config['device'])
        #print('HyGraph_Drug_d,HyGraph_protein_d:',HyGraph_Drug_disease.shape,HyGraph_protein_disease.shape)
        #print(features_d.shape,HyGraph_Drug_disease.shape)
        features_d = torch.randn((node_num[0],384)).to(config['device'])
        features_p = torch.randn((node_num[1],320)).to(config['device'])
        HyGraph_Drug = torch.add(HyGraph_Drug,HyGraph_Drug_disease)
        HyGraph_protein =torch.add(HyGraph_protein,HyGraph_protein_disease)
        '''
        features_d  = torch.cat((features_d,HyGraph_Drug_disease), dim=1)
        features_p  = torch.cat((features_p, HyGraph_protein_disease), dim=1)
        '''
        #print(3,features_d.shape,features_p.shape)
        print('load HyperDrug-Disease & HyperProtein-Disease features and CONCAT feature! ')
    
    elif  feature_list  == [1,2,3]: 
        if dataset_name == "Luo" or dataset_name == "Zheng":
            hd = pd.read_csv(f'/data/zyf/HyperGCN-DTI/data/{dataset_name}/drug_smiles.csv')
            hp = pd.read_csv(f'/data/zyf/HyperGCN-DTI/data/{dataset_name}/protein_seq.csv')
            drug_disease,protein_disease = load_disease_info(f'/data/zyf/HyperGCN-DTI/data/{dataset_name}/','mat_drug_disease.txt', 'mat_protein_disease.txt',dataset_name)
            
        else:
            hd = pd.read_csv(f'/data/zyf/HyperGCN-DTI/data/Yamanishi/{dataset_name}/{dataset_name}_drug_smiles.csv')
            hp = pd.read_csv(f'/data/zyf/HyperGCN-DTI/data/Yamanishi/{dataset_name}/{dataset_name}_protein_seq.csv')
            drug_disease,protein_disease = load_disease_info(f'/data/zyf/HyperGCN-DTI/data/Yamanishi/{dataset_name}/',f'{dataset_name}_mat_drug_disease.txt', 'mat_protein_disease.txt',dataset_name)
        drug_disease_eye = torch.cat((drug_disease, torch.eye(node_num[0])), dim=1)
        protein_disease_eye = torch.cat((protein_disease, torch.eye(node_num[1])), dim=1)
        HyGraph_Drug_disease = generate_G_from_H(drug_disease_eye).to(config['device'])
        HyGraph_protein_disease = generate_G_from_H(protein_disease_eye).to(config['device'])
        features_d = torch.tensor(hd.iloc[:,2:].values,dtype=torch.float32).to(config['device'])
        features_p = torch.tensor(hp.iloc[:,2:].values,dtype=torch.float32).to(config['device'])
        HyGraph_Drug = torch.add(HyGraph_Drug,HyGraph_Drug_disease)
        HyGraph_protein =torch.add(HyGraph_protein,HyGraph_protein_disease)
        '''
        features_d  = torch.cat((features_d,HyGraph_Drug_disease), dim=1)
        features_p  = torch.cat((features_p, HyGraph_protein_disease), dim=1)
        '''
        print('load HyperDrug-Disease & HyperProtein-Disease features and CONCAT feature! &&&& LLM features! ')
        '''两种策略：1，把HyperDrug-Disease & HyperProtein-Disease 与HyGraph_Drug,HyGraph_protein融合作为特征输入；2，直接作为特征连接
    
        if 3 in feature_list:
            drug_disease,protein_disease = load_disease_info(f'/data/zyf/HyperGCN-DTI/data/{dataset_name}/','mat_drug_disease.txt', 'mat_protein_disease.txt',dataset_name)
            drug_disease_eye = torch.cat((drug_disease, torch.eye(node_num[0])), dim=1)
            protein_disease_eye = torch.cat((protein_disease, torch.eye(node_num[1])), dim=1)
            HyGraph_Drug_disease = generate_G_from_H(drug_disease_eye).to(config['device'])
            HyGraph_protein_disease = generate_G_from_H(protein_disease_eye).to(config['device'])
            #print('HyGraph_Drug_d,HyGraph_protein_d:',HyGraph_Drug_disease.shape,HyGraph_protein_disease.shape)
            if normalize:
                HyGraph_Drug_normalized = normalize_min_max(torch.add(HyGraph_Drug,HyGraph_Drug_disease))
                HyGraph_Protein_normalized = normalize_min_max(torch.add(HyGraph_protein,HyGraph_protein_disease))
                HyGraph_Drug, HyGraph_protein = HyGraph_Drug_normalized,HyGraph_Protein_normalized
            else:
                HyGraph_Drug_unnormalized = torch.add(HyGraph_Drug,HyGraph_Drug_disease)
                HyGraph_Protein_unnormalized =torch.add(HyGraph_protein,HyGraph_protein_disease)
                HyGraph_Drug,HyGraph_protein = HyGraph_Drug_unnormalized,HyGraph_Protein_unnormalized
            print('load HyperDrug-Disease & HyperProtein-Disease features and MERGE feature ! ')
        '''
    else:
        hd = torch.randn((node_num[0], 100))
        hp = torch.randn((node_num[1], 100))
        features_d = hd.to(config['device'])
        features_p = hp.to(config['device'])
        print('use random features')
    return node_num, drug_protein, protein_drug, dtidata,features_d,features_p,HyGraph_Drug,HyGraph_protein
