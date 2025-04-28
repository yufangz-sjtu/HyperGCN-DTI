import torch
import numpy as np
from scipy.spatial.distance import cdist


def Eu_dis(x):
    """
    :param x: N X D
                N: the object number
                D: Dimension of the feature
    :return: N X N distance matrix
    """
    dist_mat = cdist(x, x, metric='euclidean')
    return dist_mat



def feature_concat(normal_col=False, *F_list):
    """
    Concatenate multiple modality feature matrices. If a feature matrix has more than two dimensions,
    it will be reduced to two dimensions (using the last dimension as the feature dimension).
    
    :param F_list: Feature matrix list
    :param normal_col: normalize each column of the feature if set to True
    :return: Concatenated feature matrix
    """
    feature_list = []
    for f in F_list:
        if f is not None and f.size > 0:  # 检查是否为空矩阵
            # 如果特征矩阵维度大于2，进行重塑
            if len(f.shape) > 2:
                f = f.reshape(-1, f.shape[-1])
            # 对每列进行归一化（如果需要）
            if normal_col:
                f_max = np.max(np.abs(f), axis=0, keepdims=True)
                f_max[f_max == 0] = 1  # 防止除以0
                f = f / f_max
    
            # 将处理后的特征矩阵添加到列表中
            feature_list.append(f)
    
    # 如果所有输入都是空的，返回 None
    if len(feature_list) == 0:
        return None
   
    # 一次性将所有特征矩阵进行拼接
    features = np.concatenate(feature_list, axis=1)
    
    # 最终整体归一化（如果需要）
    if normal_col:
        features_max = np.max(np.abs(features), axis=0, keepdims=True)
        features_max[features_max == 0] = 1  # 防止除以0
        features = features / features_max
    
    return features



def hyperedge_concat(*H_list):
    """
    Concatenate hyperedge group in H_list
    :param H_list: Hyperedge groups which contain two or more hypergraph incidence matrix
    :return: Fused hypergraph incidence matrix
    """
    H = None
    for h in H_list:
        if h is not None and len(h) != 0:
            if H is None:
                H = h
            else:
                if not isinstance(h, list):
                    # 如果H和h都是单一矩阵，直接拼接
                    H = np.hstack([H, h])
                else:
                    # 如果H和h都是包含矩阵的列表，逐一拼接对应位置的矩阵
                    H = [np.hstack((a, b)) for a, b in zip(H, h)]
    
    return H

def generate_G_from_H(H, variable_weight=False):
    """
    calculate G from hypgraph incidence matrix H
    :param H: hypergraph incidence matrix H
    :param variable_weight: whether the weight of hyperedge is variable
    :return: G
    """
    if type(H) != list:
        return _generate_G_from_H(H, variable_weight)
    else:
        G = []
        for sub_H in H:
            G.append(generate_G_from_H(sub_H, variable_weight))
        return G



def _generate_G_from_H(H, variable_weight=False):
    """
    根据超图关联矩阵 H 计算矩阵 G
    :param H: 超图关联矩阵 H
    :param variable_weight: 是否使用可变权重
    :return: G 或 (DV2_H, W, invDE_HT_DV2) 取决于 variable_weight 参数
    """
    H = np.array(H)  # 确保 H 是 NumPy 数组
    n_edge = H.shape[1]
    
    # 超边的权重，默认设置为全1
    W = np.ones(n_edge)
    
    # 计算节点度和超边度
    DV = np.sum(H * W, axis=1)
    DE = np.sum(H, axis=0)

    # 计算 invDE 和 DV2 使用 NumPy 的向量化操作
    invDE = np.diag(np.where(DE == 0, 0, 1.0 / DE))
    DV2 = np.diag(np.where(DV == 0, 0, np.power(DV, -0.5)))

    # 将权重 W 转换为对角矩阵
    W = np.diag(W)

    # 使用 NumPy 数组进行矩阵乘法
    if variable_weight:
        # 如果使用可变权重，返回三个矩阵
        DV2_H = np.dot(DV2, H)
        invDE_HT_DV2 = np.dot(np.dot(invDE, H.T), DV2)
        return DV2_H, W, invDE_HT_DV2
    else:
        # 计算 G
        G = np.dot(np.dot(np.dot(np.dot(DV2, H), W), invDE), np.dot(H.T, DV2))
    
    # 转换为 PyTorch 张量
    G = torch.tensor(G, dtype=torch.float32)
    
    return G



def construct_H_with_KNN_from_distance(dis_mat, k_neig, is_probH=True, m_prob=1):
    """
    根据节点距离矩阵构建超图关联矩阵
    :param dis_mat: 节点距离矩阵
    :param k_neig: K 近邻的数量
    :param is_probH: 是否构建概率型关联矩阵
    :param m_prob: 概率参数
    :return: 超图关联矩阵 H
    """
    n_obj = dis_mat.shape[0]
    H = np.zeros((n_obj, n_obj))  # 初始化超图关联矩阵
    avg_dis = np.mean(dis_mat, axis=1)  # 预先计算每个节点到其他节点的平均距离
    
    for center_idx in range(n_obj):
        dis_vec = dis_mat[center_idx]
        nearest_idx = np.argsort(dis_vec)[:k_neig]  # 获取最近的 k 个节点
        
        # 确保当前节点在自己的邻居列表中
        if center_idx not in nearest_idx:
            nearest_idx[-1] = center_idx
        
        if is_probH:
            # 使用距离构造概率值
            H[nearest_idx, center_idx] = np.exp(-(dis_vec[nearest_idx] ** 2) / (m_prob * avg_dis[center_idx]) ** 2)
        else:
            # 二进制值
            H[nearest_idx, center_idx] = 1.0
            
    return H


def construct_H_with_KNN(X, K_neigs=[10], split_diff_scale=False, is_probH=True, m_prob=1):
    """
    从原始节点特征矩阵初始化多尺度超图关联矩阵
    :param X: 节点特征矩阵，形状为 N_object x feature_number
    :param K_neigs: 邻居数量的列表
    :param split_diff_scale: 是否将不同尺度的超边分开
    :param is_probH: 是否构建概率型关联矩阵
    :param m_prob: 概率参数
    :return: 超图关联矩阵 H
    """
    if len(X.shape) != 2:
        X = X.reshape(-1, X.shape[-1])
    
    if isinstance(K_neigs, int):
        K_neigs = [K_neigs]
    
    dis_mat = Eu_dis(X)  # 计算欧式距离矩阵
    H_list = []
    
    for k_neig in K_neigs:
        H_tmp = construct_H_with_KNN_from_distance(dis_mat, k_neig, is_probH, m_prob)
        if split_diff_scale:
            H_list.append(H_tmp)
        else:
            if len(H_list) == 0:
                H_list = H_tmp
            else:
                H_list = np.hstack((H_list, H_tmp))
    
    return H_list if split_diff_scale else H_list
