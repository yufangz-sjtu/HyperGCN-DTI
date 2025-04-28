import os
import torch
import numpy as np
from hypergraph_utils import _generate_G_from_H


def HyGraph_Matrix_DPP_Structure(dateset, num_drug, num_protein,dataset_name):
    def build_graph(data_col, num_elements, filename):
        Graph = np.zeros((dateset.shape[0], num_elements))
        lines = []
        
        for k in range(num_elements):
            indices = np.where(dateset[:, data_col] == k)[0]
            if len(indices) > 1:
                Graph[indices, k] = 1
                line = "\t".join(map(str, indices)) + "\n"
                lines.append(line)
        
        # 写入文件
        with open(filename, "w", encoding="utf-8") as f:
            f.writelines(lines)
        
        return Graph

    # 构建药物和蛋白质的关联矩阵并写入文件
    Graph_1 = build_graph(0, num_drug, f"/data/zyf/HyperGCN-DTI/graph_structure/{dataset_name}_HyGraph_Structure_DPP_drug.txt")
    Graph_2 = build_graph(1, num_protein, f"/data/zyf/HyperGCN-DTI/graph_structure/{dataset_name}_HyGraph_Structure_DPP_protein.txt")
    
    # 合并两个图关联矩阵
    HyGraph_Structure_DPP = np.concatenate([Graph_1, Graph_2], axis=1)
    
    # 删除所有列全为零的列
    HyGraph_Structure_DPP = HyGraph_Structure_DPP[:, HyGraph_Structure_DPP.sum(axis=0) != 0]
    
    # 使用之前定义的 generate_G_from_H 函数生成最终的超图矩阵
    HyGraph_Structure_DPP = _generate_G_from_H(HyGraph_Structure_DPP)
    
    return HyGraph_Structure_DPP

