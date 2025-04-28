# HyperGCN-DTI

Prediction of Drug-Target Interactions Based on Hypergraph Neural Networks with Multimodal Feature Fusion

---

## Introduction

HyperGCN-DTI is a novel framework that leverages hypergraph neural networks combined with multimodal feature fusion to enhance drug-target interaction (DTI) predictions. It integrates multimodal features, including embeddings from pretrained language models and biological networks, to construct hypergraphs for drugs, targets, and drug-target pairs. HyperGCN-DTI has been evaluated on multiple datasets, outperforming state-of-the-art DTI prediction models. Top-ranked predictions are validated through molecular docking, demonstrating significant improvements in DTI prediction accuracy. This framework holds promise for advancing drug repurposing, discovery, and target identification.

---

## Project Structure
HyperGCN-DTI/ ├── codes/ # 模型训练、测试及评估脚本 ├── data/ # 药物与靶点数据集 (本地存放，未上传) ├── Figures/ # 绘制的分析图表 ├── graph_structure/ # 超图结构相关的中间文件 ├── results/ # 实验输出结果 ├── savedmodel/ # 训练得到的模型权重文件 ├── requirements.txt # Python依赖包列表 ├── README.md # 项目说明文件 └── LICENSE # 项目许可证 (MIT License)

## Quick Start

1. Clone the repository:
   ```bash
   git clone https://github.com/yufangz-sjtu/HyperGCN-DTI.git
   cd HyperGCN-DTI
   ```

2. Prepare the datasets:

Due to the large size of the data, the data/ folder is not uploaded to GitHub.
Please contact the authors to obtain the data or refer to the Data Description section to manually download and place the files in the data/ folder.

3.Train and evaluate the model:

## Model training and evaluation

python codes/main.py

Data Description
The data/ directory should include the following:

Drug features
Target features
Interaction matrices
All input data must meet the preprocessing requirements specified in codes/preprocess.py.
⚠️ Note: Ensure that the file names and formats match the paths and requirements in the code.

Experimental Results
HyperGCN-DTI has been validated on multiple datasets, demonstrating its effectiveness.
Compared to standard GCN and KNN methods, HyperGCN-DTI achieves improvements in both AUC and AUPR metrics.

License
This project is open-sourced under the MIT License.