# HyperGCN-DTI

Prediction of Drug-Target Interactions Based on Hypergraph Neural Networks with Multimodal Feature Fusion

---

## Introduction

HyperGCN-DTI is a novel framework that leverages hypergraph neural networks combined with multimodal feature fusion to enhance drug-target interaction (DTI) predictions. It integrates multimodal features, including embeddings from pretrained language models and biological networks, to construct hypergraphs for drugs, targets, and drug-target pairs. HyperGCN-DTI has been evaluated on multiple datasets, outperforming state-of-the-art DTI prediction models. Top-ranked predictions are validated through molecular docking, demonstrating significant improvements in DTI prediction accuracy. This framework holds promise for advancing drug repurposing, discovery, and target identification.

---

## Project Structure

```
HyperGCN-DTI/
├── codes/           # Scripts for model training, testing, and evaluation
├── data/            # Drug and target datasets (not uploaded due to size)
├── Figures/         # Analysis figures
├── graph_structure/ # Intermediate files related to hypergraph structures
├── results/         # Experimental output results
├── savedmodel/      # Trained model weight files
├── requirements.txt # Python dependency list
├── README.md        # Project description file
└── LICENSE          # Project license (MIT License)
```

## Quick Start

1. Clone the repository:
   ```bash
   conda create -n my_env
   pip install -r requirements.txt
   git clone https://github.com/yufangz-sjtu/HyperGCN-DTI.git
   cd HyperGCN-DTI
   ```

2. Prepare the datasets:
   - Due to the large size of the data, the `data/BindingDB` folder is not uploaded to GitHub.
   - Please contact the authors to obtain the data or refer to the **Data Description** section to manually download and place the files in the `data/` folder.

3. Train and evaluate the model:
   ```bash
   python codes/main.py
   ```

---

## Data Description

The `data/` directory should include the following files:

### Drug-related files:
- **`drug.txt`**: List of drugs.
- **`drug_smiles.csv`**: SMILES representations of drugs.
- **`drug_dict_map.txt`**: Mapping of drug IDs to indices.
- **`mat_drug_drug.txt`**: Drug-drug similarity matrix.
- **`mat_drug_disease.txt`**: Drug-disease association matrix.
- **`mat_drug_se.txt`**: Drug-side effect association matrix.

### Protein-related files:
- **`protein.txt`**: List of proteins.
- **`protein_seq.csv`**: Protein sequences.
- **`protein_seq_with_features.csv`**: Protein sequences with extracted features.
- **`protein_dict_map.txt`**: Mapping of protein IDs to indices.
- **`mat_protein_protein.txt`**: Protein-protein similarity matrix.
- **`mat_protein_disease.txt`**: Protein-disease association matrix.

### Drug-Target Interaction (DTI) files:
- **`dti_index.txt`**: Drug-target interaction indices.
- **`dtiedge.txt`**: Drug-target interaction edges.
- **`imbalanced_X_dti_index.txt`**: Imbalanced DTI indices for different configurations (e.g., `X=1, 2, 3, 4, 5`).
- **`imbalancedX_dtiedge.txt`**: Imbalanced DTI edges for different configurations.

### Similarity matrices:
- **`Similarity_Matrix_Drugs.txt`**: Drug similarity matrix.
- **`Similarity_Matrix_Proteins.txt`**: Protein similarity matrix.

### Other files:
- **`disease.txt`**: List of diseases.
- **`se.txt`**: List of side effects.

⚠️ **Note**: Ensure that all files are placed in the `data/` directory and meet the preprocessing requirements specified in `codes/preprocess.py`.

---

## Experimental Results

HyperGCN-DTI has been validated on multiple datasets, demonstrating its effectiveness.  
Compared to standard GCN and KNN methods, HyperGCN-DTI achieves significant improvements in both AUC and AUPR metrics.

---

## License

This project is open-sourced under the MIT License. See the [LICENSE](LICENSE) file for details.