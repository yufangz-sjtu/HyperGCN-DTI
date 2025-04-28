#feature fusion

import requests
import time
import psutil
import os
import pandas as pd
from transformers import AutoTokenizer, AutoModel
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm
import torch
import numpy as np
import esm

def get_protein_sequence(uniprot_id):
    """
    获取给定 UniProt ID 的蛋白质序列。

    参数:
    uniprot_id (str): 蛋白质的 UniProt ID，例如 "P12345"。

    返回:
    str: 蛋白质序列，或 None 如果未找到。
    """
    url = f"https://www.uniprot.org/uniprot/{uniprot_id}.fasta"
    
    try:
        response = requests.get(url)
        if response.status_code == 200:
            fasta_data = response.text
            sequence = ''.join(fasta_data.splitlines()[1:])
            return sequence
        else:
            print(f"未能获取 {uniprot_id} 的序列: {response.status_code}")
            return None
    except requests.RequestException as e:
        print(f"请求出错: {e}")
        return None

def length_info(protein_name):
    
    # 打印前几行以查看结果
    print(protein_name.head())
    protein_name['protein_sequence'] = protein_name['protein_sequence'].fillna('').astype(str)
    protein_name['sequence_length'] = protein_name['protein_sequence'].apply(len)
    # 统计基本信息：平均长度、最大长度、最小长度
    average_length = protein_name['sequence_length'].mean()
    max_length = protein_name['sequence_length'].max()
    min_length = protein_name['sequence_length'].min()
    zero_length_count = (protein_name['sequence_length'] == 0).sum()

    print(f"Number of sequences with length 0: {zero_length_count}")
    if zero_length_count != 0:
        # 找到序列长度为 0 的行
        zero_length_rows = protein_name[protein_name['sequence_length'] == 0]
        # 获取这些行的索引
        zero_length_indices = zero_length_rows.index.tolist()
        print(f"Indices of sequences with length 0: {zero_length_indices}")
    print(f"Average sequence length: {average_length}")
    print(f"Maximum sequence length: {max_length}")
    print(f"Minimum sequence length: {min_length}")

class ProteinFeatureExtractor:
    def __init__(self, model_name='esm2_t33_650M_UR50D'):
        # 加载 ESM 模型和字母表
        self.model, self.alphabet = esm.pretrained.__dict__[model_name]()
        self.batch_converter = self.alphabet.get_batch_converter()
        
        self.device = torch.device('cpu' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()
        
        # 获取模型的层数
        self.num_layers = self.model.num_layers
        print(f"Loaded model with {self.num_layers} layers.")

    def get_protein_features(self, sequence):
        data = [("protein", sequence)]
        batch_labels, batch_strs, batch_tokens = self.batch_converter(data)
        batch_lens = (batch_tokens != self.alphabet.padding_idx).sum(1)
        batch_tokens = batch_tokens.to(self.device)
        
        # 提取特征
        with torch.no_grad():
            # 使用模型的最后一层编号
            results = self.model(batch_tokens, repr_layers=[self.num_layers], return_contacts=True)
        #print(results["representations"])
        token_representations = results["representations"][self.num_layers]
        
        # 去掉起始和终止标记，并计算序列的平均特征
        sequence_embedding = token_representations[0, 1:len(sequence)+1].mean(0).cpu().numpy()
        return sequence_embedding

    def extract_features_from_csv(self, csv_path):
        # 读取包含蛋白质序列的 CSV 文件
        try:
            df = pd.read_csv(csv_path)
            # 检查是否包含 'protein_sequence' 列
            if 'protein_sequence' not in df.columns:
                raise ValueError("Input CSV must contain a 'protein_sequence' column.")
        except (pd.errors.ParserError, ValueError) as e:
            df = pd.read_csv(csv_path, sep='\t')

  
        features_list = []
        for sequence in tqdm(df['protein_sequence'], desc="Processing sequences"):
            #print(sequence)
            features = self.get_protein_features(sequence)
            features_list.append(features)

    
        features_array = np.array(features_list)
        feature_columns = [f'feature_{i+1}' for i in range(features_array.shape[1])]
        features_df = pd.DataFrame(features_array, columns=feature_columns)
        df_with_features = pd.concat([df, features_df], axis=1)
        df_with_features.to_csv(csv_path, index=False)
        print(f"Features successfully extracted and saved to: {csv_path}")
         
class SMILESFeatureExtractor:
    def __init__(self, model_name='DeepChem/ChemBERTa-77M-MTR'):
        # 使用 load_plm 方法加载预训练模型和 tokenizer
        self.tokenizer, self.model = self.load_plm(model_name)
        
        # 设置设备为 GPU 或 CPU
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()

    @staticmethod
    def load_plm(model_name='DeepChem/ChemBERTa-77M-MTR'):
        path = '/data/zyf/HyperGCN-DTI/codes/ChemBert'
        tokenizer = AutoTokenizer.from_pretrained(path)
        model = AutoModel.from_pretrained(path)
        return tokenizer, model

    def extract_drug_features(self, smiles):
        try:
            # 使用 tokenizer 将 SMILES 编码为模型输入
            inputs = self.tokenizer(smiles, return_tensors='pt', padding=True, truncation=True)
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            
            # 使用模型进行前向传播，提取特征
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            # 获取池化后的输出（句子级别特征）
            embeddings = outputs.pooler_output.squeeze().cpu().numpy()
            return embeddings
        except Exception as e:
            print(f"Error extracting features for SMILES: {smiles}. Error: {e}")
            return None

    def process_and_save_features(self, csv_path):
        try:
            df = pd.read_csv(csv_path)
            #print( df.columns)
            # 检查是否包含 'protein_sequence' 列
            if 'SMILES' not in df.columns:
                raise ValueError("Input CSV must contain a 'smiles' column.")
        except (pd.errors.ParserError, ValueError) as e:
            df = pd.read_csv(csv_path, sep='\t')

        # 提取 SMILES 列的特征，使用 tqdm 查看进度
        features_list = []
        for smiles in tqdm(df['SMILES'], desc="Processing SMILES"):
            features = self.extract_drug_features(smiles)
            if features is not None:
                features_list.append(features)
            else:
                # 如果提取失败，填充 NaN
                features_list.append(np.nan)

        # 将特征列表转换为 DataFrame
        feature_columns = [f'feature_{i+1}' for i in range(len(features_list[0]))] if features_list[0] is not np.nan else []
        features_df = pd.DataFrame(features_list, columns=feature_columns)

        # 合并原始数据和特征
        df_with_features = pd.concat([df, features_df], axis=1)

        # 保存到 CSV 文件
        df_with_features.to_csv(csv_path, index=False)
        print(f"Features successfully extracted and saved to: {csv_path}")



class KEGGDrugScraper:
    def __init__(self, input_csv_path='/data/zyf/HyperGCN-DTI/data/Yamanishi/drug_smiles.csv',
                        output_csv_path='/data/zyf/HyperGCN-DTI/data/Yamanishi/drug_disease.csv'):
        """
        初始化 KEGGDrugScraper 类

        Parameters:
        - kegg_ids: list
            KEGG Drug ID 列表
        - output_csv_path: str, optional
            保存 CSV 文件的路径，默认路径为 '/mnt/data/kegg_drug_info.csv'
        """
        self.drug_name = pd.read_csv(input_csv_path)
        self.kegg_ids =self.drug_name['KEGG Drug ID'].dropna().unique().tolist()
        self.output_csv_path = output_csv_path
        self.results = []
    
    def fetch_kegg_info(self, kegg_id):
        """
        根据 KEGG Drug ID 获取 Efficacy 和 Disease 信息，并将它们合并
        """
        url = f"https://www.genome.jp/dbget-bin/www_bget?dr:{kegg_id}"
        response = requests.get(url)
        
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # 查找 Efficacy 信息
            efficacy_section = soup.find(string="Efficacy")
            if efficacy_section:
                efficacy = efficacy_section.find_next('td').get_text(strip=True, separator='; ')
            else:
                efficacy = 'N/A'
            
            # 查找 Disease 信息
            disease_section = soup.find(string="Disease")
            if disease_section:
                disease_entries = disease_section.find_next('td').find_all('a')
                disease_list = [entry.get_text(strip=True) for entry in disease_entries]
                disease = '; '.join(disease_list)
            else:
                disease = 'N/A'
            
            # 将 Efficacy 和 Disease 合并
            efficacy_disease = f"{efficacy}; {disease}" if efficacy != 'N/A' and disease != 'N/A' else efficacy if disease == 'N/A' else disease
            
            return efficacy_disease
        else:
            return 'N/A'
    
    def scrape_data(self):
        """
        遍历 KEGG Drug ID 列表并爬取信息
        """
        for kegg_id in tqdm(self.kegg_ids):
            #print(f"Fetching data for KEGG Drug ID: {kegg_id}")
            efficacy_disease = self.fetch_kegg_info(kegg_id)
            #print(f"Efficacy and Disease: {efficacy_disease}")
            
            self.results.append({'KEGG Drug ID': kegg_id, 'Efficacy and Disease': efficacy_disease})
            
            # 等待一会儿以避免对服务器造成压力
            time.sleep(1)
    
    def save_to_csv(self):
        """
        将爬取结果保存到 CSV 文件
        """
        df = pd.DataFrame(self.results)
        df.to_csv(self.output_csv_path, index=False)
        print(f"Data has been saved to {self.output_csv_path}")
    
    def run(self):
        """
        执行完整的爬取和保存流程
        """
        self.scrape_data()
        self.save_to_csv()


def get_smiles(drug_name,mapping_data):
    drug_smiles = drug_name.merge(mapping_data, on='DrugBank ID', how='left')
    # Check for any missing values in the 'SMILES' column
    missing_values = drug_smiles['SMILES'].isna().sum()
    # Display the cleaned data and the count of missing values
    print(f"Number of missing values in 'SMILES' column: {missing_values}")
    #print(drug_smiles)
    return drug_smiles

def convert_xlsx_to_csv(excel_file_path,csv_file_path,qtype):
    df = pd.read_excel(excel_file_path, header=None)
    df.columns = ['Index',qtype]
    df.to_csv(csv_file_path, index=False, encoding='utf-8')

def merge_data_by(onwhat,name,qtype):
    data1 =  pd.read_csv(f'/data/zyf/HyperGCN-DTI/data/Yamanishi/{name}/{name}_{qtype}.csv')
    print(data1.columns)
    data1 = data1.drop('Index', axis=1)
    data2 = pd.read_csv(f'/data/zyf/HyperGCN-DTI/data/Yamanishi/{qtype}.csv')
    merged_data = data1.merge(data2,on=onwhat,how='left')
    merged_data.to_csv(f'/data/zyf/HyperGCN-DTI/data/Yamanishi/{name}/{name}_{qtype}.csv')

def generate_nteraction_matrix(name):
    # 生成药物与疾病的相互作用矩阵
    drug_disease_df = pd.read_csv(f'/data/zyf/HyperGCN-DTI/data/Yamanishi/{name}/{name}_drug_disease.csv')
    interaction_matrix = pd.crosstab(drug_disease_df['KEGG Drug ID'], drug_disease_df['Efficacy and Disease'])
    #interaction_matrix = interaction_matrix.drop(columns=['KEGG Drug ID']) 
    # 将矩阵保存到新的CSV文件中
    interaction_matrix_file = f'/data/zyf/HyperGCN-DTI/data/Yamanishi/{name}/{name}_mat_drug_disease.txt'
    interaction_matrix.to_csv(interaction_matrix_file, sep='\t',index=False,header=False)

    # 返回生成的文件路径
    interaction_matrix_file

if __name__ == "__main__":
    # 加载数据
    luo_protein_name = pd.read_csv('/data/zyf/HyperGCN-DTI/data/Luo/protein.txt', header=None, names=['UniProt ID'])
    Zheng_protein_name = pd.read_csv('/data/zyf/HyperGCN-DTI/data/Zheng/target_dic.txt', header=None, names=['UniProt ID'])

    tqdm.pandas(desc="Downloading Luo protein sequences")
    luo_protein_name['protein_sequence'] = luo_protein_name['UniProt ID'].progress_apply(get_protein_sequence)

    tqdm.pandas(desc="Downloading Zheng protein sequences")
    Zheng_protein_name['protein_sequence'] = Zheng_protein_name['UniProt ID'].progress_apply(get_protein_sequence)

    # 保存结果
    luo_protein_name.to_csv('/data/zyf/HyperGCN-DTI/data/Luo/protein_seq.csv', index=False)
    Zheng_protein_name.to_csv('/data/zyf/HyperGCN-DTI/data/Zheng/protein_seq.csv', index=False)
    
    #统计蛋白质序列信息
    luo_protein_name['sequence_length'] = length_info(luo_protein_name)
    Zheng_protein_name['sequence_length'] = length_info(Zheng_protein_name)

    #提取蛋白质LLM特征
    os.chdir('/data/zyf/HyperGCN-DTI/data')
    for csv_path in ['./Luo/protein_seq.csv','./Yamanishi/protein_seq.csv','./Zheng/protein_seq.csv']:
        extractor = ProteinFeatureExtractor(model_name='esm2_t6_8M_UR50D')
        # 从 CSV 文件中提取特征并保存
        extractor.extract_features_from_csv(csv_path)
    
    for csv_path in[ './Luo/drug_smiles.csv' ,
                    './Yamanishi/drug_smiles.csv',
                    './Zheng/drug_smiles.csv']:
        extractor = SMILESFeatureExtractor(model_name='DeepChem/ChemBERTa-77M-MTR')
        extractor.process_and_save_features(csv_path)


    #获取drug-disease信息
    drug_disease_scraper = KEGGDrugScraper()
    drug_disease_scraper.run()


    for name in ['Es','GPCRs','ICs','NRs']:
        convert_xlsx_to_csv(f'/data/zyf/HyperGCN-DTI/data/Yamanishi/{name}/{name}_drug.xlsx',f'/data/zyf/HyperGCN-DTI/data/Yamanishi/{name}/{name}_drug_smiles.csv','KEGG Drug ID')
        convert_xlsx_to_csv(f'/data/zyf/HyperGCN-DTI/data/Yamanishi/{name}/{name}_protein.xlsx',f'/data/zyf/HyperGCN-DTI/data/Yamanishi/{name}/{name}_protein_seq.csv','pro_id')
        merge_data_by('KEGG Drug ID',name,'drug_smiles')
        merge_data_by('pro_id',name,'protein_seq')
        generate_nteraction_matrix(name)