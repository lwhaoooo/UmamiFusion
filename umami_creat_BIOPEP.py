import torch
import esm
from torch_geometric.data import Data
from tqdm import tqdm
import pandas as pd
import numpy as np
import os
import json, pickle
from collections import OrderedDict
from rdkit import Chem
from rdkit.Chem import MolFromSmiles
import networkx as nx
from utils import TestbedDataset  # 假设 TestbedDataset 在 utils 中定义
import pdb

# -------------------- 配置 --------------------
FIXED_SEQ_LENGTH = 39          # 固定序列长度
CSV_PATH = '/root/autodl-tmp/new_protein_graph/data/umami/BIOPEP_external_set/BIOPEP_external_test_set.csv'
OUTPUT_DATASET_NAME = 'biopep_test'   # 保存的 .pt 文件名（会在 data/processed/ 下生成 biopep_test.pt）
ROOT_DIR = 'data'               # 根目录，与原始代码一致

# 氨基酸词汇表
seq_voc = "ACDEFGHIKLMNPQRSTVWY"
seq_dict = {v: (i + 1) for i, v in enumerate(seq_voc)}
# ---------------------------------------------

# 1. 加载 ESM-2 模型
model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
batch_converter = alphabet.get_batch_converter()
model.eval()

# 2. 定义函数：将 FASTA 序列转换为图数据（固定长度）
def fasta_to_graph(fasta_sequence, fixed_length=FIXED_SEQ_LENGTH):
    """
    与原始代码完全一致，返回固定长度的节点特征、边、mask等。
    """
    original_length = len(fasta_sequence)
    if original_length > fixed_length:
        fasta_sequence = fasta_sequence[:fixed_length]
        original_length = fixed_length

    batch_labels, batch_strs, batch_tokens = batch_converter([(fasta_sequence, fasta_sequence)])
    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[6], return_contacts=True)

    token_representations = results["representations"][6][0]
    contacts = results["contacts"][0]

    # 去掉 <cls> 和 <eos>
    token_representations = token_representations[1:-1]
    feature_dim = token_representations.shape[1]

    # 填充到固定长度
    if original_length < fixed_length:
        padding_length = fixed_length - original_length
        padding_features = torch.zeros(padding_length, feature_dim)
        node_features_esm = torch.cat([token_representations, padding_features], dim=0)
    else:
        node_features_esm = token_representations

    # mask
    mask = torch.zeros(fixed_length)
    mask[:original_length] = 1

    # 构建接触图（仅真实节点）
    contact_binary = torch.zeros(fixed_length, fixed_length)
    contact_binary[:original_length, :original_length] = contacts[:original_length, :original_length].clone()
    # 添加相邻边
    for i in range(original_length - 1):
        contact_binary[i, i + 1] = 1
        contact_binary[i + 1, i] = 1

    edge_list = torch.nonzero(contact_binary > 0.5)
    edge_weights = contact_binary[edge_list[:, 0], edge_list[:, 1]]
    edge_index = edge_list.t().contiguous()
    edge_attr = edge_weights.view(-1, 1).float()

    return fixed_length, node_features_esm, edge_index, edge_attr, mask

# 3. 序列整数编码函数
def seq_cat(prot):
    x = np.zeros(FIXED_SEQ_LENGTH)
    for i, ch in enumerate(prot[:FIXED_SEQ_LENGTH]):
        x[i] = seq_dict.get(ch, 0)
    return x

# 4. 读取外部测试集 CSV
df = pd.read_csv(CSV_PATH)
peps = list(df['SEQUENCE'])
Y = list(df['TASTE'])

print(f"读取外部测试集：共 {len(peps)} 条肽序列")

# 5. 为所有序列生成图数据（只处理一次，供 TestbedDataset 使用）
print("正在为外部测试集生成固定长度的图数据...")
fasta_graph = {}
for seq in tqdm(peps, desc="生成图数据"):
    c_size, features, edge_index, edge_attr, mask = fasta_to_graph(seq, fixed_length=FIXED_SEQ_LENGTH)
    if c_size is not None:
        fasta_graph[seq] = (c_size, features, edge_index, edge_attr, mask)

print(f"成功生成 {len(fasta_graph)} 个图数据，每个图节点数固定为 {FIXED_SEQ_LENGTH}")

# 6. 生成序列的整数编码特征（作为辅助输入）
XT = [seq_cat(t) for t in peps]
embeding = np.asarray(XT)
Y = np.asarray(Y)

# 7. 使用 TestbedDataset 创建并保存 .pt 文件
processed_file = os.path.join(ROOT_DIR, 'processed', f'{OUTPUT_DATASET_NAME}.pt')
if not os.path.isfile(processed_file):
    print(f"正在创建 {processed_file} ...")
    data = TestbedDataset(
        root=ROOT_DIR,
        dataset=OUTPUT_DATASET_NAME,
        xd=peps,
        xt=embeding,
        y=Y,
        fasta_graph=fasta_graph
    )
    print(f"外部测试集数据已保存至 {processed_file}")
else:
    print(f"{processed_file} 已存在，跳过创建。")
