# # import torch
# # import esm
# # import pandas as pd
# # import numpy as np
# # import os
# # import pickle
# # from torch_geometric.data import Data
# # from tqdm import tqdm
# # import pdb
# #
# # # # 加载 PSSM 特征字典（字典的键是FASTA序列，值是PSSM特征向量）
# # # pssm_feature_dict = torch.load('/root/autodl-tmp/new_protein_graph/data/GPCR_feature_pssm/test/test_feature_dict_tensor.pt')
# #
# # # pdb.set_trace()
# #
# # # 读取数据路径
# # file_path_test = "/root/autodl-tmp/new_protein_graph/data/umami/ump442_train.csv"
# #
# # # 读取数据
# # data = pd.read_csv(file_path_test)
# #
# # pdb.set_trace()
# #
# # # # 蛋白质序列去重
# # # data_cleaned_protein = data.drop_duplicates(subset=['SEQUENCE'])
# #
# # # 转化为列表
# # compound_protein = list(data['SEQUENCE'])
# #
# # 将列表转换为字典，键和值都为列表中的值
# compound_protein_dict = {idx: value for idx, value in enumerate(compound_protein)}
#
# # 加载 ESM-2 预训练模型和字母表
# model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
#
# # 从 alphabet 对象中获取一个批量转换器（batch converter）
# batch_converter = alphabet.get_batch_converter()
#
# # 确保模型处于评估模式
# model.eval()
#
# # 存储所有图数据的字典
# graphs_dict = {}
#
# # One-Hot编码函数
# # def one_hot_encoding(sequence):
# #     amino_acids = "ACDEFGHIKLMNPQRSTVWY"
# #     amino_acid_dict = {aa: i for i, aa in enumerate(amino_acids)}
# #     one_hot_matrix = []
# #     for aa in sequence:
# #         one_hot_vector = np.zeros(len(amino_acids))
# #         if aa in amino_acid_dict:
# #             one_hot_vector[amino_acid_dict[aa]] = 1
# #         one_hot_matrix.append(one_hot_vector)
# #
# #     # 先转换为 numpy 数组，再转换为 tensor
# #     one_hot_matrix = np.array(one_hot_matrix)
# #     return torch.tensor(one_hot_matrix, dtype=torch.float)
#
# # 处理每个序列，生成图数据
# for seq_id, (seq_name, seq) in tqdm(enumerate(compound_protein_dict.items()), total=len(compound_protein_dict), desc="Processing sequences"):
#     # 用batch_converter处理当前的蛋白质序列
#     batch_labels, batch_strs, batch_tokens = batch_converter([(seq_name, seq)])
#
#     # 通过模型获取当前序列的表示和接触图
#     with torch.no_grad():
#         results = model(batch_tokens, repr_layers=[6], return_contacts=True)
#
#     # pdb.set_trace()
#     # 获取 token 表示和接触图
#     token_representations = results["representations"][6][0]  # 当前序列的token表示
#     contacts = results["contacts"][0]  # 当前序列的接触图
#
#     # 如果不需要<cls>和<eos>，可以进行截取
#     token_representations = token_representations[1:-1]  # 去掉第一个和最后一个 token（即 <cls> 和 <eos>）
#
#     num_tokens = len(seq)
#
#     # 1. 获取ESM特征向量，形状为 (num_tokens, 1280)
#     node_features_esm = token_representations  # ESM特征向量
#
#     # pdb.set_trace()
#
#     # # 2. 获取PSSM特征向量（从PSSM特征字典中获取）
#     # if seq in pssm_feature_dict:  # 确保字典中有对应的PSSM特征
#     #     pssm_features = pssm_feature_dict[seq]  # 获取PSSM特征向量
#     # else:
#     #     print(f"PSSM features for sequence {seq_name} not found.")
#     #     continue  # 如果PSSM特征不存在，可以跳过此序列
#
#     # # 3. 获取One-Hot编码
#     # one_hot_features = one_hot_encoding(seq)  # 生成one-hot编码特征
#
#     # # 4. 拼接特征：将 ESM、PSSM 和 One-Hot 编码特征拼接
#     # combined_node_features = torch.cat([one_hot_features, pssm_features, node_features_esm], dim=-1)
#
#     # # 4. 拼接特征：将 ESM、PSSM 和 One-Hot 编码特征拼接
#     # combined_node_features = torch.cat([one_hot_features, node_features_esm], dim=-1)
#
#     # 5. 获取接触图的边
#     edge_list = torch.nonzero(contacts > 0.2)  # 获取接触概率 > 0 的位置
#     edge_weights = contacts[edge_list[:, 0], edge_list[:, 1]]  # 获取每个边的接触概率值作为边的权重
#
#     # 6. 手动添加相邻氨基酸之间的边
#     contact_binary = contacts.clone()  # 克隆原始接触图
#     manual_edges = []
#     for i in range(num_tokens - 1):  # 从第二个氨基酸到倒数第二个
#         manual_edges.append([i, i + 1])  # 为相邻的氨基酸添加边
#
#     # 更新接触图：对于每个相邻氨基酸，直接设置接触图值为 1
#     for manual_edge in manual_edges:
#         i, j = manual_edge
#         contact_binary[i, j] = 1  # 设置接触图中 (i, j) 的值为 1
#         contact_binary[j, i] = 1  # 无向图，确保反向的边也被设置
#
#     # 7. 获取更新后的接触图中所有非零位置作为边
#     edge_list = torch.nonzero(contact_binary > 0.2)  # 获取更新后的接触图中的所有边
#     edge_weights = contact_binary[edge_list[:, 0], edge_list[:, 1]]  # 获取每个边的接触概率值作为边的权重
#
#     # 8. 构建图数据
#     edge_index = edge_list.t().contiguous()  # 转置并确保是连续的张量
#     edge_attr = edge_weights.view(-1, 1).float()  # 确保边的权重为浮点数，并调整形状
#
#     # # 为当前蛋白质序列构建图数据对象
#     # graph = Data(x=combined_node_features, edge_index=edge_index, edge_attr=edge_attr)
#
#     # 为当前蛋白质序列构建图数据对象
#     graph = Data(x=node_features_esm, edge_index=edge_index, edge_attr=edge_attr)
#
#     # pdb.set_trace()
#
#     # 将图数据保存到字典中
#     graphs_dict[seq] = graph
#
# # 保存图数据字典到本地
# output_path = "/root/autodl-tmp/new_protein_graph/data/umami/graph_data/train_esm.pkl"
# with open(output_path, 'wb') as f:
#     pickle.dump(graphs_dict, f)
#
# print(f"Graphs dictionary saved to {output_path}")


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
from utils import *
import pdb

# 加载 ESM-2 预训练模型和字母表
model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()

# 从 alphabet 对象中获取一个批量转换器（batch converter）
batch_converter = alphabet.get_batch_converter()

# 确保模型处于评估模式
model.eval()

# 定义函数: 将FASTA序列转换为图数据
def fasta_to_graph(fasta_sequence):
    """
    根据FASTA序列生成图数据，包括节点特征、边及边权重。

    参数:
    fasta_sequence (str): 蛋白质的FASTA序列。

    返回:
    tuple:
        - num_tokens (int): 序列的token数（即节点数）。
        - node_features_esm (tensor): 由ESM-2生成的节点特征（特征维度为1280）。
        - edge_index (tensor): 图的边的索引。
        - edge_attr (tensor): 边的权重（接触概率）。
    """
    # pdb.set_trace()
    # 使用batch_converter处理当前的蛋白质序列
    batch_labels, batch_strs, batch_tokens = batch_converter([(fasta_sequence, fasta_sequence)])

    # 通过模型获取当前序列的表示和接触图
    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[6], return_contacts=True)

    # 获取 token 表示和接触图
    token_representations = results["representations"][6][0]  # 当前序列的token表示
    contacts = results["contacts"][0]  # 当前序列的接触图

    # 如果不需要<cls>和<eos>，可以进行截取
    token_representations = token_representations[1:-1]  # 去掉第一个和最后一个 token（即 <cls> 和 <eos>）

    num_tokens = len(fasta_sequence)

    # 获取ESM特征向量，形状为 (num_tokens, 1280)
    node_features_esm = token_representations  # ESM特征向量

    # 获取接触图的边
    edge_list = torch.nonzero(contacts > 0.5)  # 获取接触概率 > 0 的位置
    edge_weights = contacts[edge_list[:, 0], edge_list[:, 1]]  # 获取每个边的接触概率值作为边的权重

    # 手动添加相邻氨基酸之间的边
    contact_binary = contacts.clone()  # 克隆原始接触图
    manual_edges = []
    for i in range(num_tokens - 1):  # 从第二个氨基酸到倒数第二个
        manual_edges.append([i, i + 1])  # 为相邻的氨基酸添加边

    # 更新接触图：对于每个相邻氨基酸，直接设置接触图值为 1
    for manual_edge in manual_edges:
        i, j = manual_edge
        contact_binary[i, j] = 1  # 设置接触图中 (i, j) 的值为 1
        contact_binary[j, i] = 1  # 无向图，确保反向的边也被设置

    # pdb.set_trace()
    # 获取更新后的接触图中所有非零位置作为边
    edge_list = torch.nonzero(contact_binary > 0.5)  # 获取更新后的接触图中的所有边
    edge_weights = contact_binary[edge_list[:, 0], edge_list[:, 1]]  # 获取每个边的接触概率值作为边的权重

    # 构建图数据
    edge_index = edge_list.t().contiguous()  # 转置并确保是连续的张量
    edge_attr = edge_weights.view(-1, 1).float()  # 确保边的权重为浮点数，并调整形状

    # 返回节点数、特征和边
    return num_tokens, node_features_esm, edge_index, edge_attr

def seq_cat(prot):
    x = np.zeros(max_seq_len)
    for i, ch in enumerate(prot[:max_seq_len]):
        x[i] = seq_dict.get(ch, 0)  # 使用字典的get方法，找不到字符则返回0
    return x

# 处理新数据集
train_dataset_path = '/root/autodl-tmp/new_protein_graph/data/umami/ump442_train.csv'
test_dataset_path = '/root/autodl-tmp/new_protein_graph/data/umami/ump442_test.csv'

# 定义序列字典和最大序列长度
seq_voc = "ABCDEFGHIKLMNOPQRSTUVWXYZ"
seq_dict = {v: (i + 1) for i, v in enumerate(seq_voc)}
seq_dict_len = len(seq_dict)
max_seq_len = 1000

# 生成SMILES图
peps_iso_fasta = []
for dt_name in ['ump789']:
    for opt in ['train', 'test']:
        df = pd.read_csv(f'data/umami/{dt_name}_{opt}.csv')
        # pdb.set_trace()
        peps_iso_fasta += list(df['SEQUENCE'])

peps_iso_fasta = set(peps_iso_fasta)
fasta_graph = {}
for fasta in peps_iso_fasta:
    c_size, features, edge_index, edge_attr = fasta_to_graph(fasta)
    if c_size is not None:
        fasta_graph[fasta] = (c_size, features, edge_index, edge_attr)

datasets = ['ump789']
for dataset in datasets:
    for opt in ['train', 'test']:
        processed_data_file = f'data/processed/{dataset}_{opt}.pt'
        if not os.path.isfile(processed_data_file):
            df = pd.read_csv(f'data/umami/{dataset}_{opt}.csv')
            # pdb.set_trace()
            peps, Y = list(df['SEQUENCE']), list(df['TASTE'])
            XT = [seq_cat(t) for t in peps]
            peps, embeding, Y = np.asarray(peps), np.asarray(XT), np.asarray(Y)

            print(f'preparing {dataset}_{opt}.pt in pytorch format!')
            # pdb.set_trace()
            data = TestbedDataset(root='data', dataset=f'{dataset}_{opt}', xd=peps, xt=embeding, y=Y, fasta_graph=fasta_graph)
            print(f'{processed_data_file} has been created')
        else:
            print(f'{processed_data_file} is already created')
