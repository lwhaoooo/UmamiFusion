import pdb
import torch
import torch.nn as nn
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from torch_geometric.data import Data, Batch
from torch.utils.data import DataLoader
from captum.attr import Saliency
import pandas as pd

# 导入模型和工具
from models.gin_attention import GINConvNet
from utils import TestbedDataset

# ==============================
# 1. 配置与设备
# ==============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')

save_dir = "/root/autodl-tmp/new_protein_graph/data/shap"
os.makedirs(save_dir, exist_ok=True)

# ==============================
# 2. 定义数据加载函数
# ==============================
def collate_fn(batch):
    return Batch.from_data_list(batch)

# ==============================
# 3. 加载外部数据集和模型
# ==============================
dataset_name = 'biopep_test'   # 修改：外部数据集名称
model_file = 'umami_model_GINConvNet_ump442_eval.model'  # 使用原模型

# 加载测试集（外部数据集）
test_dataset = TestbedDataset(root='data', dataset=dataset_name)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

# 加载训练好的模型（与原模型相同）
model = GINConvNet().to(device)
if os.path.isfile(model_file):
    model.load_state_dict(torch.load(model_file, map_location=device))
    print(f'Loaded model from {model_file}')
else:
    raise FileNotFoundError(f'Model file {model_file} not found.')
model.eval()

# ==============================
# 4. 获取测试集批次（外部数据集可能只有一个批次）
# ==============================
all_batches = list(test_loader)
if len(all_batches) > 0:
    batch = all_batches[0].to(device)   # 取第一个（也是唯一一个）批次
else:
    raise ValueError("测试集为空！")

print(f"批次包含 {batch.batch.max().item() + 1} 个样本")

# ==============================
# 5. 模型包装函数
# ==============================
def model_wrapper(x, target, edge_index, edge_attr, batch, c_size, ptr, mask):
    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        target=target,
        batch=batch,
        c_size=c_size,
        ptr=ptr,
        mask=mask
    )
    output = model(data)
    return output

# ==============================
# 6. 提取单个样本数据（修正 target 类型）
# ==============================
def get_single_sample(batch, idx=0):
    node_mask = (batch.batch == idx)
    num_nodes = node_mask.sum().item()

    x = batch.x[node_mask]

    # 提取边并确保为 long 类型
    edge_mask = node_mask[batch.edge_index[0]]
    edge_index = batch.edge_index[:, edge_mask].long()  # 强制转换
    edge_attr = batch.edge_attr[edge_mask]

    # 调整边索引编号
    unique_nodes = torch.unique(edge_index)
    node_mapping = {n.item(): i for i, n in enumerate(unique_nodes)}
    src = torch.tensor([node_mapping[n.item()] for n in edge_index[0]], dtype=torch.long, device=device)
    dst = torch.tensor([node_mapping[n.item()] for n in edge_index[1]], dtype=torch.long, device=device)
    edge_index = torch.stack([src, dst], dim=0)

    batch_single = torch.zeros(num_nodes, dtype=torch.long, device=device)
    target_single = batch.target[idx:idx+1]  # 保持 long
    c_size_single = torch.tensor([num_nodes], device=device)
    ptr_single = torch.tensor([0, num_nodes], dtype=torch.long, device=device)

    max_nodes = 39
    mask_start = idx * max_nodes
    mask_single = batch.mask[mask_start:mask_start + max_nodes].float().unsqueeze(0)

    y_single = batch.y[idx:idx+1]

    return (x, edge_index, edge_attr, batch_single, target_single, c_size_single, ptr_single, mask_single), y_single

# ==============================
# 7. 获取外部数据集的鲜味肽序列（从CSV读取）
# ==============================
csv_path = "/root/autodl-tmp/new_protein_graph/data/umami/BIOPEP_external_set/BIOPEP_external_test_set.csv"
test_df = pd.read_csv(csv_path)
# 筛选鲜味肽（标签为1）
umami_sequences = test_df[test_df['TASTE'] == 1]['SEQUENCE'].tolist()

# 选择要可视化的样本数（最多9个，不足则全部）
num_samples = min(9, len(umami_sequences))
sequences = umami_sequences[:num_samples]

print("用于可视化的鲜味肽序列（外部数据集）:")
for i, seq in enumerate(sequences):
    print(f"  {i+1}: {seq}")

# ==============================
# 8. 初始化 Saliency 解释器
# ==============================
saliency = Saliency(model_wrapper)

# ==============================
# 9. 可视化 3×3 子图（适配实际样本数）
# ==============================
# 根据实际样本数动态调整子图布局
if num_samples <= 3:
    nrows, ncols = 1, num_samples
elif num_samples <= 6:
    nrows, ncols = 2, 3
else:
    nrows, ncols = 3, 3

fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(8*ncols, 6*nrows))
if num_samples == 1:
    axes = [axes]
else:
    axes = axes.flatten()

# 学术配色（根据需要扩展）
BAR_COLORS = [
    '#AADFF5', '#FF847C', '#FFEEBB', '#A2D3C0', '#F2C742',
    '#7498B5', '#E6E6E6', '#C4A8D1', '#D66969'
]

# 实际可用的样本索引（从批次中取）
total_samples = batch.batch.max().item() + 1
sample_indices = list(range(min(num_samples, total_samples)))

for plot_idx, idx in enumerate(sample_indices):
    print(f"\n处理样本 {idx}（对应序列：{sequences[plot_idx]}）...")

    inputs, y_true = get_single_sample(batch, idx=idx)
    x, edge_index, edge_attr, batch_single, target, c_size, ptr, mask = inputs

    explain_inputs = (x,)
    fixed_args = (target, edge_index, edge_attr, batch_single, c_size, ptr, mask)

    # 计算 Saliency
    saliency_attr_x = saliency.attribute(
        explain_inputs,
        target=0,
        additional_forward_args=fixed_args
    )

    saliency_tensor = saliency_attr_x[0]
    node_importance = torch.mean(saliency_tensor, dim=1).cpu().numpy()

    ax = axes[plot_idx]
    seq = sequences[plot_idx]
    n_residues = len(seq)
    importance = node_importance[:n_residues]

    ax.bar(
        range(n_residues),
        importance,
        color=BAR_COLORS[plot_idx % len(BAR_COLORS)],
        edgecolor='black',
        linewidth=0.5
    )
    ax.set_xticks(range(n_residues))
    ax.set_xticklabels(list(seq), fontsize=12)
    ax.set_title(f"Sample {plot_idx+1}: {seq} (True Label: {y_true.item()})", fontsize=12)
    ax.set_ylabel("Avg Importance", fontsize=12)
    ax.grid(axis='y', linestyle='--', alpha=0.6)
    ax.tick_params(axis='both', labelsize=12)

# 隐藏多余的子图（如果样本数不足）
for i in range(len(sample_indices), len(axes)):
    axes[i].set_visible(False)

plt.tight_layout()
fig.suptitle("Saliency-Based Residue Importance on External Dataset (BIOPEP-UWM)", fontsize=14, y=1.02)

# 保存
save_path_pdf = os.path.join(save_dir, "biopep_umami_saliency.pdf")
plt.savefig(save_path_pdf, format="pdf", dpi=300, bbox_inches="tight", pad_inches=0.1)
print(f"\nPDF保存至：{save_path_pdf}")

save_path_png = os.path.join(save_dir, "biopep_umami_saliency.png")
plt.savefig(save_path_png, dpi=300, bbox_inches="tight")
print(f"PNG保存至：{save_path_png}")

plt.show()
print("\nSaliency可解释性分析完成！")
