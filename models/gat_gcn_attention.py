import pdb

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import GATConv, GCNConv, global_add_pool
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp


# 节点级别的交互注意力模块（无mask版本）
class NodeLevelCrossAttention(nn.Module):
    """
    节点级别的双向交互注意力机制（简化版，移除mask）
    - 图的每个节点关注序列的所有位置
    - 序列的每个位置关注图的所有节点
    """

    def __init__(self, dim, num_heads=4, dropout=0.1):
        super(NodeLevelCrossAttention, self).__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.head_dim = dim // num_heads

        assert self.head_dim * num_heads == dim, "dim必须能被num_heads整除"

        # 图节点->序列的注意力
        self.q_graph = Linear(dim, dim)
        self.k_seq = Linear(dim, dim)
        self.v_seq = Linear(dim, dim)

        # 序列->图节点的注意力
        self.q_seq = Linear(dim, dim)
        self.k_graph = Linear(dim, dim)
        self.v_graph = Linear(dim, dim)

        # 输出投影
        self.out_graph = Linear(dim, dim)
        self.out_seq = Linear(dim, dim)

        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5

    def forward(self, graph_feat, seq_feat, batch):
        """
        Args:
            graph_feat: [total_nodes, dim] 所有batch的图节点特征
            seq_feat: [batch_size, seq_len, dim] 序列特征
            batch: [total_nodes] 每个节点属于哪个batch
        Returns:
            fused_graph: [batch_size, max_nodes, dim] 融合后的图节点特征
            fused_seq: [batch_size, seq_len, dim] 融合后的序列特征
        """
        batch_size = seq_feat.size(0)
        seq_len = seq_feat.size(1)
        total_nodes = graph_feat.size(0)

        # 计算每个batch的节点数
        max_nodes = total_nodes // batch_size

        # 将graph_feat重塑为 [batch_size, max_nodes, dim]
        graph_feat_batched = torch.zeros(batch_size, max_nodes, self.dim,
                                         device=graph_feat.device, dtype=graph_feat.dtype)

        for i in range(batch_size):
            mask_i = (batch == i)
            graph_feat_batched[i] = graph_feat[mask_i]

        # ===== 图节点关注序列 =====
        q_g = self.q_graph(graph_feat_batched).view(batch_size, max_nodes, self.num_heads, self.head_dim).transpose(1,
                                                                                                                    2)
        k_s = self.k_seq(seq_feat).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v_s = self.v_seq(seq_feat).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # 注意力计算（无mask）
        attn_graph = torch.matmul(q_g, k_s.transpose(-2, -1)) * self.scale
        attn_graph = F.softmax(attn_graph, dim=-1)
        attn_graph = self.dropout(attn_graph)

        out_graph = torch.matmul(attn_graph, v_s)
        out_graph = out_graph.transpose(1, 2).contiguous().view(batch_size, max_nodes, self.dim)
        out_graph = self.out_graph(out_graph)

        # ===== 序列关注图节点 =====
        q_s = self.q_seq(seq_feat).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k_g = self.k_graph(graph_feat_batched).view(batch_size, max_nodes, self.num_heads, self.head_dim).transpose(1,
                                                                                                                    2)
        v_g = self.v_graph(graph_feat_batched).view(batch_size, max_nodes, self.num_heads, self.head_dim).transpose(1,
                                                                                                                    2)

        # 注意力计算（无mask）
        attn_seq = torch.matmul(q_s, k_g.transpose(-2, -1)) * self.scale
        attn_seq = F.softmax(attn_seq, dim=-1)
        attn_seq = self.dropout(attn_seq)

        out_seq = torch.matmul(attn_seq, v_g)
        out_seq = out_seq.transpose(1, 2).contiguous().view(batch_size, seq_len, self.dim)
        out_seq = self.out_seq(out_seq)

        return out_graph, out_seq


# GAT_GCN分子图分支模块（替换原GIN分支）
class GAT_GCN_BranchWithAttention(nn.Module):
    def __init__(self, num_features_xd=320, node_dim=128, output_dim=128,
                 n_output=1, dropout=0.2, gat_heads=5):
        super(GAT_GCN_BranchWithAttention, self).__init__()
        self.node_dim = node_dim
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.n_output = n_output

        # GAT + GCN 图卷积层
        self.conv1 = GATConv(num_features_xd, num_features_xd, heads=gat_heads)
        self.conv2 = GCNConv(num_features_xd * gat_heads, num_features_xd * gat_heads)

        # 全连接层（使用global max pool和global mean pool的拼接）
        self.fc_g1 = torch.nn.Linear(num_features_xd * gat_heads * 2, 1500)
        self.fc_g2 = torch.nn.Linear(1500, node_dim)

        # 预测头
        self.predictor = nn.Sequential(
            Linear(node_dim, 64),
            ReLU(),
            nn.Dropout(dropout),
            Linear(64, n_output)
        )

    def forward(self, x, edge_index, batch):
        """
        Args:
            x: [total_nodes, num_features_xd] 节点特征
            edge_index: [2, num_edges] 边索引
            batch: [total_nodes] 每个节点属于哪个batch
        Returns:
            features: [batch_size, node_dim] 图级别特征（用于后续展开到节点级）
        """
        # GAT层
        # pdb.set_trace()
        x = self.conv1(x, edge_index)
        x = self.relu(x)

        # GCN层
        x = self.conv2(x, edge_index)
        x = self.relu(x)

        # 全局池化：拼接max pooling和mean pooling
        x = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        # 全连接层
        x = self.relu(self.fc_g1(x))
        x = self.dropout(x)
        x = self.fc_g2(x)

        return x


# 蛋白质序列分支模块(输出序列级特征)
class ProteinBranchWithAttention(nn.Module):
    def __init__(self, num_features_xt=25, embed_dim=128, n_filters=128,
                 target_seq_len=39, dropout=0.2):
        """
        Args:
            num_features_xt: 氨基酸词汇表大小
            embed_dim: embedding维度
            n_filters: 基础卷积核数量
            target_seq_len: 目标序列长度(用于匹配图节点数,如39)
            dropout: dropout率
        """
        super(ProteinBranchWithAttention, self).__init__()
        self.n_filters = n_filters
        self.target_seq_len = target_seq_len
        self.dropout = nn.Dropout(dropout)

        # 嵌入层
        self.embedding = nn.Embedding(num_features_xt + 1, embed_dim)

        # 三层卷积层设计
        self.conv1 = nn.Conv1d(in_channels=embed_dim, out_channels=n_filters,
                               kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm1d(n_filters)

        self.conv2 = nn.Conv1d(in_channels=n_filters, out_channels=n_filters * 2,
                               kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(n_filters * 2)

        self.conv3 = nn.Conv1d(in_channels=n_filters * 2, out_channels=n_filters,
                               kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(n_filters)

    def forward(self, target):
        """
        Args:
            target: [batch_size, seq_len] 序列输入
        Returns:
            seq_features: [batch_size, seq_len, n_filters] 序列级特征
        """
        embedded = self.embedding(target)
        conv_input = embedded.permute(0, 2, 1)

        # 卷积块1
        x = F.relu(self.bn1(self.conv1(conv_input)))
        x = F.dropout(x, p=0.2, training=self.training)

        # 卷积块2
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.dropout(x, p=0.2, training=self.training)

        # 卷积块3
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.dropout(x, p=0.2, training=self.training)

        # 转置为 [batch_size, seq_len, n_filters]
        conv_out = x.permute(0, 2, 1)

        return conv_out


class MultiTaskFusionNetWithAttention(nn.Module):
    def __init__(self, graph_branch, protein_branch, output_dim=128,
                 num_heads=4, n_output=1, dropout=0.2):
        super().__init__()
        self.graph_branch = graph_branch
        self.protein_branch = protein_branch

        node_dim = graph_branch.node_dim
        seq_dim = protein_branch.n_filters
        target_seq_len = 39  # 固定为39

        # 交互注意力模块
        self.cross_attention = NodeLevelCrossAttention(
            dim=node_dim,
            num_heads=num_heads,
            dropout=dropout
        )

        # 简化：直接处理注意力输出
        self.fc_graph = Linear(node_dim, output_dim)
        self.fc_seq = Linear(seq_dim * target_seq_len, output_dim)

        # 融合预测器
        fusion_dim = output_dim * 2
        self.fusion_predictor = nn.Sequential(
            Linear(fusion_dim, 1024),
            ReLU(),
            nn.Dropout(dropout),
            Linear(1024, 256),
            ReLU(),
            nn.Dropout(dropout),
            Linear(256, n_output)
        )

    def forward(self, data, return_features=False):
        # 提取图级别特征
        graph_features = self.graph_branch(data.x, data.edge_index, data.batch)
        seq_features = self.protein_branch(data.target)

        # 将图级别特征扩展到节点级别
        batch_size = graph_features.size(0)
        num_nodes_per_graph = 39  # 假设每个图有39个节点

        # 将图特征复制到每个节点：[batch_size, node_dim] -> [total_nodes, node_dim]
        graph_node_features = graph_features.repeat_interleave(num_nodes_per_graph, dim=0)

        # 交互注意力
        graph_attended, seq_attended = self.cross_attention(
            graph_node_features,
            seq_features,
            data.batch
        )

        # 对图的注意力输出进行池化
        graph_attended_flat = graph_attended.view(-1, self.graph_branch.node_dim)

        # 重新构建batch索引（每个样本39个节点）
        batch_indices = torch.arange(batch_size, device=data.batch.device).repeat_interleave(39)

        graph_pooled = global_add_pool(graph_attended_flat, batch_indices)
        graph_pooled = F.relu(self.fc_graph(graph_pooled))
        graph_pooled = F.dropout(graph_pooled, p=0.2, training=self.training)

        # 对序列的注意力输出进行flatten
        seq_flat = seq_attended.reshape(batch_size, -1)
        seq_pooled = self.fc_seq(seq_flat)

        # 拼接并预测
        fused_features = torch.cat((graph_pooled, seq_pooled), dim=1)
        prediction = self.fusion_predictor(fused_features)

        if return_features:
            return prediction, fused_features
        return prediction


# ============================================
# 工厂函数：创建模型实例
# ============================================
def GAT_GCN_ConvNet(num_features_xd=320, num_features_xt=25, n_output=1, gat_heads=5):
    """
    工厂函数：创建带交互注意力的GAT_GCN模型
    """
    # 超参数配置
    embed_dim = 128
    node_dim = 128
    n_filters = 128
    output_dim = 128
    target_seq_len = 39
    num_heads = 4
    dropout = 0.2

    # 创建GAT_GCN图分支
    graph_branch = GAT_GCN_BranchWithAttention(
        num_features_xd=num_features_xd,
        node_dim=node_dim,
        output_dim=output_dim,
        n_output=n_output,
        dropout=dropout,
        gat_heads=gat_heads
    )

    # 创建序列分支
    protein_branch = ProteinBranchWithAttention(
        num_features_xt=num_features_xt,
        embed_dim=embed_dim,
        n_filters=n_filters,
        target_seq_len=target_seq_len,
        dropout=dropout
    )

    # 创建融合模型
    model = MultiTaskFusionNetWithAttention(
        graph_branch=graph_branch,
        protein_branch=protein_branch,
        output_dim=output_dim,
        num_heads=num_heads,
        n_output=n_output,
        dropout=dropout
    )

    return model
