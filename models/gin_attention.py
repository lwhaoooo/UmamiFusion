import pdb

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import GINConv, global_add_pool


# 节点级别的交互注意力模块
class NodeLevelCrossAttention(nn.Module):
    """
    节点级别的双向交互注意力机制
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

    def forward(self, graph_feat, seq_feat, batch, mask=None):
        """
        Args:
            graph_feat: [total_nodes, dim] 所有batch的图节点特征
            seq_feat: [batch_size, seq_len, dim] 序列特征
            batch: [total_nodes] 每个节点属于哪个batch
            mask: [batch_size, max_nodes] 可选，标记哪些节点是真实的
        Returns:
            fused_graph: [batch_size, max_nodes, dim] 融合后的图节点特征
            fused_seq: [batch_size, seq_len, dim] 融合后的序列特征
        """
        batch_size = seq_feat.size(0)
        seq_len = seq_feat.size(1)
        total_nodes = graph_feat.size(0)

        # 计算每个batch的节点数（假设都是39）
        max_nodes = total_nodes // batch_size

        # 将graph_feat重塑为 [batch_size, max_nodes, dim]
        graph_feat_batched = torch.zeros(batch_size, max_nodes, self.dim,
                                         device=graph_feat.device, dtype=graph_feat.dtype)

        for i in range(batch_size):
            mask_i = (batch == i)
            graph_feat_batched[i] = graph_feat[mask_i]

        # 处理mask
        if mask is not None:
            if mask.dim() == 1:
                mask_reshaped = mask.view(batch_size, max_nodes)
            else:
                mask_reshaped = mask
        else:
            mask_reshaped = None

        # ===== 图节点关注序列 =====
        q_g = self.q_graph(graph_feat_batched).view(batch_size, max_nodes, self.num_heads, self.head_dim).transpose(1,
                                                                                                                    2)
        k_s = self.k_seq(seq_feat).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v_s = self.v_seq(seq_feat).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        attn_graph = torch.matmul(q_g, k_s.transpose(-2, -1)) * self.scale

        if mask_reshaped is not None:
            seq_mask_expanded = mask_reshaped.unsqueeze(1).unsqueeze(2)
            attn_graph = attn_graph.masked_fill(seq_mask_expanded == 0, -1e9)

        attn_graph = F.softmax(attn_graph, dim=-1)
        attn_graph = self.dropout(attn_graph)

        out_graph = torch.matmul(attn_graph, v_s)
        out_graph = out_graph.transpose(1, 2).contiguous().view(batch_size, max_nodes, self.dim)
        out_graph = self.out_graph(out_graph)

        if mask_reshaped is not None:
            graph_mask_expanded = mask_reshaped.unsqueeze(-1)
            out_graph = out_graph * graph_mask_expanded

        fused_graph = out_graph

        # ===== 序列关注图节点 =====
        q_s = self.q_seq(seq_feat).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k_g = self.k_graph(graph_feat_batched).view(batch_size, max_nodes, self.num_heads, self.head_dim).transpose(1,
                                                                                                                    2)
        v_g = self.v_graph(graph_feat_batched).view(batch_size, max_nodes, self.num_heads, self.head_dim).transpose(1,
                                                                                                                    2)

        attn_seq = torch.matmul(q_s, k_g.transpose(-2, -1)) * self.scale

        if mask_reshaped is not None:
            mask_expanded = mask_reshaped.unsqueeze(1).unsqueeze(2)
            attn_seq = attn_seq.masked_fill(mask_expanded == 0, -1e9)

        attn_seq = F.softmax(attn_seq, dim=-1)
        attn_seq = self.dropout(attn_seq)

        out_seq = torch.matmul(attn_seq, v_g)
        out_seq = out_seq.transpose(1, 2).contiguous().view(batch_size, seq_len, self.dim)
        out_seq = self.out_seq(out_seq)

        if mask_reshaped is not None:
            seq_mask_expanded = mask_reshaped.unsqueeze(-1)
            out_seq = out_seq * seq_mask_expanded

        fused_seq = out_seq

        return fused_graph, fused_seq


# GIN图分支模块(输出节点级特征)
class GINBranchWithAttention(nn.Module):
    def __init__(self, num_features_xd=320, node_dim=32, n_output=1, dropout=0.2):
        super(GINBranchWithAttention, self).__init__()
        self.node_dim = node_dim
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.n_output = n_output

        # GIN卷积层
        nn1 = Sequential(Linear(num_features_xd, node_dim), ReLU(), Linear(node_dim, node_dim))
        self.conv1 = GINConv(nn1)
        self.bn1 = torch.nn.BatchNorm1d(node_dim)

        nn2 = Sequential(Linear(node_dim, node_dim), ReLU(), Linear(node_dim, node_dim))
        self.conv2 = GINConv(nn2)
        self.bn2 = torch.nn.BatchNorm1d(node_dim)

        nn3 = Sequential(Linear(node_dim, node_dim), ReLU(), Linear(node_dim, node_dim))
        self.conv3 = GINConv(nn3)
        self.bn3 = torch.nn.BatchNorm1d(node_dim)

        nn4 = Sequential(Linear(node_dim, node_dim), ReLU(), Linear(node_dim, node_dim))
        self.conv4 = GINConv(nn4)
        self.bn4 = torch.nn.BatchNorm1d(node_dim)

        nn5 = Sequential(Linear(node_dim, node_dim), ReLU(), Linear(node_dim, node_dim))
        self.conv5 = GINConv(nn5)
        self.bn5 = torch.nn.BatchNorm1d(node_dim)

    def forward(self, x, edge_index):
        """
        Returns:
            node_features: [total_nodes, node_dim] 节点级特征,用于交互注意力
        """
        # 图卷积处理 - 保持节点级特征
        # pdb.set_trace()
        x = F.relu(self.conv1(x, edge_index))
        x = self.bn1(x)
        x = F.relu(self.conv2(x, edge_index))
        x = self.bn2(x)
        x = F.relu(self.conv3(x, edge_index))
        x = self.bn3(x)
        x = F.relu(self.conv4(x, edge_index))
        x = self.bn4(x)
        x = F.relu(self.conv5(x, edge_index))
        x = self.bn5(x)

        return x  # [total_nodes, node_dim]


# 蛋白质序列分支模块(输出序列级特征) - 三层CNN版本
class ProteinBranchWithAttention(nn.Module):
    def __init__(self, num_features_xt=25, embed_dim=128, n_filters=32,
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

        # 三层卷积层设计 (保持长度)
        self.conv1 = nn.Conv1d(in_channels=embed_dim, out_channels=n_filters,
                               kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm1d(n_filters)

        self.conv2 = nn.Conv1d(in_channels=n_filters, out_channels=n_filters * 2,
                               kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(n_filters * 2)

        self.conv3 = nn.Conv1d(in_channels=n_filters * 2, out_channels=n_filters,
                               kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(n_filters)

        # 自适应池化层 - 将任意长度序列压缩到target_seq_len
        self.adaptive_pool = nn.AdaptiveAvgPool1d(target_seq_len)

    def forward(self, target):
        """
        Args:
            target: [batch_size, seq_len] 序列输入 (如 [32, 1000])
        Returns:
            seq_features: [batch_size, target_seq_len, n_filters] 序列级特征,用于交互注意力
        """
        # 嵌入层: [batch_size, seq_len] -> [batch_size, seq_len, embed_dim]

        # pdb.set_trace()

        embedded = self.embedding(target)

        # 调整维度以适应Conv1d: [batch_size, embed_dim, seq_len]
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

        # 自适应池化到目标序列长度
        x = self.adaptive_pool(x)  # [batch_size, n_filters, target_seq_len]

        # 转置为 [batch_size, target_seq_len, n_filters]
        conv_out = x.permute(0, 2, 1)

        return conv_out  # [batch_size, target_seq_len, n_filters]


# 多任务融合模型(带交互注意力)
class MultiTaskFusionNetWithAttention(nn.Module):
    def __init__(self, gin_branch, protein_branch, output_dim=128,
                 num_heads=4, n_output=1, dropout=0.2):
        super(MultiTaskFusionNetWithAttention, self).__init__()
        self.gin_branch = gin_branch
        self.protein_branch = protein_branch

        # 获取特征维度(图和序列的特征维度应该相同才能做交互注意力)
        node_dim = gin_branch.node_dim
        seq_dim = protein_branch.n_filters
        target_seq_len = protein_branch.target_seq_len

        assert node_dim == seq_dim, "图节点特征维度和序列特征维度必须相同才能使用交互注意力"

        # 交互注意力模块
        self.cross_attention = NodeLevelCrossAttention(
            dim=node_dim,
            num_heads=num_heads,
            dropout=dropout
        )

        # 注意力后的图特征处理
        self.fc_graph = Linear(node_dim, output_dim)

        # 注意力后的序列特征处理
        self.fc_seq = nn.Linear(seq_dim * target_seq_len, output_dim)

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

        # 用于返回中间特征
        self.intermediate_fc = Linear(fusion_dim, 256)
        self.dropout_layer = nn.Dropout(dropout)
        self.relu = ReLU()

    def forward(self, data, return_features=False):
        """
        ⭐ 关键修改：添加 return_features 参数控制返回值

        Args:
            data: PyG数据对象,包含:
                - x: 节点特征
                - edge_index: 边索引
                - batch: 节点所属batch
                - target: 序列输入
                - mask: (可选)节点/序列mask
            return_features: 是否返回中间特征(默认False,适配训练代码)
        Returns:
            如果 return_features=False: 只返回 prediction [适配训练代码]
            如果 return_features=True: 返回 (prediction, intermediate_features)
        """
        # 提取图节点特征

        graph_node_features = self.gin_branch(data.x, data.edge_index)  # [total_nodes, node_dim]

        # 提取序列特征
        seq_features = self.protein_branch(data.target)  # [batch_size, seq_len, seq_dim]

        # 获取mask(如果有)
        mask = data.mask if hasattr(data, 'mask') else None

        # 使用交互注意力
        graph_attended, seq_attended = self.cross_attention(
            graph_node_features,
            seq_features,
            data.batch,
            mask
        )

        # 残差连接
        batch_size = seq_features.size(0)
        total_nodes = graph_node_features.size(0)
        max_nodes = total_nodes // batch_size

        graph_node_batched = torch.zeros(batch_size, max_nodes, self.gin_branch.node_dim,
                                         device=graph_node_features.device,
                                         dtype=graph_node_features.dtype)
        for i in range(batch_size):
            mask_i = (data.batch == i)
            graph_node_batched[i] = graph_node_features[mask_i]


        graph_fused = graph_node_batched + graph_attended  # [batch_size, max_nodes, node_dim]
        seq_fused = seq_features + seq_attended  # [batch_size, seq_len, seq_dim]

        # 对图节点特征进行池化
        graph_fused_flat = graph_fused.view(-1, self.gin_branch.node_dim)
        graph_pooled = global_add_pool(graph_fused_flat, data.batch)
        graph_pooled = F.relu(self.fc_graph(graph_pooled))
        graph_pooled = F.dropout(graph_pooled, p=0.2, training=self.training)

        # pdb.set_trace()

        # 对序列特征进行flatten
        seq_flat = seq_fused.reshape(seq_fused.size(0), -1)
        seq_pooled = self.fc_seq(seq_flat)

        # 特征融合
        fused_features = torch.cat((graph_pooled, seq_pooled), dim=1)

        # 融合预测
        prediction = self.fusion_predictor(fused_features)

        # ⭐ 根据参数决定返回值
        if return_features:
            # 计算中间特征(用于可视化或分析)
            intermediate_features = self.intermediate_fc(fused_features)
            intermediate_features = self.relu(intermediate_features)
            intermediate_features = self.dropout_layer(intermediate_features)
            # pdb.set_trace()
            return prediction, intermediate_features
        else:
            # 默认只返回prediction (适配训练代码)
            return prediction


# ============================================
# 工厂函数：创建模型实例
# ============================================
def GINConvNet(num_features_xd=320, num_features_xt=25, n_output=1):
    """
    工厂函数：创建带交互注意力的GIN模型

    这个函数签名与训练代码中的 modeling() 调用兼容:
        model = modeling().to(device)

    Args:
        num_features_xd: 分子图节点特征维度
        num_features_xt: 蛋白质序列词汇表大小
        n_output: 输出维度 (二分类为1)

    Returns:
        MultiTaskFusionNetWithAttention: 完整的融合模型
    """
    # 超参数配置
    embed_dim = 128  # embedding维度
    node_dim = 32  # 图节点特征维度
    n_filters = 32  # CNN卷积核数量 (必须等于node_dim)
    output_dim = 128  # 融合后的特征维度
    target_seq_len = 39  # 目标序列长度(与图节点数匹配)
    num_heads = 4  # 注意力头数
    dropout = 0.2  # dropout率

    # 创建图分支
    gin_branch = GINBranchWithAttention(
        num_features_xd=num_features_xd,
        node_dim=node_dim,
        n_output=n_output,
        dropout=dropout
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
        gin_branch=gin_branch,
        protein_branch=protein_branch,
        output_dim=output_dim,
        num_heads=num_heads,
        n_output=n_output,
        dropout=dropout
    )

    return model
