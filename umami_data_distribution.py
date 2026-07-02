import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

# 设置字体和样式
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")

# ========== 定义函数 ==========
def get_amino_acid_frequency(df):
    """计算氨基酸频率（百分比）"""
    all_sequences = ''.join(df['SEQUENCE'].tolist())
    aa_counts = Counter(all_sequences)
    total = len(all_sequences)
    aa_freq = {aa: (count / total) * 100 for aa, count in aa_counts.items()}
    return aa_freq

def plot_amino_acid_freq(ax, train_df, test_df, title):
    """绘制氨基酸频率分布（并排条形图）"""
    train_aa_freq = get_amino_acid_frequency(train_df)
    test_aa_freq = get_amino_acid_frequency(test_df)
    
    all_amino_acids = sorted(set(list(train_aa_freq.keys()) + list(test_aa_freq.keys())))
    train_freqs = [train_aa_freq.get(aa, 0) for aa in all_amino_acids]
    test_freqs = [test_aa_freq.get(aa, 0) for aa in all_amino_acids]
    
    x = range(len(all_amino_acids))
    width = 0.35
    
    ax.bar([i - width/2 for i in x], train_freqs, width, label='Train', color='#E57373', alpha=0.8)
    ax.bar([i + width/2 for i in x], test_freqs, width, label='Test', color='#64B5F6', alpha=0.8)
    
    ax.set_xlabel('Amino Acids', fontsize=11)
    ax.set_ylabel('Frequency (%)', fontsize=11)
    ax.set_title(title, fontsize=13, pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(all_amino_acids)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

def plot_seq_length_dist(ax, train_df, test_df, title):
    """绘制序列长度分布（并排条形图）"""
    train_df = train_df.copy()
    test_df = test_df.copy()
    train_df['seq_length'] = train_df['SEQUENCE'].str.len()
    test_df['seq_length'] = test_df['SEQUENCE'].str.len()
    
    all_lengths = list(train_df['seq_length']) + list(test_df['seq_length'])
    min_len, max_len = min(all_lengths), max(all_lengths)
    
    train_length_counts = train_df['seq_length'].value_counts().sort_index()
    test_length_counts = test_df['seq_length'].value_counts().sort_index()
    
    length_range = range(min_len, max_len + 1)
    train_counts = [train_length_counts.get(l, 0) for l in length_range]
    test_counts = [test_length_counts.get(l, 0) for l in length_range]
    
    x = list(length_range)
    width = 0.4
    
    ax.bar([i - width/2 for i in x], train_counts, width, label='Train', color='#E57373', alpha=0.8)
    ax.bar([i + width/2 for i in x], test_counts, width, label='Test', color='#64B5F6', alpha=0.8)
    
    ax.set_xlabel('Sequence Length', fontsize=11)
    ax.set_ylabel('Frequency', fontsize=11)
    ax.set_title(title, fontsize=13, pad=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')

# ========== 加载数据 ==========
# UMP442
train_442 = pd.read_csv('/root/autodl-tmp/new_protein_graph/data/umami/ump442_train.csv')
test_442 = pd.read_csv('/root/autodl-tmp/new_protein_graph/data/umami/ump442_test.csv')

# UMP614
train_614 = pd.read_csv('/root/autodl-tmp/new_protein_graph/data/umami/ump789_train.csv')
test_614 = pd.read_csv('/root/autodl-tmp/new_protein_graph/data/umami/ump789_test.csv')

# ========== 创建2×2子图 ==========
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# A: UMP442 氨基酸频率
plot_amino_acid_freq(axes[0, 0], train_442, test_442, 'A. UMP442 Amino Acid Frequency Distribution')

# B: UMP442 序列长度
plot_seq_length_dist(axes[0, 1], train_442, test_442, 'B. UMP442 Sequence Length Distribution')

# C: UMP614 氨基酸频率
plot_amino_acid_freq(axes[1, 0], train_614, test_614, 'C. UMP614 Amino Acid Frequency Distribution')

# D: UMP614 序列长度
plot_seq_length_dist(axes[1, 1], train_614, test_614, 'D. UMP614 Sequence Length Distribution')

# 调整布局
plt.tight_layout()

# 保存为PDF
output_path = "/root/autodl-tmp/new_protein_graph/data/umami/shujufenbu/UMP442_UMP614_Distributions.pdf"
plt.savefig(output_path, format="pdf", dpi=300, bbox_inches="tight", pad_inches=0.1)
print(f"图表已保存至: {output_path}")

# 显示
plt.show()

# ========== 打印统计信息 ==========
def print_stats(df, name):
    print(f"\n{name}:")
    print("="*50)
    print(f"总序列数: {len(df)}")
    print(f"序列长度范围: {df['SEQUENCE'].str.len().min()} - {df['SEQUENCE'].str.len().max()}")
    print(f"平均长度: {df['SEQUENCE'].str.len().mean():.2f}")

print_stats(train_442, "UMP442 训练集")
print_stats(test_442, "UMP442 测试集")
print_stats(train_614, "UMP614 训练集")
print_stats(test_614, "UMP614 测试集")
