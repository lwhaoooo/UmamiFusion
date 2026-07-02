import pandas as pd

# ==================== 配置区 ====================
# 请根据您的实际文件路径修改
file_biopep = 'BIOPEP_umami_peptides.xlsx'
file_ump442 = 'UMP442_umami.xlsx'
file_ump614 = 'UMP614_umami.xlsx'

# 请指定每张表中肽序列所在的列名
# 如果您的列名不同，请修改下面的变量
col_biopep = 'SEQUENCE'   # BIOPEP 表中的序列列名
col_ump442 = 'SEQUENCE'   # UMP442 表中的序列列名（也可能是 'Sequence' 或 'Peptide'）
col_ump614 = 'SEQUENCE'   # UMP614 表中的序列列名
# ===============================================

# 1. 读取数据
df_biopep = pd.read_excel(file_biopep)
df_ump442 = pd.read_excel(file_ump442)
df_ump614 = pd.read_excel(file_ump614)

# 2. 提取序列集合（去除空格，统一大写以进行比较）
def get_sequence_set(df, col):
    # 检查列是否存在，若不存在则抛出提示
    if col not in df.columns:
        raise KeyError(f"列 '{col}' 在数据中未找到，可用列: {df.columns.tolist()}")
    # 提取并转换为大写，去除首尾空格
    seqs = df[col].astype(str).str.strip().str.upper()
    return set(seqs)

seq_ump442 = get_sequence_set(df_ump442, col_ump442)
seq_ump614 = get_sequence_set(df_ump614, col_ump614)

# 合并两个数据集的序列集合
existing_seqs = seq_ump442 | seq_ump614

print(f"UMP442 中肽段数: {len(seq_ump442)}")
print(f"UMP614 中肽段数: {len(seq_ump614)}")
print(f"合并后共有 {len(existing_seqs)} 个独特肽段")

# 3. 过滤 BIOPEP 中的数据，保留未在已有集合中出现的
def filter_biopep(df, col, existing_set):
    # 将 BIOPEP 序列标准化
    df['seq_upper'] = df[col].astype(str).str.strip().str.upper()
    # 保留不在 existing_set 中的
    mask = ~df['seq_upper'].isin(existing_set)
    filtered = df[mask].copy()
    # 删除临时列
    filtered.drop(columns=['seq_upper'], inplace=True)
    return filtered

df_filtered = filter_biopep(df_biopep, col_biopep, existing_seqs)

# 4. 重新编号（从0开始）
df_filtered.reset_index(drop=True, inplace=True)
df_filtered['Column1'] = range(len(df_filtered))

# 确保列顺序为：Column1, SEQUENCE, TASTE
# 保留原有列名，如果有其他列则仅保留这三列
output_columns = ['Column1', col_biopep, 'TASTE']
# 如果 TASTE 列不存在，则新建一个全为1的列
if 'TASTE' not in df_filtered.columns:
    df_filtered['TASTE'] = 1

df_final = df_filtered[output_columns]

# 5. 保存结果
output_file = 'BIOPEP_external_test_set.xlsx'
df_final.to_excel(output_file, index=False)
print(f"过滤完成！原始 BIOPEP 中有 {len(df_biopep)} 条，保留了 {len(df_final)} 条未在 UMP442 或 UMP614 中出现的肽段。")
print(f"结果已保存至: {output_file}")
