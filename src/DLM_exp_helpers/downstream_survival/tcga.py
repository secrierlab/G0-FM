import pandas as pd
import anndata as ad


def preprocess_bulk_survival_data(data_X_file, survival_label_file, map_dict = {'relapse_time':'time','relapse_status':'event'},adata_output=None):
    print(f"Preprocessing data from {data_X_file} and label from {survival_label_file}")
    # Read the data and label files
    data = pd.read_csv(data_X_file)
    label = pd.read_csv(survival_label_file)
    
    # Determine the number of columns in the label file
    label_col_count = label.shape[1] +1
    print(f"Number of columns in label file: {label_col_count}")
    
    # Extract a short version of sample_id for merging
    data['sample_id_short'] = data['sample_id'].str[:12]
    merged = pd.merge(data, label, left_on='sample_id_short', right_on='sample_id', how='inner')
    
    # Define `X`, `obs`, and `var` based on the label column count
    adata = ad.AnnData(
        X=merged.iloc[:, 1:-label_col_count].values,
        obs=merged.iloc[:, -label_col_count:], 
        var=pd.DataFrame(index=merged.columns.to_list()[1:-label_col_count])
    )
    adata.var['gene_symbols'] = merged.columns.to_list()[1:-label_col_count]
    adata.obs_names = merged.iloc[:, 0]
    
    # 重命名列
    print(f"rename columns: {map_dict}")
    for new_name, old_name in map_dict.items():
        if old_name in adata.obs.columns:
            adata.obs[new_name] = adata.obs[old_name]
        else:
            raise ValueError(f"Column {old_name} not found in adata.obs")
    
    # QC 
    print(f" Start doing QC for nan, dtype, etc.")
    # 检查'relapse_time'列中的非数值数据
    print(f"'relapse_time' dtype: {adata.obs['relapse_time'].dtype}")

    # 尝试将'relapse_time'列转换为数值类型，非数值的数据会变为NaN
    adata.obs['relapse_time'] = pd.to_numeric(adata.obs['relapse_time'], errors='coerce')

    # 检查是否有NaN值
    print(f"'relapse_time' NaN count: {adata.obs['relapse_time'].isna().sum()}")


    # 删除包含NaN值的样本
    adata = adata[~adata.obs['relapse_time'].isna(), :]

    # 检查'relapse_status'列的类型
    print(f"'relapse_status' dtype: {adata.obs['relapse_status'].dtype}")

    # 尝试将'relapse_status'转换为数值类型，非数值的数据会变为NaN
    adata.obs['relapse_status'] = pd.to_numeric(adata.obs['relapse_status'], errors='coerce')

    # 处理NaN值
    adata = adata[~adata.obs['relapse_status'].isna(), :]

    # Optionally write the AnnData object to a file
    if adata_output:
        adata.write(adata_output)
    print(f"Done with preprocessing, return adata")
    
    return adata


def load_bulk_survival_adata(adata_file):
    print(f"Loading data from {adata_file}")
    adata = ad.read_h5ad(adata_file)
    print(f"Done with loading, return adata")
    return adata
