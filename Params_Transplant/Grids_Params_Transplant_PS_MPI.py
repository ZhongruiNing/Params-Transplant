import numpy as np
import pandas as pd
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))

from netCDF4 import Dataset
from scipy.spatial.distance import mahalanobis
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances

from mpi4py import MPI

def calculate_similarity(target_properties, basin_properties, method='mahalanobis'):
    """
    计算目标流域与其他流域之间的相似性。
    可以使用马氏距离或余弦相似度。
    
    参数：
    -----------
    target_properties: 目标流域的属性 (1x19 numpy 数组或 pandas Series)
    basin_properties: 所有流域的属性 (N x 19 numpy 数组或 pandas DataFrame)
    method: 计算相似度的方法 ('mahalanobis' 或 'cosine')
    
    返回：
    --------
    list: 按相似性排序的流域索引
    """
    if method == 'mahalanobis':
        # 计算马氏距离
        covariance_matrix = np.cov(basin_properties.T)
        inverse_cov_matrix = np.linalg.inv(covariance_matrix)
        similarities = []
        
        for basin in basin_properties:
            dist = mahalanobis(np.squeeze(target_properties), basin, inverse_cov_matrix)
            similarities.append(dist)
    
    elif method == 'cosine':
        # 计算余弦相似度
        similarities = pairwise_distances([target_properties], basin_properties, metric='cosine')[0]
    
    # 按相似度排序（升序）
    sorted_indices = np.argsort(similarities)
    return sorted_indices

def get_params_by_PhS(basin_properties, params, target_properties, N=5, method='mahalanobis'):
    """
    使用物理相似性方法为目标流域获取参数。
    
    参数：
    -----------
    basin_properties: 所有流域的属性，包含19个属性 (numpy 数组或 pandas DataFrame)
    params: 所有流域的参数 (numpy 数组或 pandas DataFrame)
    target_properties: 目标流域的属性 (1x19 numpy 数组或 pandas Series)
    N: 用于参数移植的相似流域数量
    method: 计算相似度的方法 ('mahalanobis' 或 'cosine')
    
    返回：
    --------
    numpy.array: 目标流域的参数（N个最相似流域的参数平均值）
    """
    # 标准化流域属性
    scaler = StandardScaler()
    basin_properties_scaled = scaler.fit_transform(basin_properties)
    target_properties_scaled = scaler.transform(target_properties)  # 标准化目标流域属性
    
    # 使用PCA选择最重要的属性
    pca = PCA(n_components=0.95)  # 保留95%的方差
    basin_properties_pca = pca.fit_transform(basin_properties_scaled)
    target_properties_pca = pca.transform(target_properties_scaled)
    
    # 计算目标流域与所有供体流域之间的相似性
    similar_basins = calculate_similarity(target_properties_pca, basin_properties_pca, method)
    
    # 选择N个最相似的供体流域
    selected_params = params[similar_basins[:N]]
    
    # 返回N个供体流域参数的平均值作为目标流域的参数
    target_params = np.mean(selected_params, axis=0)
    
    return target_params

def regroup(series, num):
    series_len = len(series)
    num_per_group = int(np.ceil(series_len / num))

    start_idx = np.arange(0, (num - 1) * num_per_group + 1, num_per_group)
    end_idx = np.arange(num_per_group, series_len + 1, num_per_group)
    if len(end_idx) < num:
        end_idx = np.append(end_idx, series_len)
    else:
        end_idx[num - 1] = series_len
    RESULT = np.zeros((2, num), dtype=int)
    for i in range(num):
        RESULT[0, i] = start_idx[i]
        RESULT[1, i] = end_idx[i]
    return RESULT

# 获取全局编号
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

Basin_Properties = pd.read_csv("../../Data/Properties/Basin_Properties.txt", sep = '\t', header=0, index_col='stat_num')
basin_props_np   = Basin_Properties[['Climate', 'Clay', 'Silt', 'Sand', 'Slope', 'BFI', 'PRE', 'TMP', 'PET', 'TMAX', 'TMIN', 'NDVI', 'TI']].to_numpy()

params_mYWBM  = pd.read_csv("../../Data/Params/03_mYWBM_Best_Params_CF.txt", sep = '\t', header=0, index_col='stat_num')
params_abcd   = pd.read_csv("../../Data/Params/03_abcd_Best_Params_CF.txt", sep = '\t', header=0, index_col='stat_num')
params_DWBM   = pd.read_csv("../../Data/Params/03_DWBM_Best_Params_CF.txt", sep = '\t', header=0, index_col='stat_num')
params_mYWBM_np = params_mYWBM.to_numpy()
params_abcd_np  = params_abcd.to_numpy()
params_DWBM_np  = params_DWBM.to_numpy()

grids_prop    = pd.read_csv("../../Data/Grids_Prop/Grids_Properties.txt", sep='\t', index_col='NUM')
grids_prop_np = grids_prop[['Climate', 'Clay', 'Silt', 'Sand', 'Slope', 'BFI', 'PRE', 'TMP', 'PET', 'TMAX', 'TMIN', 'NDVI', 'TI']].to_numpy()

idx_list = regroup(np.arange(grids_prop.shape[0]), 196)
start_list = idx_list[0, :]
end_list = idx_list[1, :]

if __name__ == "__main__":
    grids_params_YM_PhS  = np.full((grids_prop.shape[0], 8), np.nan)
    grids_params_AM_PhS  = np.full((grids_prop.shape[0], 8), np.nan)
    grids_params_DM_PhS  = np.full((grids_prop.shape[0], 8), np.nan)

    for i in range(start_list[rank], end_list[rank]):
        print(i)

        grids_params_YM_PhS[i, :5] = get_params_by_PhS(basin_props_np, params_mYWBM_np, grids_prop_np[i].reshape(1, -1))
        grids_params_AM_PhS[i, :5] = get_params_by_PhS(basin_props_np, params_abcd_np, grids_prop_np[i].reshape(1, -1))
        grids_params_DM_PhS[i, :5] = get_params_by_PhS(basin_props_np, params_DWBM_np, grids_prop_np[i].reshape(1, -1))

        grids_params_YM_PhS[i, 5:8] = np.array([grids_prop.index[i], grids_prop['HANG'].iloc[i], grids_prop['LIE'].iloc[i]])
        grids_params_AM_PhS[i, 5:8] = np.array([grids_prop.index[i], grids_prop['HANG'].iloc[i], grids_prop['LIE'].iloc[i]])
        grids_params_DM_PhS[i, 5:8] = np.array([grids_prop.index[i], grids_prop['HANG'].iloc[i], grids_prop['LIE'].iloc[i]])

    grids_params_YM_PhS_df = pd.DataFrame(grids_params_YM_PhS, columns=['Ks', 'Kg', 'alpha', 'smax', 'Ksn', 'NUM', 'HANG', 'LIE'])
    grids_params_AM_PhS_df = pd.DataFrame(grids_params_AM_PhS, columns=['a', 'b', 'c', 'd', 'Ksn', 'NUM', 'HANG', 'LIE'])
    grids_params_DM_PhS_df = pd.DataFrame(grids_params_DM_PhS, columns=['alpha1', 'alpha2', 'smax', 'd', 'Ksn', 'NUM', 'HANG', 'LIE'])

    grids_params_YM_PhS_df.dropna().reset_index(drop=True)
    grids_params_AM_PhS_df.dropna().reset_index(drop=True)
    grids_params_DM_PhS_df.dropna().reset_index(drop=True)

    grids_params_YM_PhS_df.to_csv(f"../../Results/Grids_Params_Transplant/YM_PS_{rank}.txt", sep='\t', float_format='%.4f', index=True, header=True)
    grids_params_AM_PhS_df.to_csv(f"../../Results/Grids_Params_Transplant/AM_PS_{rank}.txt", sep='\t', float_format='%.4f', index=True, header=True)
    grids_params_DM_PhS_df.to_csv(f"../../Results/Grids_Params_Transplant/DM_PS_{rank}.txt", sep='\t', float_format='%.4f', index=True, header=True)

    comm.Barrier()