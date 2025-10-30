import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import rasterio
import os
import sys
import time
import xarray as xr
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))

from Water_Blance_Model import mYWBMnlS, abcdnlS, DWBMnlS
from Rewrite_Func import nash_sutcliffe_efficiency, relative_error, kling_gupta_efficiency
from numba import float64, njit
from numba.experimental import jitclass
from netCDF4 import Dataset
from scipy.spatial.distance import mahalanobis
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor
from xgboost import XGBRegressor
from scipy.ndimage import median_filter
from concurrent.futures import ThreadPoolExecutor, as_completed
from mpi4py import MPI

# 获取全局编号
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# 读取流域信息
basin_info      = pd.read_excel('../../Data/Basin_Selection/All_Selected_Basins.xlsx')
basin_list      = basin_info['stat_num']
cali_start_list = basin_info['cali_start']
cali_end_list   = basin_info['cali_end']
vali_start_list = basin_info['vali_start']
vali_end_list   = basin_info['vali_end']

# 集总式模型数据读取
def get_data_lumped(basin, basin_idx):
    filepath = f"../../../2025_03_Hydrological_Models/Data/New_Hydro_Climatic/NHC_{basin}.txt"
    hc_data = pd.read_csv(filepath, sep = '\t', header=0, index_col='Time', parse_dates=['Time'])
    cali_start = pd.to_datetime(f"{str(cali_start_list[basin_idx])}-01-01")
    cali_end   = pd.to_datetime(f"{str(cali_end_list[basin_idx])}-12-31")
    vali_start = pd.to_datetime(f"{str(vali_start_list[basin_idx])}-01-01")
    vali_end   = pd.to_datetime(f"{str(vali_end_list[basin_idx])}-12-31")

    cali_data = hc_data.loc[cali_start : cali_end]
    vali_data = hc_data.loc[vali_start : vali_end]

    x_cali = cali_data[['PRE_CRU', 'TMP_CRU', 'PET_CRU']].to_numpy()
    y_cali = cali_data['RUN'].to_numpy()
    x_vali = vali_data[['PRE_CRU', 'TMP_CRU', 'PET_CRU']].to_numpy()
    y_vali = vali_data['RUN'].to_numpy()
    return x_cali, y_cali, x_vali, y_vali

def get_params_by_spatial_proximity(basin_properties, params, N, lon, lat):
    # 提取经纬度
    longitudes = basin_properties[:, 0]
    latitudes = basin_properties[:, 1]

    # 计算欧氏距离
    distances = np.sqrt((longitudes - lon) ** 2 + (latitudes - lat) ** 2)
    
    # 找到距离最近的 N 个流域的索引
    sorted_indices = np.argsort(distances)
    # 如果第一个距离非常小，认为是目标流域本身
    if distances[sorted_indices[0]] <= 1e-5:
        return params[sorted_indices[0]]
    
    # 获取最近的 N 个流域参数
    nearest_params = params[sorted_indices[:N]]
    
    # 返回参数平均值
    return nearest_params.mean(axis=0)

def calculate_similarity(target_properties, basin_properties, method='mahalanobis'):
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

def get_params_by_physical_similarity(basin_properties, params, target_properties, N=5, method='mahalanobis'):
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

def train_random_forest(basin_properties_scaled, params):
    # 初始化并训练模型
    rf_model = MultiOutputRegressor(RandomForestRegressor(n_estimators      = 100, 
                                                          max_depth         = 10,
                                                          min_samples_split = 5,
                                                          min_samples_leaf  = 2,
                                                          random_state      = 42,
                                                          n_jobs            = -1), n_jobs=-1)
    rf_model.fit(basin_properties_scaled, params)
    
    return rf_model

def train_svm(basin_properties_scaled, params):
    # 初始化并训练模型
    svr_model = MultiOutputRegressor(SVR(kernel     = 'rbf',
                                         C          = 100,
                                         epsilon    = 0.01,
                                         gamma      = 0.1), n_jobs=-1)
    svr_model.fit(basin_properties_scaled, params)
    
    return svr_model

def train_xgboost(basin_properties_scaled, params):
    # 初始化并训练模型
    xgb_model = MultiOutputRegressor(XGBRegressor(n_estimators      = 100,
                                                  learning_rate     = 0.1,
                                                  max_depth         = 6,
                                                  min_child_weight  = 3,
                                                  gamma             = 0.1,
                                                  colsample_bytree  = 0.8,
                                                  subsample         = 0.8,
                                                  reg_alpha         = 0.1,
                                                  random_state      = 42,
                                                  n_jobs            = -1), n_jobs=-1)
    xgb_model.fit(basin_properties_scaled, params)
    
    return xgb_model

def get_params_by_regression(target_properties_scaled, trained_models):
    # 使用模型进行预测
    predicted_params = trained_models.predict(target_properties_scaled)
    return np.squeeze(predicted_params)

Basin_Properties = pd.read_csv("../../Data/Properties/Basin_Properties.txt", sep = '\t', header=0, index_col='stat_num')

params_mYWBM   = pd.read_csv("../../Data/Params/03_mYWBM_Best_Params_CF.txt", sep = '\t', header=0, index_col='stat_num')
params_abcd    = pd.read_csv("../../Data/Params/03_abcd_Best_Params_CF.txt", sep = '\t', header=0, index_col='stat_num')
params_DWBM    = pd.read_csv("../../Data/Params/03_DWBM_Best_Params_CF.txt", sep = '\t', header=0, index_col='stat_num')

# mYWBM模型结果数组
cali_kge_YM_list = np.full((len(basin_list), 6), np.nan)
vali_kge_YM_list = np.full((len(basin_list), 6), np.nan)
cali_re_YM_list  = np.full((len(basin_list), 6), np.nan)
vali_re_YM_list  = np.full((len(basin_list), 6), np.nan)
# abcd模型结果数组
cali_kge_AM_list = np.full((len(basin_list), 6), np.nan)
vali_kge_AM_list = np.full((len(basin_list), 6), np.nan)
cali_re_AM_list  = np.full((len(basin_list), 6), np.nan)
vali_re_AM_list  = np.full((len(basin_list), 6), np.nan)
# DWBM模型结果数组
cali_kge_DM_list = np.full((len(basin_list), 6), np.nan)
vali_kge_DM_list = np.full((len(basin_list), 6), np.nan)
cali_re_DM_list  = np.full((len(basin_list), 6), np.nan)
vali_re_DM_list  = np.full((len(basin_list), 6), np.nan)

params_YM_spatial_proximity     = np.full((len(basin_list), 5), np.nan)
params_YM_physical_similarity   = np.full((len(basin_list), 5), np.nan)
params_YM_rf                    = np.full((len(basin_list), 5), np.nan)
params_YM_svm                   = np.full((len(basin_list), 5), np.nan)
params_YM_xgb                   = np.full((len(basin_list), 5), np.nan)

params_AM_spatial_proximity    = np.full((len(basin_list), 5), np.nan)
params_AM_physical_similarity  = np.full((len(basin_list), 5), np.nan)
params_AM_rf                   = np.full((len(basin_list), 5), np.nan)
params_AM_svm                  = np.full((len(basin_list), 5), np.nan)
params_AM_xgb                  = np.full((len(basin_list), 5), np.nan)

params_DM_spatial_proximity     = np.full((len(basin_list), 5), np.nan)
params_DM_physical_similarity   = np.full((len(basin_list), 5), np.nan)
params_DM_rf                    = np.full((len(basin_list), 5), np.nan)
params_DM_svm                   = np.full((len(basin_list), 5), np.nan)
params_DM_xgb                   = np.full((len(basin_list), 5), np.nan)

def get_params(basin, Basin_Properties, params_mYWBM):
    # 获取该流域的经纬度
    basin_lon = Basin_Properties.loc[basin]['Longitude']
    basin_lat = Basin_Properties.loc[basin]['Latitude']
    # 删除该流域的经纬度
    Basin_Properties = Basin_Properties.drop(columns=['Longitude', 'Latitude'])
    # 获取率定的参数
    cali_params = params_mYWBM.loc[basin].to_numpy()
    # 获取流域属性
    basin_properties = Basin_Properties.loc[basin].to_numpy().reshape(1, -1)
    # 获取该流域剩余流域的属性
    rest_properties = Basin_Properties.copy().drop(index=basin).to_numpy()
    # 获取该流域剩余流域的参数
    rest_params = params_mYWBM.copy().drop(index=basin).to_numpy()
    return cali_params, basin_properties, rest_properties, rest_params, basin_lon, basin_lat
def cal_metrics(cali_obs, vali_obs, cali_sim, vali_sim):
    # 计算率定期和验证期的NSE和RE
    cali_nse = kling_gupta_efficiency(cali_obs, cali_sim)
    vali_nse = kling_gupta_efficiency(vali_obs, vali_sim)
    cali_re  = relative_error(cali_obs, cali_sim) * 100
    vali_re  = relative_error(vali_obs, vali_sim) * 100
    return cali_nse, vali_nse, cali_re, vali_re
def clean_params(params, all_params, lower_bound, upper_bound):
    r, c = params.shape
    for i in range(r):
        for j in range(c):
            if params[i, j] < lower_bound[j] or params[i, j] > upper_bound[j]:
                params[i, j] = np.nanmean(all_params[:, j])
    return params

def process_basin(basin_idx):
    st = time.time()
    basin = str(basin_list[basin_idx])
    
    ## mYWBM
    cali_params, basin_properties, rest_properties, rest_params, basin_lon, basin_lat = get_params(
        basin, Basin_Properties[['Climate', 'Clay', 'Silt', 'Sand', 'Slope', 'BFI', 'PRE', 'TMP', 'PET', 'TMAX', 'TMIN', 'AE', 'NDVI', 'TI', 'Longitude', 'Latitude']], params_mYWBM)

    scaler = StandardScaler()
    source_properties_scaled = scaler.fit_transform(rest_properties)
    target_properties_scaled = scaler.transform(basin_properties)

    rf_model  = train_random_forest(source_properties_scaled, rest_params)
    svm_model = train_svm(source_properties_scaled, rest_params)
    xgb_model = train_xgboost(source_properties_scaled, rest_params)

    pred_params_spatial_proximity = get_params_by_spatial_proximity(rest_properties, rest_params, 50, basin_lon, basin_lat)
    pred_params_physical_similarity = get_params_by_physical_similarity(rest_properties, rest_params, basin_properties, N=5, method='mahalanobis')
    pred_params_rf  = clean_params(get_params_by_regression(target_properties_scaled, rf_model).reshape(1, -1), rest_params, [0, 0, 0.05, 100, 0], [2, 0.65, 0.95, 1000, 1])
    pred_params_svr = clean_params(get_params_by_regression(target_properties_scaled, svm_model).reshape(1, -1), rest_params, [0, 0, 0.05, 100, 0], [2, 0.65, 0.95, 1000, 1])
    pred_params_xgb = clean_params(get_params_by_regression(target_properties_scaled, xgb_model).reshape(1, -1), rest_params, [0, 0, 0.05, 100, 0], [2, 0.65, 0.95, 1000, 1])

    params_YM_spatial_proximity[basin_idx]     = pred_params_spatial_proximity
    params_YM_physical_similarity[basin_idx]   = pred_params_physical_similarity
    params_YM_rf[basin_idx]                    = np.squeeze(pred_params_rf)
    params_YM_svm[basin_idx]                   = np.squeeze(pred_params_svr)
    params_YM_xgb[basin_idx]                   = np.squeeze(pred_params_xgb)

    ## abcd
    cali_params, basin_properties, rest_properties, rest_params, basin_lon, basin_lat = get_params(
        basin, Basin_Properties[['Climate', 'Clay', 'Silt', 'Sand', 'Slope', 'BFI', 'PRE', 'TMP', 'PET', 'TMAX', 'TMIN', 'AE', 'NDVI', 'TI', 'Longitude', 'Latitude']], params_abcd)

    scaler = StandardScaler()
    source_properties_scaled = scaler.fit_transform(rest_properties)
    target_properties_scaled = scaler.transform(basin_properties)

    rf_model  = train_random_forest(source_properties_scaled, rest_params)
    svm_model = train_svm(source_properties_scaled, rest_params)
    xgb_model = train_xgboost(source_properties_scaled, rest_params)

    pred_params_spatial_proximity = get_params_by_spatial_proximity(rest_properties, rest_params, 50, basin_lon, basin_lat)
    pred_params_physical_similarity = get_params_by_physical_similarity(rest_properties, rest_params, basin_properties, N=5, method='mahalanobis')
    pred_params_rf  = clean_params(get_params_by_regression(target_properties_scaled, rf_model).reshape(1, -1), rest_params, [0, 100, 0, 0, 0], [1, 2000, 1, 1, 1])
    pred_params_svr = clean_params(get_params_by_regression(target_properties_scaled, svm_model).reshape(1, -1), rest_params, [0, 100, 0, 0, 0], [1, 2000, 1, 1, 1])
    pred_params_xgb = clean_params(get_params_by_regression(target_properties_scaled, xgb_model).reshape(1, -1), rest_params, [0, 100, 0, 0, 0], [1, 2000, 1, 1, 1])

    params_AM_spatial_proximity[basin_idx]     = pred_params_spatial_proximity
    params_AM_physical_similarity[basin_idx]   = pred_params_physical_similarity
    params_AM_rf[basin_idx]                    = np.squeeze(pred_params_rf)
    params_AM_svm[basin_idx]                   = np.squeeze(pred_params_svr)
    params_AM_xgb[basin_idx]                   = np.squeeze(pred_params_xgb)

    ## DWBM模型
    cali_params, basin_properties, rest_properties, rest_params, basin_lon, basin_lat = get_params(
        basin, Basin_Properties[['Climate', 'Clay', 'Silt', 'Sand', 'Slope', 'BFI', 'PRE', 'TMP', 'PET', 'TMAX', 'TMIN', 'AE', 'NDVI', 'TI', 'Longitude', 'Latitude']], params_DWBM)

    scaler = StandardScaler()
    source_properties_scaled = scaler.fit_transform(rest_properties)
    target_properties_scaled = scaler.transform(basin_properties)

    rf_model  = train_random_forest(source_properties_scaled, rest_params)
    svm_model = train_svm(source_properties_scaled, rest_params)
    xgb_model = train_xgboost(source_properties_scaled, rest_params)

    pred_params_spatial_proximity = get_params_by_spatial_proximity(rest_properties, rest_params, 50, basin_lon, basin_lat)
    pred_params_physical_similarity = get_params_by_physical_similarity(rest_properties, rest_params, basin_properties, N=5, method='mahalanobis')
    pred_params_rf  = clean_params(get_params_by_regression(target_properties_scaled, rf_model).reshape(1, -1), rest_params, [0, 0, 100, 0, 0], [1, 1, 2000, 1, 1])
    pred_params_svr = clean_params(get_params_by_regression(target_properties_scaled, svm_model).reshape(1, -1), rest_params, [0, 0, 100, 0, 0], [1, 1, 2000, 1, 1])
    pred_params_xgb = clean_params(get_params_by_regression(target_properties_scaled, xgb_model).reshape(1, -1), rest_params, [0, 0, 100, 0, 0], [1, 1, 2000, 1, 1])

    params_DM_spatial_proximity[basin_idx]     = pred_params_spatial_proximity
    params_DM_physical_similarity[basin_idx]   = pred_params_physical_similarity
    params_DM_rf[basin_idx]                    = np.squeeze(pred_params_rf)
    params_DM_svm[basin_idx]                   = np.squeeze(pred_params_svr)
    params_DM_xgb[basin_idx]                   = np.squeeze(pred_params_xgb)

    et = time.time()
    print(f"No. {basin_idx+1} 流域 {basin} 参数获取完成，耗时 {et - st:.4f} 秒")

max_workers = 14

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

idx_list = regroup(np.arange(len(basin_list)), 40)
start_list = idx_list[0, :]
end_list = idx_list[1, :]

if __name__ == "__main__":
    params_cali = np.full((len(basin_list), 6), np.nan)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_basin, i) for i in range(start_list[rank], end_list[rank])]
        for future in as_completed(futures):
            future.result()

    params_YM_spatial_proximity_df    = pd.DataFrame(params_YM_spatial_proximity, index=params_mYWBM.index, columns=params_mYWBM.columns)
    params_YM_physical_similarity_df  = pd.DataFrame(params_YM_physical_similarity, index=params_mYWBM.index, columns=params_mYWBM.columns)
    params_YM_rf_df                   = pd.DataFrame(params_YM_rf, index=params_mYWBM.index, columns=params_mYWBM.columns)
    params_YM_svm_df                  = pd.DataFrame(params_YM_svm, index=params_mYWBM.index, columns=params_mYWBM.columns)
    params_YM_xgb_df                  = pd.DataFrame(params_YM_xgb, index=params_mYWBM.index, columns=params_mYWBM.columns)

    params_AM_spatial_proximity_df    = pd.DataFrame(params_AM_spatial_proximity, index=params_abcd.index, columns=params_abcd.columns)
    params_AM_physical_similarity_df  = pd.DataFrame(params_AM_physical_similarity, index=params_abcd.index, columns=params_abcd.columns)
    params_AM_rf_df                   = pd.DataFrame(params_AM_rf, index=params_abcd.index, columns=params_abcd.columns)
    params_AM_svm_df                  = pd.DataFrame(params_AM_svm, index=params_abcd.index, columns=params_abcd.columns)
    params_AM_xgb_df                  = pd.DataFrame(params_AM_xgb, index=params_abcd.index, columns=params_abcd.columns)

    params_DM_spatial_proximity_df    = pd.DataFrame(params_DM_spatial_proximity, index=params_DWBM.index, columns=params_DWBM.columns)
    params_DM_physical_similarity_df  = pd.DataFrame(params_DM_physical_similarity, index=params_DWBM.index, columns=params_DWBM.columns)
    params_DM_rf_df                   = pd.DataFrame(params_DM_rf, index=params_DWBM.index, columns=params_DWBM.columns)
    params_DM_svm_df                  = pd.DataFrame(params_DM_svm, index=params_DWBM.index, columns=params_DWBM.columns)
    params_DM_xgb_df                  = pd.DataFrame(params_DM_xgb, index=params_DWBM.index, columns=params_DWBM.columns)

    params_YM_spatial_proximity_df.to_csv("../../Results/Params_Transplant/mYWBM_Spatial_Proximity_Params_{rank}.txt", sep = '\t', float_format='%.4f')
    params_YM_physical_similarity_df.to_csv("../../Results/Params_Transplant/mYWBM_Physical_Similarity_Params_{rank}.txt", sep = '\t', float_format='%.4f')
    params_YM_rf_df.to_csv("../../Results/Params_Transplant/mYWBM_RF_Params_{rank}.txt", sep = '\t', float_format='%.4f')
    params_YM_svm_df.to_csv("../../Results/Params_Transplant/mYWBM_SVM_Params_{rank}.txt", sep = '\t', float_format='%.4f')
    params_YM_xgb_df.to_csv("../../Results/Params_Transplant/mYWBM_XGB_Params_{rank}.txt", sep = '\t', float_format='%.4f')

    params_AM_spatial_proximity_df.to_csv("../../Results/Params_Transplant/abcd_Spatial_Proximity_Params_{rank}.txt", sep = '\t', float_format='%.4f')
    params_AM_physical_similarity_df.to_csv("../../Results/Params_Transplant/abcd_Physical_Similarity_Params_{rank}.txt", sep = '\t', float_format='%.4f')
    params_AM_rf_df.to_csv("../../Results/Params_Transplant/abcd_RF_Params_{rank}.txt", sep = '\t', float_format='%.4f')
    params_AM_svm_df.to_csv("../../Results/Params_Transplant/abcd_SVM_Params_{rank}.txt", sep = '\t', float_format='%.4f')
    params_AM_xgb_df.to_csv("../../Results/Params_Transplant/abcd_XGB_Params_{rank}.txt", sep = '\t', float_format='%.4f')

    params_DM_spatial_proximity_df.to_csv("../../Results/Params_Transplant/DWBM_Spatial_Proximity_Params_{rank}.txt", sep = '\t', float_format='%.4f')
    params_DM_physical_similarity_df.to_csv("../../Results/Params_Transplant/DWBM_Physical_Similarity_Params_{rank}.txt", sep = '\t', float_format='%.4f')
    params_DM_rf_df.to_csv("../../Results/Params_Transplant/DWBM_RF_Params_{rank}.txt", sep = '\t', float_format='%.4f')
    params_DM_svm_df.to_csv("../../Results/Params_Transplant/DWBM_SVM_Params_{rank}.txt", sep = '\t', float_format='%.4f')
    params_DM_xgb_df.to_csv("../../Results/Params_Transplant/DWBM_XGB_Params_{rank}.txt", sep = '\t', float_format='%.4f')
