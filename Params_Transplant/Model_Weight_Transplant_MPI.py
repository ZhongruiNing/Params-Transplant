import numpy as np
import pandas as pd
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor
from xgboost import XGBRegressor
from scipy.ndimage import median_filter
from mpi4py import MPI
from concurrent.futures import ThreadPoolExecutor, as_completed

# 获取全局编号
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# 读取流域信息
basin_info      = pd.read_excel('../../Data/Basin_Selection/All_Selected_Basins.xlsx')
basin_list      = basin_info['stat_num'].astype(str)
cali_start_list = basin_info['cali_start']
cali_end_list   = basin_info['cali_end']
vali_start_list = basin_info['vali_start']
vali_end_list   = basin_info['vali_end']

Basin_Properties = pd.read_csv("../../Data/Properties/Basin_Properties.txt", sep = '\t', header=0, index_col='stat_num')
source_properties = Basin_Properties[['Climate', 'Clay', 'Silt', 'Sand', 'Slope', 'BFI', 'PRE', 'TMP', 'PET', 'TMAX', 'TMIN', 'AE', 'NDVI', 'TI']].values

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

sim_results = pd.read_csv("../../Results/Weighted_Average/Weighted_Average_Results.txt", sep="\t", index_col='stat_num')[['w_ic_YM', 'w_ic_AM', 'w_ic_DM']].values

def process_basin(b):
    basin = basin_list[b]
    print(f"Processing basin {basin} ({b+1}/{len(basin_list)})")

    X_train = np.vstack([source_properties[:b], source_properties[b+1:]])
    y_train = np.vstack([sim_results[:b], sim_results[b+1:]])

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # 获取当前流域属性
    target_properties = Basin_Properties.loc[basin, ['Climate', 'Clay', 'Silt', 'Sand', 'Slope', 'BFI', 'PRE', 'TMP', 'PET', 'TMAX', 'TMIN', 'AE', 'NDVI', 'TI']].values.reshape(1, -1)
    target_properties_scaled = scaler.transform(target_properties)

    # 训练并预测随机森林模型
    rf_model  = train_random_forest(X_train_scaled, y_train)
    svr_model = train_svm(X_train_scaled, y_train)
    xgb_model = train_xgboost(X_train_scaled, y_train)

    rf_pred  = rf_model.predict(target_properties_scaled)[0]
    svr_pred = svr_model.predict(target_properties_scaled)[0]
    xgb_pred = xgb_model.predict(target_properties_scaled)[0]

    # 归一化，确保和为1
    rf_pred  = np.clip(rf_pred, 0, None)
    svr_pred = np.clip(svr_pred, 0, None)
    xgb_pred = np.clip(xgb_pred, 0, None)
    rf_pred  = rf_pred / np.sum(rf_pred)
    svr_pred = svr_pred / np.sum(svr_pred)
    xgb_pred = xgb_pred / np.sum(xgb_pred)

    pred_weight_rf.loc[basin]  = rf_pred
    pred_weight_svr.loc[basin] = svr_pred
    pred_weight_xgb.loc[basin] = xgb_pred

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
    pred_weight_rf  = pd.DataFrame(index=basin_list, columns=['w_ic_YM', 'w_ic_AM', 'w_ic_DM'])
    pred_weight_svr = pd.DataFrame(index=basin_list, columns=['w_ic_YM', 'w_ic_AM', 'w_ic_DM'])
    pred_weight_xgb = pd.DataFrame(index=basin_list, columns=['w_ic_YM', 'w_ic_AM', 'w_ic_DM'])

    pred_weight_rf.index.name  = 'stat_num'
    pred_weight_svr.index.name = 'stat_num'
    pred_weight_xgb.index.name = 'stat_num'

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_basin, i) for i in range(start_list[rank], end_list[rank])]
        for future in as_completed(futures):
            future.result()

    pred_weight_rf.to_csv(f"../../Results/Model_Weight_Transplant/Model_Weight_RF_Part{rank+1}.txt", sep="\t", float_format='%.4f')
    pred_weight_svr.to_csv(f"../../Results/Model_Weight_Transplant/Model_Weight_SVR_Part{rank+1}.txt", sep="\t", float_format='%.4f')
    pred_weight_xgb.to_csv(f"../../Results/Model_Weight_Transplant/Model_Weight_XGB_Part{rank+1}.txt", sep="\t", float_format='%.4f')

    comm.Barrier()
