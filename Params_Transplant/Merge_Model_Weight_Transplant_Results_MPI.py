import numpy as np
import pandas as pd
import os

pred_weight_rf  = pd.read_csv(f"../../Results/Model_Weight_Transplant/E_Model_Weight_GRC_RF_Part0.txt", sep='\t', header=0, index_col='stat_num')
pred_weight_svr = pd.read_csv(f"../../Results/Model_Weight_Transplant/E_Model_Weight_GRC_SVR_Part0.txt", sep='\t', header=0, index_col='stat_num')
pred_weight_xgb = pd.read_csv(f"../../Results/Model_Weight_Transplant/E_Model_Weight_GRC_XGB_Part0.txt", sep='\t', header=0, index_col='stat_num')

for i in range(1, 40):
    temp_pred_weight_rf  = pd.read_csv(f"../../Results/Model_Weight_Transplant/E_Model_Weight_GRC_RF_Part{i}.txt", sep='\t', header=0, index_col='stat_num')
    temp_pred_weight_svr = pd.read_csv(f"../../Results/Model_Weight_Transplant/E_Model_Weight_GRC_SVR_Part{i}.txt", sep='\t', header=0, index_col='stat_num')
    temp_pred_weight_xgb = pd.read_csv(f"../../Results/Model_Weight_Transplant/E_Model_Weight_GRC_XGB_Part{i}.txt", sep='\t', header=0, index_col='stat_num')

    not_nan_idx = temp_pred_weight_rf.dropna(how='all').index
    pred_weight_rf.loc[not_nan_idx] = temp_pred_weight_rf.loc[not_nan_idx]
    not_nan_idx = temp_pred_weight_svr.dropna(how='all').index
    pred_weight_svr.loc[not_nan_idx] = temp_pred_weight_svr.loc[not_nan_idx]
    not_nan_idx = temp_pred_weight_xgb.dropna(how='all').index
    pred_weight_xgb.loc[not_nan_idx] = temp_pred_weight_xgb.loc[not_nan_idx]

pred_weight_rf.to_csv(f"../../Results/Model_Weight_Transplant/E_pred_weight_GRC_rf.txt", sep='\t', float_format='%.4f')
pred_weight_svr.to_csv(f"../../Results/Model_Weight_Transplant/E_pred_weight_GRC_svr.txt", sep='\t', float_format='%.4f')
pred_weight_xgb.to_csv(f"../../Results/Model_Weight_Transplant/E_pred_weight_GRC_xgb.txt", sep='\t', float_format='%.4f')

for i in range(0, 40):
    os.remove(f"../../Results/Model_Weight_Transplant/E_Model_Weight_GRC_RF_Part{i}.txt")
    os.remove(f"../../Results/Model_Weight_Transplant/E_Model_Weight_GRC_SVR_Part{i}.txt")
    os.remove(f"../../Results/Model_Weight_Transplant/E_Model_Weight_GRC_XGB_Part{i}.txt")