import numpy as np
import pandas as pd
import os

params_YM_spatial_proximity_df = pd.read_csv("../../Results/Params_Transplant/mYWBM_Spatial_Proximity_Params_0.txt", sep = '\t', header=0, index_col='stat_num')
params_YM_physical_similarity_df = pd.read_csv("../../Results/Params_Transplant/mYWBM_Physical_Similarity_Params_0.txt", sep = '\t', header=0, index_col='stat_num')
params_YM_rf_df = pd.read_csv("../../Results/Params_Transplant/mYWBM_RF_Params_0.txt", sep = '\t', header=0, index_col='stat_num')
params_YM_svm_df = pd.read_csv("../../Results/Params_Transplant/mYWBM_SVM_Params_0.txt", sep = '\t', header=0, index_col='stat_num')
params_YM_xgb_df = pd.read_csv("../../Results/Params_Transplant/mYWBM_XGB_Params_0.txt", sep = '\t', header=0, index_col='stat_num')

params_AM_spatial_proximity_df = pd.read_csv("../../Results/Params_Transplant/abcd_Spatial_Proximity_Params_0.txt", sep = '\t', header=0, index_col='stat_num')
params_AM_physical_similarity_df = pd.read_csv("../../Results/Params_Transplant/abcd_Physical_Similarity_Params_0.txt", sep = '\t', header=0, index_col='stat_num')
params_AM_rf_df = pd.read_csv("../../Results/Params_Transplant/abcd_RF_Params_0.txt", sep = '\t', header=0, index_col='stat_num')
params_AM_svm_df = pd.read_csv("../../Results/Params_Transplant/abcd_SVM_Params_0.txt", sep = '\t', header=0, index_col='stat_num')
params_AM_xgb_df = pd.read_csv("../../Results/Params_Transplant/abcd_XGB_Params_0.txt", sep = '\t', header=0, index_col='stat_num')

params_DM_spatial_proximity_df = pd.read_csv("../../Results/Params_Transplant/DWBM_Spatial_Proximity_Params_0.txt", sep = '\t', header=0, index_col='stat_num')
params_DM_physical_similarity_df = pd.read_csv("../../Results/Params_Transplant/DWBM_Physical_Similarity_Params_0.txt", sep = '\t', header=0, index_col='stat_num')
params_DM_rf_df = pd.read_csv("../../Results/Params_Transplant/DWBM_RF_Params_0.txt", sep = '\t', header=0, index_col='stat_num')
params_DM_svm_df = pd.read_csv("../../Results/Params_Transplant/DWBM_SVM_Params_0.txt", sep = '\t', header=0, index_col='stat_num')
params_DM_xgb_df = pd.read_csv("../../Results/Params_Transplant/DWBM_XGB_Params_0.txt", sep = '\t', header=0, index_col='stat_num')

for i in range(1, 40):
    temp_params_YM_spatial_proximity_df = pd.read_csv(f"../../Results/Params_Transplant/mYWBM_Spatial_Proximity_Params_{i}.txt", sep = '\t', header=0, index_col='stat_num')
    temp_params_YM_physical_similarity_df = pd.read_csv(f"../../Results/Params_Transplant/mYWBM_Physical_Similarity_Params_{i}.txt", sep = '\t', header=0, index_col='stat_num')
    temp_params_YM_rf_df = pd.read_csv(f"../../Results/Params_Transplant/mYWBM_RF_Params_{i}.txt", sep = '\t', header=0, index_col='stat_num')
    temp_params_YM_svm_df = pd.read_csv(f"../../Results/Params_Transplant/mYWBM_SVM_Params_{i}.txt", sep = '\t', header=0, index_col='stat_num')
    temp_params_YM_xgb_df = pd.read_csv(f"../../Results/Params_Transplant/mYWBM_XGB_Params_{i}.txt", sep = '\t', header=0, index_col='stat_num')
  
    temp_params_AM_spatial_proximity_df = pd.read_csv(f"../../Results/Params_Transplant/abcd_Spatial_Proximity_Params_{i}.txt", sep = '\t', header=0, index_col='stat_num')
    temp_params_AM_physical_similarity_df = pd.read_csv(f"../../Results/Params_Transplant/abcd_Physical_Similarity_Params_{i}.txt", sep = '\t', header=0, index_col='stat_num')
    temp_params_AM_rf_df = pd.read_csv(f"../../Results/Params_Transplant/abcd_RF_Params_{i}.txt", sep = '\t', header=0, index_col='stat_num')
    temp_params_AM_svm_df = pd.read_csv(f"../../Results/Params_Transplant/abcd_SVM_Params_{i}.txt", sep = '\t', header=0, index_col='stat_num')
    temp_params_AM_xgb_df = pd.read_csv(f"../../Results/Params_Transplant/abcd_XGB_Params_{i}.txt", sep = '\t', header=0, index_col='stat_num')

    temp_params_DM_spatial_proximity_df = pd.read_csv(f"../../Results/Params_Transplant/DWBM_Spatial_Proximity_Params_{i}.txt", sep = '\t', header=0, index_col='stat_num')
    temp_params_DM_physical_similarity_df = pd.read_csv(f"../../Results/Params_Transplant/DWBM_Physical_Similarity_Params_{i}.txt", sep = '\t', header=0, index_col='stat_num')
    temp_params_DM_rf_df = pd.read_csv(f"../../Results/Params_Transplant/DWBM_RF_Params_{i}.txt", sep = '\t', header=0, index_col='stat_num')
    temp_params_DM_svm_df = pd.read_csv(f"../../Results/Params_Transplant/DWBM_SVM_Params_{i}.txt", sep = '\t', header=0, index_col='stat_num')
    temp_params_DM_xgb_df = pd.read_csv(f"../../Results/Params_Transplant/DWBM_XGB_Params_{i}.txt", sep = '\t', header=0, index_col='stat_num')

    not_nan_idx = temp_params_YM_spatial_proximity_df.dropna(how='all').index

    params_YM_spatial_proximity_df.loc[not_nan_idx] = temp_params_YM_spatial_proximity_df.loc[not_nan_idx]
    params_YM_physical_similarity_df.loc[not_nan_idx] = temp_params_YM_physical_similarity_df.loc[not_nan_idx]
    params_YM_rf_df.loc[not_nan_idx] = temp_params_YM_rf_df.loc[not_nan_idx]
    params_YM_svm_df.loc[not_nan_idx] = temp_params_YM_svm_df.loc[not_nan_idx]
    params_YM_xgb_df.loc[not_nan_idx] = temp_params_YM_xgb_df.loc[not_nan_idx]

    params_AM_spatial_proximity_df.loc[not_nan_idx] = temp_params_AM_spatial_proximity_df.loc[not_nan_idx]
    params_AM_physical_similarity_df.loc[not_nan_idx] = temp_params_AM_physical_similarity_df.loc[not_nan_idx]
    params_AM_rf_df.loc[not_nan_idx] = temp_params_AM_rf_df.loc[not_nan_idx]
    params_AM_svm_df.loc[not_nan_idx] = temp_params_AM_svm_df.loc[not_nan_idx]
    params_AM_xgb_df.loc[not_nan_idx] = temp_params_AM_xgb_df.loc[not_nan_idx]

    params_DM_spatial_proximity_df.loc[not_nan_idx] = temp_params_DM_spatial_proximity_df.loc[not_nan_idx]
    params_DM_physical_similarity_df.loc[not_nan_idx] = temp_params_DM_physical_similarity_df.loc[not_nan_idx]
    params_DM_rf_df.loc[not_nan_idx] = temp_params_DM_rf_df.loc[not_nan_idx]
    params_DM_svm_df.loc[not_nan_idx] = temp_params_DM_svm_df.loc[not_nan_idx]
    params_DM_xgb_df.loc[not_nan_idx] = temp_params_DM_xgb_df.loc[not_nan_idx]

params_YM_spatial_proximity_df.to_csv("../../Results/Params_Transplant/mYWBM_Spatial_Proximity_Params.txt", sep = '\t', float_format='%.4f')
params_YM_physical_similarity_df.to_csv("../../Results/Params_Transplant/mYWBM_Physical_Similarity_Params.txt", sep = '\t', float_format='%.4f')
params_YM_rf_df.to_csv("../../Results/Params_Transplant/mYWBM_RF_Params.txt", sep = '\t', float_format='%.4f')
params_YM_svm_df.to_csv("../../Results/Params_Transplant/mYWBM_SVM_Params.txt", sep = '\t', float_format='%.4f')
params_YM_xgb_df.to_csv("../../Results/Params_Transplant/mYWBM_XGB_Params.txt", sep = '\t', float_format='%.4f')

params_AM_spatial_proximity_df.to_csv("../../Results/Params_Transplant/abcd_Spatial_Proximity_Params.txt", sep = '\t', float_format='%.4f')
params_AM_physical_similarity_df.to_csv("../../Results/Params_Transplant/abcd_Physical_Similarity_Params.txt", sep = '\t', float_format='%.4f')
params_AM_rf_df.to_csv("../../Results/Params_Transplant/abcd_RF_Params.txt", sep = '\t', float_format='%.4f')
params_AM_svm_df.to_csv("../../Results/Params_Transplant/abcd_SVM_Params.txt", sep = '\t', float_format='%.4f')
params_AM_xgb_df.to_csv("../../Results/Params_Transplant/abcd_XGB_Params.txt", sep = '\t', float_format='%.4f')

params_DM_spatial_proximity_df.to_csv("../../Results/Params_Transplant/DWBM_Spatial_Proximity_Params.txt", sep = '\t', float_format='%.4f')
params_DM_physical_similarity_df.to_csv("../../Results/Params_Transplant/DWBM_Physical_Similarity_Params.txt", sep = '\t', float_format='%.4f')
params_DM_rf_df.to_csv("../../Results/Params_Transplant/DWBM_RF_Params.txt", sep = '\t', float_format='%.4f')
params_DM_svm_df.to_csv("../../Results/Params_Transplant/DWBM_SVM_Params.txt", sep = '\t', float_format='%.4f')
params_DM_xgb_df.to_csv("../../Results/Params_Transplant/DWBM_XGB_Params.txt", sep = '\t', float_format='%.4f')

# for i in range(40):
#     for prefix in ["mYWBM", "abcd", "DWBM"]:
#         for method in ["Spatial_Proximity", "Physical_Similarity", "RF", "SVM", "XGB"]:
#             file_path = f"../../Results/Params_Transplant/{prefix}_{method}_Params_{i}.txt"
#             if os.path.exists(file_path):
#                 os.remove(file_path)