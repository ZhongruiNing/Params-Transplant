import numpy as np
import pandas as pd
import os

params_YM_SP_AM_df  = pd.read_csv("../../Results/Params_Transplant/mYWBM_SP_AM_Params_0.txt", sep = '\t', header=0, index_col='stat_num')
params_YM_SP_IDW_df = pd.read_csv("../../Results/Params_Transplant/mYWBM_SP_IDW_Params_0.txt", sep = '\t', header=0, index_col='stat_num')
params_YM_PS_df     = pd.read_csv("../../Results/Params_Transplant/mYWBM_PS_Params_0.txt", sep = '\t', header=0, index_col='stat_num')
params_YM_rf_df     = pd.read_csv("../../Results/Params_Transplant/mYWBM_RF_Params_0.txt", sep = '\t', header=0, index_col='stat_num')
params_YM_svm_df    = pd.read_csv("../../Results/Params_Transplant/mYWBM_SVM_Params_0.txt", sep = '\t', header=0, index_col='stat_num')
params_YM_xgb_df    = pd.read_csv("../../Results/Params_Transplant/mYWBM_XGB_Params_0.txt", sep = '\t', header=0, index_col='stat_num')

params_AM_SP_AM_df  = pd.read_csv("../../Results/Params_Transplant/abcd_SP_AM_Params_0.txt", sep = '\t', header=0, index_col='stat_num')
params_AM_SP_IDW_df = pd.read_csv("../../Results/Params_Transplant/abcd_SP_IDW_Params_0.txt", sep = '\t', header=0, index_col='stat_num')
params_AM_PS_df     = pd.read_csv("../../Results/Params_Transplant/abcd_PS_Params_0.txt", sep = '\t', header=0, index_col='stat_num')
params_AM_rf_df     = pd.read_csv("../../Results/Params_Transplant/abcd_RF_Params_0.txt", sep = '\t', header=0, index_col='stat_num')
params_AM_svm_df    = pd.read_csv("../../Results/Params_Transplant/abcd_SVM_Params_0.txt", sep = '\t', header=0, index_col='stat_num')
params_AM_xgb_df    = pd.read_csv("../../Results/Params_Transplant/abcd_XGB_Params_0.txt", sep = '\t', header=0, index_col='stat_num')

params_DM_SP_AM_df  = pd.read_csv("../../Results/Params_Transplant/DWBM_SP_AM_Params_0.txt", sep = '\t', header=0, index_col='stat_num')
params_DM_SP_IDW_df = pd.read_csv("../../Results/Params_Transplant/DWBM_SP_IDW_Params_0.txt", sep = '\t', header=0, index_col='stat_num')
params_DM_PS_df     = pd.read_csv("../../Results/Params_Transplant/DWBM_PS_Params_0.txt", sep = '\t', header=0, index_col='stat_num')
params_DM_rf_df     = pd.read_csv("../../Results/Params_Transplant/DWBM_RF_Params_0.txt", sep = '\t', header=0, index_col='stat_num')
params_DM_svm_df    = pd.read_csv("../../Results/Params_Transplant/DWBM_SVM_Params_0.txt", sep = '\t', header=0, index_col='stat_num')
params_DM_xgb_df    = pd.read_csv("../../Results/Params_Transplant/DWBM_XGB_Params_0.txt", sep = '\t', header=0, index_col='stat_num')

params_GYM_SP_AM_df     = pd.read_csv("../../Results/Params_Transplant/GmYWBM_SP_AM_Params_0.txt", sep = '\t', header=0, index_col='stat_num')
params_GYM_SP_IDW_df    = pd.read_csv("../../Results/Params_Transplant/GmYWBM_SP_IDW_Params_0.txt", sep = '\t', header=0, index_col='stat_num')
params_GYM_PS_df        = pd.read_csv("../../Results/Params_Transplant/GmYWBM_PS_Params_0.txt", sep = '\t', header=0, index_col='stat_num')
params_GYM_rf_df        = pd.read_csv("../../Results/Params_Transplant/GmYWBM_RF_Params_0.txt", sep = '\t', header=0, index_col='stat_num')
params_GYM_svm_df       = pd.read_csv("../../Results/Params_Transplant/GmYWBM_SVM_Params_0.txt", sep = '\t', header=0, index_col='stat_num')
params_GYM_xgb_df       = pd.read_csv("../../Results/Params_Transplant/GmYWBM_XGB_Params_0.txt", sep = '\t', header=0, index_col='stat_num')

for i in range(1, 40):
    temp_params_YM_SP_AM_df     = pd.read_csv(f"../../Results/Params_Transplant/mYWBM_SP_AM_Params_{i}.txt", sep = '\t', header=0, index_col='stat_num')
    temp_params_YM_SP_IDW_df    = pd.read_csv(f"../../Results/Params_Transplant/mYWBM_SP_IDW_Params_{i}.txt", sep = '\t', header=0, index_col='stat_num')
    temp_params_YM_PS_df        = pd.read_csv(f"../../Results/Params_Transplant/mYWBM_PS_Params_{i}.txt", sep = '\t', header=0, index_col='stat_num')
    temp_params_YM_rf_df        = pd.read_csv(f"../../Results/Params_Transplant/mYWBM_RF_Params_{i}.txt", sep = '\t', header=0, index_col='stat_num')
    temp_params_YM_svm_df       = pd.read_csv(f"../../Results/Params_Transplant/mYWBM_SVM_Params_{i}.txt", sep = '\t', header=0, index_col='stat_num')
    temp_params_YM_xgb_df       = pd.read_csv(f"../../Results/Params_Transplant/mYWBM_XGB_Params_{i}.txt", sep = '\t', header=0, index_col='stat_num')
  
    temp_params_AM_SP_AM_df     = pd.read_csv(f"../../Results/Params_Transplant/abcd_SP_AM_Params_{i}.txt", sep = '\t', header=0, index_col='stat_num')
    temp_params_AM_SP_IDW_df    = pd.read_csv(f"../../Results/Params_Transplant/abcd_SP_IDW_Params_{i}.txt", sep = '\t', header=0, index_col='stat_num')
    temp_params_AM_PS_df        = pd.read_csv(f"../../Results/Params_Transplant/abcd_PS_Params_{i}.txt", sep = '\t', header=0, index_col='stat_num')
    temp_params_AM_rf_df        = pd.read_csv(f"../../Results/Params_Transplant/abcd_RF_Params_{i}.txt", sep = '\t', header=0, index_col='stat_num')
    temp_params_AM_svm_df       = pd.read_csv(f"../../Results/Params_Transplant/abcd_SVM_Params_{i}.txt", sep = '\t', header=0, index_col='stat_num')
    temp_params_AM_xgb_df       = pd.read_csv(f"../../Results/Params_Transplant/abcd_XGB_Params_{i}.txt", sep = '\t', header=0, index_col='stat_num')

    temp_params_DM_SP_AM_df     = pd.read_csv(f"../../Results/Params_Transplant/DWBM_SP_AM_Params_{i}.txt", sep = '\t', header=0, index_col='stat_num')
    temp_params_DM_SP_IDW_df    = pd.read_csv(f"../../Results/Params_Transplant/DWBM_SP_IDW_Params_{i}.txt", sep = '\t', header=0, index_col='stat_num')
    temp_params_DM_PS_df        = pd.read_csv(f"../../Results/Params_Transplant/DWBM_PS_Params_{i}.txt", sep = '\t', header=0, index_col='stat_num')
    temp_params_DM_rf_df        = pd.read_csv(f"../../Results/Params_Transplant/DWBM_RF_Params_{i}.txt", sep = '\t', header=0, index_col='stat_num')
    temp_params_DM_svm_df       = pd.read_csv(f"../../Results/Params_Transplant/DWBM_SVM_Params_{i}.txt", sep = '\t', header=0, index_col='stat_num')
    temp_params_DM_xgb_df       = pd.read_csv(f"../../Results/Params_Transplant/DWBM_XGB_Params_{i}.txt", sep = '\t', header=0, index_col='stat_num')

    temp_params_GYM_SP_AM_df    = pd.read_csv(f"../../Results/Params_Transplant/GmYWBM_SP_AM_Params_{i}.txt", sep = '\t', header=0, index_col='stat_num')
    temp_params_GYM_SP_IDW_df   = pd.read_csv(f"../../Results/Params_Transplant/GmYWBM_SP_IDW_Params_{i}.txt", sep = '\t', header=0, index_col='stat_num')
    temp_params_GYM_PS_df       = pd.read_csv(f"../../Results/Params_Transplant/GmYWBM_PS_Params_{i}.txt", sep = '\t', header=0, index_col='stat_num')
    temp_params_GYM_rf_df       = pd.read_csv(f"../../Results/Params_Transplant/GmYWBM_RF_Params_{i}.txt", sep = '\t', header=0, index_col='stat_num')
    temp_params_GYM_svm_df      = pd.read_csv(f"../../Results/Params_Transplant/GmYWBM_SVM_Params_{i}.txt", sep = '\t', header=0, index_col='stat_num')
    temp_params_GYM_xgb_df      = pd.read_csv(f"../../Results/Params_Transplant/GmYWBM_XGB_Params_{i}.txt", sep = '\t', header=0, index_col='stat_num')

    not_nan_idx = temp_params_YM_SP_AM_df.dropna(how='all').index

    params_YM_SP_AM_df.loc[not_nan_idx]     = temp_params_YM_SP_AM_df.loc[not_nan_idx]
    params_YM_SP_IDW_df.loc[not_nan_idx]    = temp_params_YM_SP_IDW_df.loc[not_nan_idx]
    params_YM_PS_df.loc[not_nan_idx]        = temp_params_YM_PS_df.loc[not_nan_idx]
    params_YM_rf_df.loc[not_nan_idx]        = temp_params_YM_rf_df.loc[not_nan_idx]
    params_YM_svm_df.loc[not_nan_idx]       = temp_params_YM_svm_df.loc[not_nan_idx]
    params_YM_xgb_df.loc[not_nan_idx]       = temp_params_YM_xgb_df.loc[not_nan_idx]

    params_AM_SP_AM_df.loc[not_nan_idx]     = temp_params_AM_SP_AM_df.loc[not_nan_idx]
    params_AM_SP_IDW_df.loc[not_nan_idx]    = temp_params_AM_SP_IDW_df.loc[not_nan_idx]
    params_AM_PS_df.loc[not_nan_idx]        = temp_params_AM_PS_df.loc[not_nan_idx]
    params_AM_rf_df.loc[not_nan_idx]        = temp_params_AM_rf_df.loc[not_nan_idx]
    params_AM_svm_df.loc[not_nan_idx]       = temp_params_AM_svm_df.loc[not_nan_idx]
    params_AM_xgb_df.loc[not_nan_idx]       = temp_params_AM_xgb_df.loc[not_nan_idx]

    params_DM_SP_AM_df.loc[not_nan_idx]     = temp_params_DM_SP_AM_df.loc[not_nan_idx]
    params_DM_SP_IDW_df.loc[not_nan_idx]    = temp_params_DM_SP_IDW_df.loc[not_nan_idx]
    params_DM_PS_df.loc[not_nan_idx]        = temp_params_DM_PS_df.loc[not_nan_idx]
    params_DM_rf_df.loc[not_nan_idx]        = temp_params_DM_rf_df.loc[not_nan_idx]
    params_DM_svm_df.loc[not_nan_idx]       = temp_params_DM_svm_df.loc[not_nan_idx]
    params_DM_xgb_df.loc[not_nan_idx]       = temp_params_DM_xgb_df.loc[not_nan_idx]

    params_GYM_SP_AM_df.loc[not_nan_idx]    = temp_params_GYM_SP_AM_df.loc[not_nan_idx]
    params_GYM_SP_IDW_df.loc[not_nan_idx]   = temp_params_GYM_SP_IDW_df.loc[not_nan_idx]
    params_GYM_PS_df.loc[not_nan_idx]       = temp_params_GYM_PS_df.loc[not_nan_idx]
    params_GYM_rf_df.loc[not_nan_idx]       = temp_params_GYM_rf_df.loc[not_nan_idx]
    params_GYM_svm_df.loc[not_nan_idx]      = temp_params_GYM_svm_df.loc[not_nan_idx]
    params_GYM_xgb_df.loc[not_nan_idx]      = temp_params_GYM_xgb_df.loc[not_nan_idx]

params_YM_SP_AM_df.to_csv("../../Results/Params_Transplant/mYWBM_SP_AM_Params.txt", sep = '\t', float_format='%.4f')
params_YM_SP_IDW_df.to_csv("../../Results/Params_Transplant/mYWBM_SP_IDW_Params.txt", sep = '\t', float_format='%.4f')
params_YM_PS_df.to_csv("../../Results/Params_Transplant/mYWBM_PS_Params.txt", sep = '\t', float_format='%.4f')
params_YM_rf_df.to_csv("../../Results/Params_Transplant/mYWBM_RF_Params.txt", sep = '\t', float_format='%.4f')
params_YM_svm_df.to_csv("../../Results/Params_Transplant/mYWBM_SVM_Params.txt", sep = '\t', float_format='%.4f')
params_YM_xgb_df.to_csv("../../Results/Params_Transplant/mYWBM_XGB_Params.txt", sep = '\t', float_format='%.4f')

params_AM_SP_AM_df.to_csv("../../Results/Params_Transplant/abcd_SP_AM_Params.txt", sep = '\t', float_format='%.4f')
params_AM_SP_IDW_df.to_csv("../../Results/Params_Transplant/abcd_SP_IDW_Params.txt", sep = '\t', float_format='%.4f')
params_AM_PS_df.to_csv("../../Results/Params_Transplant/abcd_PS_Params.txt", sep = '\t', float_format='%.4f')
params_AM_rf_df.to_csv("../../Results/Params_Transplant/abcd_RF_Params.txt", sep = '\t', float_format='%.4f')
params_AM_svm_df.to_csv("../../Results/Params_Transplant/abcd_SVM_Params.txt", sep = '\t', float_format='%.4f')
params_AM_xgb_df.to_csv("../../Results/Params_Transplant/abcd_XGB_Params.txt", sep = '\t', float_format='%.4f')

params_DM_SP_AM_df.to_csv("../../Results/Params_Transplant/DWBM_SP_AM_Params.txt", sep = '\t', float_format='%.4f')
params_DM_SP_IDW_df.to_csv("../../Results/Params_Transplant/DWBM_SP_IDW_Params.txt", sep = '\t', float_format='%.4f')
params_DM_PS_df.to_csv("../../Results/Params_Transplant/DWBM_PS_Params.txt", sep = '\t', float_format='%.4f')
params_DM_rf_df.to_csv("../../Results/Params_Transplant/DWBM_RF_Params.txt", sep = '\t', float_format='%.4f')
params_DM_svm_df.to_csv("../../Results/Params_Transplant/DWBM_SVM_Params.txt", sep = '\t', float_format='%.4f')
params_DM_xgb_df.to_csv("../../Results/Params_Transplant/DWBM_XGB_Params.txt", sep = '\t', float_format='%.4f')

params_GYM_SP_AM_df.to_csv("../../Results/Params_Transplant/GmYWBM_SP_AM_Params.txt", sep = '\t', float_format='%.4f')
params_GYM_SP_IDW_df.to_csv("../../Results/Params_Transplant/GmYWBM_SP_IDW_Params.txt", sep = '\t', float_format='%.4f')
params_GYM_PS_df.to_csv("../../Results/Params_Transplant/GmYWBM_PS_Params.txt", sep = '\t', float_format='%.4f')
params_GYM_rf_df.to_csv("../../Results/Params_Transplant/GmYWBM_RF_Params.txt", sep = '\t', float_format='%.4f')
params_GYM_svm_df.to_csv("../../Results/Params_Transplant/GmYWBM_SVM_Params.txt", sep = '\t', float_format='%.4f')
params_GYM_xgb_df.to_csv("../../Results/Params_Transplant/GmYWBM_XGB_Params.txt", sep = '\t', float_format='%.4f')

for i in range(40):
    for prefix in ["mYWBM", "abcd", "DWBM", "GmYWBM"]:
        for method in ["SP_AM", "SP_IDW", "PS", "RF", "SVM", "XGB"]:
            file_path = f"../../Results/Params_Transplant/{prefix}_{method}_Params_{i}.txt"
            if os.path.exists(file_path):
                os.remove(file_path)