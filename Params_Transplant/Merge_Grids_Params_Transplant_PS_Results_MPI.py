import numpy as np
import pandas as pd
import os

script_dir = os.path.abspath(os.path.dirname(os.path.abspath(__file__)))
os.chdir(script_dir)

grids_params_PS_YM_df = pd.read_csv("../../Results/Grids_Params_Transplant/YM_PS_0.txt", sep='\t', index_col=0, header=0)
grids_params_PS_AM_df = pd.read_csv("../../Results/Grids_Params_Transplant/AM_PS_0.txt", sep='\t', index_col=0, header=0)
grids_params_PS_DM_df = pd.read_csv("../../Results/Grids_Params_Transplant/DM_PS_0.txt", sep='\t', index_col=0, header=0)

print(grids_params_PS_AM_df)
print(grids_params_PS_YM_df)
print(grids_params_PS_DM_df)

for i in range(1, 196):
    print(i)
    temp_grids_params_PS_YM_df = pd.read_csv(f"../../Results/Grids_Params_Transplant/YM_PS_{i}.txt", sep='\t', index_col=0, header=0)
    temp_grids_params_PS_AM_df = pd.read_csv(f"../../Results/Grids_Params_Transplant/AM_PS_{i}.txt", sep='\t', index_col=0, header=0)
    temp_grids_params_PS_DM_df = pd.read_csv(f"../../Results/Grids_Params_Transplant/DM_PS_{i}.txt", sep='\t', index_col=0, header=0)

    not_nan_idx = temp_grids_params_PS_YM_df.dropna(how='all').index

    grids_params_PS_YM_df.loc[not_nan_idx] = temp_grids_params_PS_YM_df.loc[not_nan_idx]
    grids_params_PS_AM_df.loc[not_nan_idx] = temp_grids_params_PS_AM_df.loc[not_nan_idx]
    grids_params_PS_DM_df.loc[not_nan_idx] = temp_grids_params_PS_DM_df.loc[not_nan_idx]

grids_params_PS_YM_df.to_csv("../../Results/Grids_Params_Transplant/YM_PS.txt", sep='\t', index=True, header=True, float_format='%.4f')
grids_params_PS_AM_df.to_csv("../../Results/Grids_Params_Transplant/AM_PS.txt", sep='\t', index=True, header=True, float_format='%.4f')
grids_params_PS_DM_df.to_csv("../../Results/Grids_Params_Transplant/DM_PS.txt", sep='\t', index=True, header=True, float_format='%.4f')

for i in range(196):
    for prefix in ["YM", "AM", "DM"]:
        file_path = f"../../Results/Grids_Params_Transplant/{prefix}_PS_{i}.txt"
        if os.path.exists(file_path):
            os.remove(file_path)