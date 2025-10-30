import numpy as np
import pandas as pd
import os

Best_Model_Transplant = pd.read_csv(f"../../Results/Best_Model_Transplant/Best_Model_Transplant_1.txt", sep='\t', header=0, index_col='stat_num')

for i in range(1, 40):
    temp_Best_Model_Transplant = pd.read_csv(f"../../Results/Best_Model_Transplant/Best_Model_Transplant_{i}.txt", sep='\t', header=0, index_col='stat_num')
    not_nan_idx = temp_Best_Model_Transplant.dropna(how='all').index
    Best_Model_Transplant.loc[not_nan_idx] = temp_Best_Model_Transplant.loc[not_nan_idx]

Best_Model_Transplant.to_csv(f"../../Results/Best_Model_Transplant/Best_Model_Transplant.txt", sep='\t')

for i in range(0, 40):
    os.remove(f"../../Results/Best_Model_Transplant/Best_Model_Transplant_{i}.txt")