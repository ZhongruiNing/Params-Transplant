import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
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

Basin_Properties = pd.read_csv("../../Data/Properties/Basin_Properties.txt", sep = '\t', header=0, index_col='stat_num')
source_properties = Basin_Properties[['Climate', 'Clay', 'Silt', 'Sand', 'Slope', 'BFI', 'PRE', 'TMP', 'PET', 'TMAX', 'TMIN', 'AE', 'NDVI', 'TI']].values

sim_results = pd.read_csv("../../Results/Weighted_Average/Weighted_Average_Results.txt", sep="\t", index_col='stat_num')[['r_kge_YM', 'r_kge_AM', 'r_kge_DM']]
sim_results['best_scale'] = sim_results.idxmax(axis=1).map({'r_kge_YM': 0, 'r_kge_AM': 1, 'r_kge_DM': 2})

# 定义四个分类器
models = {
    "RandomForest": RandomForestClassifier(n_estimators=200, random_state=42),
    "SVM": Pipeline([("scaler", StandardScaler()), 
                     ("clf", SVC(kernel="rbf", probability=True, random_state=42))]),
    "XGBoost": XGBClassifier(n_estimators=300, max_depth=10, learning_rate=0.01, random_state=42),
    "KNN": Pipeline([("scaler", StandardScaler()), 
                     ("clf", KNeighborsClassifier(n_neighbors=15))])
}

pred_best_models = pd.DataFrame(index=basin_list, columns=models.keys())

def process_basin(b):
    basin = basin_list[b]
    print(f"Processing basin {basin} ({b+1}/{len(basin_list)})")

    # 获取所有流域，除这个流域外的属性和标签
    X_train = np.vstack([source_properties[:b], source_properties[b+1:]])
    y_train = np.hstack([sim_results['best_scale'][:b], sim_results['best_scale'][b+1:]])
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
        
    # 获取当前流域的属性
    target_properties = Basin_Properties.loc[basin, ['Climate', 'Clay', 'Silt', 'Sand', 'Slope', 'BFI', 'PRE', 'TMP', 'PET', 'TMAX', 'TMIN', 'AE', 'NDVI', 'TI']].values.reshape(1, -1)
    target_properties_scaled = scaler.transform(target_properties)

    # 训练并预测
    for model_name, model in models.items():
        trained_model = model.fit(X_train_scaled, y_train)
        pred_best_models.loc[basin, model_name] = trained_model.predict(target_properties_scaled)[0]

from concurrent.futures import ThreadPoolExecutor, as_completed

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
    pred_best_models = pd.DataFrame(index=basin_list, columns=models.keys())
    pred_best_models.index.name = 'stat_num'

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_basin, i) for i in range(start_list[rank], end_list[rank])]
        for future in as_completed(futures):
            future.result()

    pred_best_models.to_csv(f"../../Results/Best_Model_Transplant/Best_Model_Transplant_{rank}.txt", sep='\t')

    comm.Barrier()