import numpy as np
from numba import jit, njit, float64
import math

@njit
def _nanmean_2d(arr):
    """手动实现 np.nanmean()"""
    total, count = 0.0, 0
    rows, cols = arr.shape
    for i in range(rows):
        for j in range(cols):
            if not np.isnan(arr[i, j]):
                total += arr[i, j]
                count += 1
    return total / count if count > 0 else np.nan

@njit
def _argmin_abs(arr, target):
    """找到数组中最接近 target 值的索引"""
    min_diff = abs(arr[0] - target)
    min_index = 0
    for i in range(1, arr.shape[0]):
        diff = abs(arr[i] - target)
        if diff < min_diff:
            min_diff = diff
            min_index = i
    return min_index

@njit
def _concatenate(S1, S2, S3):
    n = S1.shape[0]  # 时间步数
    x_series = np.empty((n, 3))  # 预分配数组，避免动态扩展
    
    for t in range(n):
        x_series[t, 0] = S1[t]
        x_series[t, 1] = S2[t]
        x_series[t, 2] = S3[t]
    return x_series

@njit
def _nanmean_axis12(matrix_3d):
    t, r, c = matrix_3d.shape  # 获取维度
    result = np.empty(t)  # 存储 (1,2) 维均值

    for i in range(t):  # 遍历时间维度
        total, count = 0.0, 0  # 计算总和和计数
        for j in range(r):
            for k in range(c):
                if not np.isnan(matrix_3d[i, j, k]):  # 跳过 NaN
                    total += matrix_3d[i, j, k]
                    count += 1
        result[i] = total / count if count > 0 else np.nan  # 计算均值
    return result

@njit
def _cal_mean(x):
    """计算均值"""
    total = 0.0
    for i in range(x.shape[0]):
        total += x[i]
    return total / x.shape[0]

@njit
def _get_min_along_axis0(x):
    """获取矩阵每列的最小值"""
    min_values = np.empty(x.shape[1], dtype=np.float64)
    for j in range(x.shape[1]):
        min_values[j] = x[:, j].min()
    return min_values

@njit 
def _get_max_along_axis0(x):
    """获取矩阵每列的最大值"""
    max_values = np.empty(x.shape[1], dtype=np.float64)
    for j in range(x.shape[1]):
        max_values[j] = x[:, j].max()
    return max_values

@njit
def _get_valid_data(observation, simulation):
    n = len(observation)
    # 统计有效数据
    valid_obs = np.empty(n, dtype=np.float64)
    valid_sim = np.empty(n, dtype=np.float64)
    count = 0
    for i in range(n):
        if not math.isnan(observation[i]) and not math.isnan(simulation[i]):
            valid_obs[count] = observation[i]
            valid_sim[count] = simulation[i]
            count += 1
    return valid_obs[:count], valid_sim[:count], count

@njit
def nash_sutcliffe_efficiency(observation, simulation):
    """计算 Nash-Sutcliffe 效率系数 (NSE)"""
    valid_obs, valid_sim, count = _get_valid_data(observation, simulation)
    if count == 0:
        return np.nan
    # 计算均值
    obs_mean = _cal_mean(valid_obs[:count])
    # 计算 NSE 公式
    num = 0.0
    denom = 0.0
    for i in range(count):
        num += (valid_obs[i] - valid_sim[i]) ** 2
        denom += (valid_obs[i] - obs_mean) ** 2

    return 1 - num / denom if denom != 0 else np.nan  # 避免除零错误

@njit
def relative_error(observation, simulation):
    """计算相对误差"""
    valid_obs, valid_sim, count = _get_valid_data(observation, simulation)
    if count == 0:
        return np.nan
    obs_mean = _cal_mean(valid_obs)
    sim_mean = _cal_mean(valid_sim)
    RE = (sim_mean - obs_mean) / obs_mean
    return RE

@njit
def pearson_correlation_coefficient(observation, simulation):
    """计算 Pearson 相关系数"""
    valid_obs, valid_sim, count = _get_valid_data(observation, simulation)
    if count == 0:
        return np.nan
    # 计算均值
    obs_mean = _cal_mean(valid_obs)
    sim_mean = _cal_mean(valid_sim)
    # 计算标准差
    obs_std = 0.0
    sim_std = 0.0
    for i in range(count):
        obs_std += (valid_obs[i] - obs_mean) ** 2
        sim_std += (valid_sim[i] - sim_mean) ** 2
    obs_std = np.sqrt(obs_std / count)
    sim_std = np.sqrt(sim_std / count)
    # 计算相关系数
    cov = 0.0
    for i in range(count):
        cov += (valid_obs[i] - obs_mean) * (valid_sim[i] - sim_mean)
    r = cov / (count * obs_std * sim_std) if (obs_std * sim_std) != 0 else 0.0
    return r

@njit
def relative_standard_deviation(observation, simulation):
    """计算相对标准差 (rSD)"""
    valid_obs, valid_sim, count = _get_valid_data(observation, simulation)
    if count == 0:
        return np.nan
    # 计算均值
    obs_mean = _cal_mean(valid_obs)
    sim_mean = _cal_mean(valid_sim)
    # 计算标准差
    obs_std = 0.0
    sim_std = 0.0
    for i in range(count):
        obs_std += (valid_obs[i] - obs_mean) ** 2
        sim_std += (valid_sim[i] - sim_mean) ** 2
    obs_std = np.sqrt(obs_std / count)
    sim_std = np.sqrt(sim_std / count)
    # 计算 rSD
    rSD = (sim_std / sim_mean) / (obs_std / obs_mean) if obs_mean != 0 and sim_mean != 0 else np.nan
    return rSD

@njit
def kling_gupta_efficiency(observation, simulation):
    """计算 Kling-Gupta 效率系数 (KGE)"""
    valid_obs, valid_sim, count = _get_valid_data(observation, simulation)
    if count == 0:
        return np.nan
    # 计算均值
    obs_mean = _cal_mean(valid_obs)
    sim_mean = _cal_mean(valid_sim)
    # 计算标准差
    obs_std = 0.0
    sim_std = 0.0
    for i in range(count):
        obs_std += (valid_obs[i] - obs_mean) ** 2
        sim_std += (valid_sim[i] - sim_mean) ** 2
    obs_std = np.sqrt(obs_std / count)
    sim_std = np.sqrt(sim_std / count)
    # 计算相关系数
    cov = 0.0
    for i in range(count):
        cov += (valid_obs[i] - obs_mean) * (valid_sim[i] - sim_mean)
    r = cov / (count * obs_std * sim_std) if (obs_std * sim_std) != 0 else 0.0
    # 计算 alpha 和 beta
    alpha = sim_std / obs_std if obs_std != 0 else np.nan
    beta = sim_mean / obs_mean if obs_mean != 0 else np.nan
    # 计算 KGE
    kge = 1.0 - np.sqrt((r - 1.0)**2 + (alpha - 1.0)**2 + (beta - 1.0)**2)
    return kge