####### 水量平衡模型
import numpy as np
from numba import njit
import math
import os
import sys
script_dir = os.path.dirname(os.path.abspath(__name__))
os.chdir(script_dir)
sys.path.append('../SCEUA')
sys.path.append('../')

from Rewrite_Func import _nanmean_2d, _argmin_abs, _concatenate, _nanmean_axis12

### Budyko-Fu
@njit
def Fu(x, alpha):
    w = 1 / (1 - alpha)
    y = 1 + x - math.pow((1 + math.pow(x, w)), 1.0 / w)
    return y

### Snow Module
@njit
def linear_snow(Ts, Tm, Ksn, P, T, SN):
    if T < Ts:
        snow_ratio = 1.0
        melt_ratio = 0.0
    elif T > Tm:
        snow_ratio = 0.0
        melt_ratio = 1.0
    else:
        snow_ratio = (Tm - T) / (Tm - Ts)
        melt_ratio = (T - Ts) / (Tm - Ts)
    Psn = P * snow_ratio
    Pr = P * (1.0 - snow_ratio)

    snow = SN + Psn

    melt = Ksn * snow * melt_ratio
    snow -= melt
    SN = snow
    Pr += melt
    return Pr, SN

@njit
def non_linear_snow(Ts, Tm, Ksn, P, T, SN):
    temp_ratio = (T - Ts) / (Ts - Tm)
    temp_clipped = min(temp_ratio, 0.0)
    expr = 1.0 - np.exp(-(temp_clipped**4))
    snow_ratio = max(expr, 0.0)

    Psn = P * snow_ratio
    Pr = P * (1.0 - snow_ratio)

    snow = SN + Psn

    # 修复2: 分步计算避免嵌套
    melt_ratio_temp = (Tm - T - 4) / (Ts - Tm)
    melt_clipped = min(melt_ratio_temp, 0.0)
    melt_expr = 1.0 - np.exp(-(melt_clipped**4))
    melt_ratio = max(melt_expr, 0.0)  # 先计算下限
    melt_ratio = min(melt_ratio, 1.0)  # 再计算上限
    
    melt = Ksn * snow * melt_ratio
    snow -= melt
    SN = snow
    Pr += melt
    return Pr, SN

### GR2M模型
@njit
def GR2M(x, params):
    x1 = params[0]
    x2 = params[1]

    P = x[:, 0]
    E = x[:, 2]

    n = len(P)

    S = np.full(n, np.nan)
    Q = np.full(n, np.nan)
    R = np.full(n, np.nan)
    S2 = np.full(n, np.nan)

    S0 = x1
    R0 = 0

    for i in range(n):
        vphi = np.tanh(P[i] / x1)
        psi = np.tanh(E[i] / x1)

        S1 = (S0 + x1 * vphi) / (1 + vphi * (S0 / x1))
        P1 = P[i] + S0 - S1
        S2[i] = S1 * (1 - psi) / (1 + psi * (1 - (S1 / x1)))

        S[i] = S2[i] / ((1 + (S2[i] / x1) ** 3) ** (1 / 3))

        P2 = S2[i] - S0
        P3 = P1 + P2
        R1 = R0 + P3
        R2 = x2 * R1

        Q[i] = (R2 ** 2) / (R2 + 60)
        R[i] = R2 - Q[i]

        S0 = S[i]
        R0 = R[i]

    return Q

### TPWB
@njit
def TPWB(x, params):
    c = params[0]
    SC = params[1]

    P = x[:, 0]
    PET = x[:, 2]

    n = len(P)

    Q = np.full(n, np.nan)
    E = np.full(n, np.nan)

    S0 = SC

    for i in range(n):
        E[i] = c * P[i] * np.tanh(P[i] / PET[i])
        S0 = S0 + P[i] - E[i]
        Q[i] = S0 * np.tanh(S0 / SC)
        S0 = S0 - Q[i]

    return Q

### abcd
@njit
def abcd_single_step(P, PET, H, S, a, b, c, d):
    W = P + H
    Y_numerator = 0.5 * (W + b) / a
    discriminant = max(Y_numerator**2 - (W * b) / a, 0.0)  # 防止负数开根
    Y_sqrt = np.sqrt(discriminant)
    Y = Y_numerator - Y_sqrt
    H = Y * np.exp(-PET / b)

    Ea_temp = Y * (1.0 - np.exp(-PET / b))
    Ea = min(PET, max(Ea_temp, 0.0))

    S = (c * (W - Y) + S) / (1 + d)
    Q = (1 - c) * (W - Y) + d * S
    S = (1 - d) * S
    return Q, Ea, H, S

@njit
def abcd(x, params):
    a = params[0]
    b = params[1]
    c = params[2]
    d = params[3]

    P = x[:, 0]
    PET = x[:, 2]
    Inv = [200, 5]

    H = Inv[0]
    S = Inv[1]

    n = len(P)

    Q = np.full(n, np.nan)
    Ea = np.full(n, np.nan)

    for i in range(1, n):
        Q[i], Ea[i], H, S = abcd_single_step(P[i], PET[i], H, S, a, b, c, d)
        
    return Q

### abcdlS
@njit
def abcdlS(x, params):
    # 解包参数
    a = params[0]
    b = params[1]
    c = params[2]
    d = params[3]
    Ksn = params[4]

    # 初始化数组
    P = x[:, 0]
    T = x[:, 1]
    PET = x[:, 2]
    Inv = [200, 5]

    H = Inv[0]
    S = Inv[1]

    n = len(P)

    Q = np.full(n, np.nan)
    SN = np.full(n, np.nan)
    Ea = np.full(n, np.nan)
    SN[0] = 0.0

    Tm = -4.0   # 积雪下限温度
    Ts = 4.0    # 积雪上限温度

    for i in range(1, n):
        
        Pr, SN[i] = linear_snow(Ts, Tm, Ksn, P[i], T[i], SN[i-1])
        Q[i], Ea[i], H, S = abcd_single_step(Pr, PET[i], H, S, a, b, c, d)

    return Q

### abcdnlS
@njit
def abcdnlS(x, params):
    # 解包参数
    a = params[0]
    b = params[1]
    c = params[2]
    d = params[3]
    Ksn = params[4]

    # 初始化数组
    P = x[:, 0]
    T = x[:, 1]
    PET = x[:, 2]
    Inv = [200, 5]

    H = Inv[0]
    S = Inv[1]

    n = len(P)

    Q = np.full(n, np.nan)
    SN = np.full(n, np.nan)
    Ea = np.full(n, np.nan)
    SN[0] = 0.0

    Tm = -4.0   # 积雪下限温度
    Ts = 4.0    # 积雪上限温度

    for i in range(1, n):
        
        Pr, SN[i] = non_linear_snow(Ts, Tm, Ksn, P[i], T[i], SN[i-1])
        Q[i], Ea[i], H, S = abcd_single_step(Pr, PET[i], H, S, a, b, c, d)

    return Q
### abcdnlS
@njit
def abcdnlS_RE(x, params):
    # 解包参数
    a = params[0]
    b = params[1]
    c = params[2]
    d = params[3]
    Ksn = params[4]

    # 初始化数组
    P = x[:, 0]
    T = x[:, 1]
    PET = x[:, 2]
    Inv = [200, 5]

    H = Inv[0]
    S = Inv[1]

    n = len(P)

    Q = np.full(n, np.nan)
    SN = np.full(n, np.nan)
    Ea = np.full(n, np.nan)
    SN[0] = 0.0

    Tm = -4.0   # 积雪下限温度
    Ts = 4.0    # 积雪上限温度

    for i in range(1, n):
        
        Pr, SN[i] = non_linear_snow(Ts, Tm, Ksn, P[i], T[i], SN[i-1])
        Q[i], Ea[i], H, S = abcd_single_step(Pr, PET[i], H, S, a, b, c, d)

    return Q, Ea

### DWBM
@njit
def DWBM_single_step(P, PET, S0, G0, alpha1, alpha2, smax, d):
    X0 = smax - S0 + PET
    if P == 0:
        X = 0.0
    else:
        X  = P * Fu(X0 / P, alpha1)
    Qd = max(P - X, 0.0)

    W  = max(X + S0, 0.0)
    Y0 = smax + PET
    if W == 0:
        Y = 0.0
    else:
        Y  = W * Fu(Y0 / W, alpha2)
    if not math.isfinite(Y):
        Y = 0.0

    R  = W - Y
    if W == 0:
        AE = 0.0
    else:
        AE = W * Fu(PET / W, alpha2)
    if not math.isfinite(AE):
        AE = 0.0

    S  = max(Y - AE, 0.01)

    Qb = d * G0
    G  = (1 - d) * G0 + R
    Q  = Qd + Qb

    return Q, S, G, AE

@njit
def DWBM(x, params):
    alpha1  = params[0]
    alpha2  = params[1]
    smax    = params[2]
    d       = params[3]

    P   = x[:, 0]
    PET = x[:, 2]

    n = len(P)

    Q   = np.full(n, np.nan)
    S   = np.full(n, np.nan)
    G   = np.full(n, np.nan)
    AE  = np.full(n, np.nan)

    S[0] = 0.5 * smax
    G[0] = 0.5 * smax

    for i in range(1, n):
        Q[i], S[i], G[i], AE[i] = DWBM_single_step(P[i], PET[i], S[i - 1], G[i - 1], alpha1, alpha2, smax, d)
        
    return Q

### DWBMlS
@njit
def DWBMlS(x, params):
    alpha1  = params[0]
    alpha2  = params[1]
    smax    = params[2]
    d       = params[3]
    Ksn     = params[4]

    P   = x[:, 0]
    T   = x[:, 1]
    PET = x[:, 2]

    n = len(P)

    Q   = np.full(n, np.nan)
    S   = np.full(n, np.nan)
    G   = np.full(n, np.nan)
    AE  = np.full(n, np.nan)
    SN  = np.full(n, np.nan)
    SN[0] = 0.0

    S[0] = 0.5 * smax
    G[0] = 0.5 * smax

    Tm = -4.0   # 积雪下限温度
    Ts = 4.0    # 积雪上限温度

    for i in range(1, n):
        Pr, SN[i] = linear_snow(Ts, Tm, Ksn, P[i], T[i], SN[i-1])
        Q[i], S[i], G[i], AE[i] = DWBM_single_step(Pr, PET[i], S[i - 1], G[i - 1], alpha1, alpha2, smax, d)

    return Q

### DWBMnlS
@njit
def DWBMnlS(x, params):
    alpha1  = params[0]
    alpha2  = params[1]
    smax    = params[2]
    d       = params[3]
    Ksn     = params[4]

    P   = x[:, 0]
    T   = x[:, 1]
    PET = x[:, 2]

    n = len(P)

    Q   = np.full(n, np.nan)
    S   = np.full(n, np.nan)
    G   = np.full(n, np.nan)
    AE  = np.full(n, np.nan)
    SN  = np.full(n, np.nan)
    SN[0] = 0.0

    S[0] = 0.5 * smax
    G[0] = 0.5 * smax

    Tm = -4.0   # 积雪下限温度
    Ts = 4.0    # 积雪上限温度

    for i in range(1, n):
        Pr, SN[i] = non_linear_snow(Ts, Tm, Ksn, P[i], T[i], SN[i-1])
        Q[i], S[i], G[i], AE[i] = DWBM_single_step(Pr, PET[i], S[i - 1], G[i - 1], alpha1, alpha2, smax, d)

    return Q
### DWBMnlS
@njit
def DWBMnlS_RE(x, params):
    alpha1  = params[0]
    alpha2  = params[1]
    smax    = params[2]
    d       = params[3]
    Ksn     = params[4]

    P   = x[:, 0]
    T   = x[:, 1]
    PET = x[:, 2]

    n = len(P)

    Q   = np.full(n, np.nan)
    S   = np.full(n, np.nan)
    G   = np.full(n, np.nan)
    AE  = np.full(n, np.nan)
    SN  = np.full(n, np.nan)
    SN[0] = 0.0

    S[0] = 0.5 * smax
    G[0] = 0.5 * smax

    Tm = -4.0   # 积雪下限温度
    Ts = 4.0    # 积雪上限温度

    for i in range(1, n):
        Pr, SN[i] = non_linear_snow(Ts, Tm, Ksn, P[i], T[i], SN[i-1])
        Q[i], S[i], G[i], AE[i] = DWBM_single_step(Pr, PET[i], S[i - 1], G[i - 1], alpha1, alpha2, smax, d)

    return Q, AE

@njit
def cal_a_from_W0(W0, WM, b):
    WMM = WM * (1 + b)
    part1 = 1 - W0 / WM
    part2 = 1 - math.pow(part1, 1 / (1 + b))
    return WMM * part2
@njit
def cal_W0_from_a(a, WM, b):
    WMM = WM * (1 + b)
    part1 = 1 - a / WMM
    return WM - WM * math.pow(part1, 1 + b)
@njit
def cal_run_gene(a, PE, WM, b):
    if WM == 0:
        print("WM is zero")
    WMM = WM * (1 + b)
    W0 = cal_W0_from_a(a, WM, b)
    if PE <= 0:
        return 0
    if PE + a > WMM:
        return PE - WM + W0
    else:
        return PE - WM + W0 + WM * math.pow(1 - (a + PE) / WMM, (1 + b))
### YWBM（先地下径流产流再蒸发）
@njit
def YWBM_single_step(P, PET, S0, Ks, Kg, alpha, smax):
    # 地下径流
    if S0 <= 0.05 * smax:
        Qg = 0.0
        # 可用水量
        water_available = S0
    else:
        Qg = Kg * S0
        # 可用水量
        water_available = S0 - Qg
        if water_available < 0.05 * smax:
            Qg = S0 - 0.05 * smax
            water_available = 0.05 * smax
    water_available += P
    # 蒸散发
    water_available = max(water_available, 0.01 * smax)
    temp_x = PET / water_available
    AE = water_available * Fu(temp_x, alpha)
    if not math.isfinite(AE):
        AE = 0.9 * water_available

    if water_available - AE > 0.05 * smax:
        water_available -= AE
    else:
        AE = water_available - 0.05 * smax
        water_available = 0.05 * smax

    # 地表径流
    Qs1 = max(0, water_available - smax)
    water_available -= Qs1
    Qs2 = Ks * water_available * math.tanh(P / smax)
    if water_available - Qs2 > 0.05 * smax:
        water_available -= Qs2
    else:
        Qs2 = water_available - 0.05 * smax
        water_available = 0.05 * smax
    Qs = Qs1 + Qs2

    S = water_available
    Q = Qs + Qg
    return Q, S, AE

### mYWBM（先地下径流产流再蒸发）
@njit
def mYWBM_single_step(P, PET, S0, b, Kg, alpha, smax):
    WMM = smax * (1 + b)
    # 地下径流的产流是初始土壤含水量的一个比例。相当于初始土壤含水量乘上一个系数，是剩余土壤含水量，差值就是地下径流产流。如果初始土壤很干燥，含水量低于0.05*smax，就不产生地下径流
    if S0 <= 0.05 * smax:
        Qg = 0.0
        # 可用水量
        S_remain = S0
    else:
        # 地下径流
        S_remain = max((1 - Kg) * S0, 0.05 * smax)
        a0 = cal_a_from_W0(S0, smax, b)
        a1 = cal_a_from_W0(S_remain, smax, b)
        temp_PE = np.abs(a0 - a1)
        Qg = cal_run_gene(a1, temp_PE, smax, b)
    water_available = max(S_remain, 0.05 * smax)
    # 可用水量，用于计算蒸发。假设目前所有的降水落入土壤中，土壤水加上降水组成所有的可用水量
    water_available += P
    temp_WA = water_available
    # 蒸散发
    temp_x = PET / water_available
    AE = water_available * Fu(temp_x, alpha)
    if not math.isfinite(AE):
        AE = 0.9 * water_available
    # if water_available - AE > 0.05 * smax:
    #     water_available -= AE
    # else:
    #     AE = water_available - 0.05 * smax
    #     water_available = 0.05 * smax
    
    # 蒸发从降水和土壤含水量中等比例扣除
    E_from_soil = AE * S0 / temp_WA
    E_from_P = AE * P / temp_WA
    
    if S0 - E_from_soil < 0.05 * smax:
        E_from_soil = S0 - 0.05 * smax
    S_remain = S0 - E_from_soil
    if E_from_P > P:
        E_from_P = P
    PE = P - E_from_P
    AE = E_from_soil + E_from_P

    # 计算当前土壤中的水含量对应的纵坐标
    a_remain = cal_a_from_W0(S_remain, smax, b)
    # 地表径流
    Qs = cal_run_gene(a_remain, PE, smax, b)

    # 迭代土壤含水量
    if a_remain + PE > WMM:
        S = smax
    else:
        S = cal_W0_from_a(a_remain + PE, smax, b)

    Q = Qs + Qg
    return Q, S, AE


@njit
def YWBM(x, params):
    Ks      = params[0]
    Kg      = params[1]
    alpha   = params[2]
    smax    = params[3]

    P   = x[:, 0]
    PET = x[:, 2]

    n = len(P)

    Q   = np.full(n, np.nan)
    S   = np.full(n, np.nan)
    E   = np.full(n, np.nan)

    S[0] = 0.5 * smax

    for i in range(1, n):
        Q[i], S[i], E[i] = YWBM_single_step(P[i], PET[i], S[i - 1], Ks, Kg, alpha, smax)
        
    return Q

### YWBMlS
@njit
def YWBMlS(x, params):
    Ks      = params[0]
    Kg      = params[1]
    alpha   = params[2]
    smax    = params[3]
    Ksn     = params[4]

    P   = x[:, 0]
    T   = x[:, 1]
    PET = x[:, 2]

    n = len(P)

    Q   = np.full(n, np.nan)
    S   = np.full(n, np.nan)
    E   = np.full(n, np.nan)
    SN  = np.full(n, np.nan)
    SN[0] = 0.0

    S[0] = 0.5 * smax

    Tm = -4.0   # 积雪下限温度
    Ts = 4.0    # 积雪上限温度

    for i in range(1, n):
        Pr, SN[i] = linear_snow(Ts, Tm, Ksn, P[i], T[i], SN[i-1])
        Q[i], S[i], E[i] = YWBM_single_step(Pr, PET[i], S[i - 1], Ks, Kg, alpha, smax)
        
    return Q

### YWBMnlS
@njit
def YWBMnlS(x, params):
    Ks      = params[0]
    Kg      = params[1]
    alpha   = params[2]
    smax    = params[3]
    Ksn     = params[4]

    P   = x[:, 0]
    T   = x[:, 1]
    PET = x[:, 2]

    n = len(P)

    Q   = np.full(n, np.nan)
    S   = np.full(n, np.nan)
    E   = np.full(n, np.nan)
    SN  = np.full(n, np.nan)
    SN[0] = 0.0

    S[0] = 0.5 * smax

    Tm = -4.0   # 积雪下限温度
    Ts = 4.0    # 积雪上限温度

    for i in range(1, n):
        Pr, SN[i] = non_linear_snow(Ts, Tm, Ksn, P[i], T[i], SN[i-1])
        Q[i], S[i], E[i] = YWBM_single_step(Pr, PET[i], S[i - 1], Ks, Kg, alpha, smax)
    return Q
@njit
def mYWBM(x, params):
    Ks      = params[0]
    Kg      = params[1]
    alpha   = params[2]
    smax    = params[3]

    P   = x[:, 0]
    PET = x[:, 2]

    n = len(P)

    Q   = np.full(n, np.nan)
    S   = np.full(n, np.nan)
    E   = np.full(n, np.nan)

    S[0] = 0.5 * smax

    for i in range(1, n):
        Q[i], S[i], E[i] = mYWBM_single_step(P[i], PET[i], S[i - 1], Ks, Kg, alpha, smax)
        
    return Q

@njit
def mYWBMnlS(x, params):
    Ks      = params[0]
    Kg      = params[1]
    alpha   = params[2]
    smax    = params[3]
    Ksn     = params[4]

    P   = x[:, 0]
    T   = x[:, 1]
    PET = x[:, 2]

    n = len(P)

    Q   = np.full(n, np.nan)
    S   = np.full(n, np.nan)
    E   = np.full(n, np.nan)
    SN  = np.full(n, np.nan)
    SN[0] = 0.0

    S[0] = 0.5 * smax

    Tm = -4.0   # 积雪下限温度
    Ts = 4.0    # 积雪上限温度

    for i in range(1, n):
        Pr, SN[i] = non_linear_snow(Ts, Tm, Ksn, P[i], T[i], SN[i-1])
        Q[i], S[i], E[i] = mYWBM_single_step(Pr, PET[i], S[i - 1], Ks, Kg, alpha, smax)
    return Q
@njit
def mYWBMnlS_RE(x, params):
    Ks      = params[0]
    Kg      = params[1]
    alpha   = params[2]
    smax    = params[3]
    Ksn     = params[4]

    P   = x[:, 0]
    T   = x[:, 1]
    PET = x[:, 2]

    n = len(P)

    Q   = np.full(n, np.nan)
    S   = np.full(n, np.nan)
    E   = np.full(n, np.nan)
    SN  = np.full(n, np.nan)
    SN[0] = 0.0

    S[0] = 0.5 * smax

    Tm = -4.0   # 积雪下限温度
    Ts = 4.0    # 积雪上限温度

    for i in range(1, n):
        Pr, SN[i] = non_linear_snow(Ts, Tm, Ksn, P[i], T[i], SN[i-1])
        Q[i], S[i], E[i] = mYWBM_single_step(Pr, PET[i], S[i - 1], Ks, Kg, alpha, smax)
    return Q, E
@njit
def find_max_min_value(arr):
    max_val = 0  # 初始化最大值
    max_row = 0
    max_col = 0
    min_val = 1000
    min_row = 0
    min_col = 0

    rows, cols = arr.shape  # 获取数组维度

    for i in range(rows):
        for j in range(cols):
            if arr[i, j] > max_val:
                max_val = arr[i, j]
                max_row = i
                max_col = j
            if arr[i, j] < min_val:
                min_val = arr[i, j]
                min_row = i
                min_col = j
    return max_val, max_row, max_col, min_val, min_row, min_col

@njit
def water_capacity(Wmax, Wmin, TI, ST, STP, basin_mask, mask_res, mask_lon, mask_lat, TI_lon, TI_lat):
    # 找到 TI 的最大值和最小值
    TI_max, row_max, col_max, TI_min, row_min, col_min = find_max_min_value(TI)
    # 获取最大值和最小值处的土壤类型
    soil_type_TImax = np.int8(ST[row_max, col_max])
    soil_type_TImin = np.int8(ST[row_min, col_min])
    # 提取土壤参数
    theta_fc_TImax = STP[soil_type_TImax - 1, 2]
    theta_wp_TImax = STP[soil_type_TImax - 1, 3]
    theta_fc_TImin = STP[soil_type_TImin - 1, 2]
    theta_wp_TImin = STP[soil_type_TImin - 1, 3]
    # 计算最大最小包气带厚度
    if theta_fc_TImin - theta_wp_TImin == 0:
        La_max = Wmax / 0.05
    else:
        La_max = Wmax / (theta_fc_TImin - theta_wp_TImin)
    if theta_fc_TImax - theta_wp_TImax == 0:
        La_min = Wmin / 0.05
    else:
        La_min = Wmin / (theta_fc_TImax - theta_wp_TImax)
    # 线性拟合
    k = (La_min - La_max) / (TI_max - TI_min)
    b = La_max - k * TI_min
    # 计算整个流域的 La
    La = k * TI + b
    # 获取流域上每个点的土壤类型
    theta_fc_basin = np.full_like(TI, np.nan)
    theta_wp_basin = np.full_like(TI, np.nan)
    smax_TIres     = np.full_like(TI, np.nan)
    r, c = ST.shape
    for ii in range(r):
        for jj in range(c):
            if np.isnan(ST[ii, jj]):
                continue
            soil_idx = np.int8(ST[ii, jj]) - 1
            theta_fc_basin[ii, jj] = STP[soil_idx, 2]
            theta_wp_basin[ii, jj] = STP[soil_idx, 3]
            temp_smax_tires = (theta_fc_basin[ii, jj] - theta_wp_basin[ii, jj]) * La[ii, jj]
            if temp_smax_tires < 10:
                temp_smax_tires = 10.0
            smax_TIres[ii, jj] = temp_smax_tires
    # 每一个TI格点上的smax
    # smax_TIres = (theta_fc_basin - theta_wp_basin) * La
    # 将TI格点上的数据平均到basin_mask上
    smax_maskres = np.full_like(basin_mask, np.nan)
    r, c = basin_mask.shape
    for ii in range(r):
        for jj in range(c):
            if np.isnan(basin_mask[ii, jj]):
                continue
            # basin_mask格点的中心经纬度
            central_lon_basin_mask_grid = mask_lon[jj]
            central_lat_basin_mask_grid = mask_lat[ii]
            # basin_mask格点的四个角的经纬度
            left_lon_basin_mask_grid   = central_lon_basin_mask_grid - mask_res / 2
            right_lon_basin_mask_grid  = central_lon_basin_mask_grid + mask_res / 2
            top_lat_basin_mask_grid    = central_lat_basin_mask_grid + mask_res / 2
            bottom_lat_basin_mask_grid = central_lat_basin_mask_grid - mask_res / 2
            # 寻找TI格点中离basin_mask格点最近的四个格点
            left_index   = _argmin_abs(TI_lon, left_lon_basin_mask_grid)
            right_index  = _argmin_abs(TI_lon, right_lon_basin_mask_grid)
            top_index    = _argmin_abs(TI_lat, top_lat_basin_mask_grid)
            bottom_index = _argmin_abs(TI_lat, bottom_lat_basin_mask_grid)
            # 计算平均值
            smax_maskres[ii, jj] = _nanmean_2d(smax_TIres[top_index:bottom_index, left_index:right_index])
    return smax_maskres
@njit
def GWBM_YWBMnlS(x, params):
    Ks      = params[0]
    Kg      = params[1]
    alpha   = params[2]
    smax    = params[3]
    Ksn     = params[4]

    PRE = x.PRE
    TMP = x.TMP
    PET = x.PET

    TI      = x.TI
    ST      = x.ST
    mask    = x.mask
    STP     = x.STP

    mask_res = x.mask_res
    mask_lon = x.mask_lon
    mask_lat = x.mask_lat
    TI_lon   = x.TI_lon
    TI_lat   = x.TI_lat

    # smax_M  = water_capacity(smax, 50, TI, ST, STP, mask, mask_res, mask_lon, mask_lat, TI_lon, TI_lat)

    Qsim_M    = np.full_like(PRE, np.nan)

    r, c = mask.shape
    for ii in range(r):
        for jj in range(c):
            if np.isnan(mask[ii, jj]):
                continue
            # params = np.array([Ks, Kg, alpha, smax_M[ii, jj], Ksn])
            params = np.array([Ks, Kg, alpha, smax, Ksn])
            x_series = _concatenate(PRE[:, ii, jj], TMP[:, ii, jj], PET[:, ii, jj])
            Qsim_M[:, ii, jj] = YWBMnlS(x_series, params)
    return _nanmean_axis12(Qsim_M)

@njit
def GmYWBM(x, params):
    Ks      = params[0]
    Kg      = params[1]
    alpha   = params[2]
    smax    = params[3]
    Ksn     = params[4]

    PRE = x.PRE
    TMP = x.TMP
    PET = x.PET

    TI      = x.TI
    ST      = x.ST
    mask    = x.mask
    STP     = x.STP

    mask_res = x.mask_res
    mask_lon = x.mask_lon
    mask_lat = x.mask_lat
    TI_lon   = x.TI_lon
    TI_lat   = x.TI_lat

    smax_M  = water_capacity(smax, 50, TI, ST, STP, mask, mask_res, mask_lon, mask_lat, TI_lon, TI_lat)

    Qsim_M    = np.full_like(PRE, np.nan)

    r, c = mask.shape
    for ii in range(r):
        for jj in range(c):
            if np.isnan(mask[ii, jj]):
                continue
            params = np.array([Ks, Kg, alpha, smax_M[ii, jj], Ksn])
            x_series = _concatenate(PRE[:, ii, jj], TMP[:, ii, jj], PET[:, ii, jj])
            Qsim_M[:, ii, jj] = mYWBMnlS(x_series, params)
    return _nanmean_axis12(Qsim_M)

@njit
def GmYWBM_RE(x, params):
    Ks      = params[0]
    Kg      = params[1]
    alpha   = params[2]
    smax    = params[3]
    Ksn     = params[4]

    PRE = x.PRE
    TMP = x.TMP
    PET = x.PET

    TI      = x.TI
    ST      = x.ST
    mask    = x.mask
    STP     = x.STP

    mask_res = x.mask_res
    mask_lon = x.mask_lon
    mask_lat = x.mask_lat
    TI_lon   = x.TI_lon
    TI_lat   = x.TI_lat

    smax_M  = water_capacity(smax, 50, TI, ST, STP, mask, mask_res, mask_lon, mask_lat, TI_lon, TI_lat)

    Qsim_M    = np.full_like(PRE, np.nan)
    Esim_M    = np.full_like(PRE, np.nan)

    r, c = mask.shape
    for ii in range(r):
        for jj in range(c):
            if np.isnan(mask[ii, jj]):
                continue
            params = np.array([Ks, Kg, alpha, smax_M[ii, jj], Ksn])
            x_series = _concatenate(PRE[:, ii, jj], TMP[:, ii, jj], PET[:, ii, jj])
            Qsim_M[:, ii, jj], Esim_M[:, ii, jj] = mYWBMnlS_RE(x_series, params)
    return _nanmean_axis12(Qsim_M), _nanmean_axis12(Esim_M)

@njit
def RCCCWBM(x, params):
    Ks     = params[0]
    Kg     = params[1]
    alpha  = params[2]
    smax   = params[3]
    Ksn    = params[4]

    P   = x[:, 0]
    T   = x[:, 1]
    PET = x[:, 2]

    n = len(P)

    Q   = np.full(n, np.nan)
    S   = np.full(n, np.nan)
    E   = np.full(n, np.nan)
    SN  = np.full(n, np.nan)

    SN[0] = 0.0

    S[0] = 0.5 * smax

    Tm = -4.0   # 积雪下限温度
    Ts = 4.0    # 积雪上限温度

    for i in range(1, n):
        if T[i] > Ts:
            Pr = P[i]
            Psn = 0.0
        elif T[i] < Tm:
            Pr = 0.0
            Psn = P[i]
        else:
            Psn = P[i] * (Ts - T[i]) / (Ts - Tm)
            Pr  = P[i] - Psn
        # 计算地表径流
        Qs = Ks * Pr * S[i-1] / smax
        # 计算地下径流
        Qg = Kg * S[i-1]
        # 计算融雪径流
        SN[i] = SN[i-1] + Psn
        melt_ratio = min(1.0, Ksn * math.exp((T[i] - Ts) / (Ts - Tm)))
        Qsn = melt_ratio * SN[i]
        SN[i] -= Qsn
        # 计算蒸散发
        E[i] = alpha * PET[i] * S[i-1] / smax
        # 计算土壤蓄水量
        S[i] = S[i-1] + Pr - Qs - Qg - E[i]
        if S[i] > smax:
            Qs = Qs + Qs * (S[i] - smax) / (Qs + E[i] + 0.05)
            E[i]  = E[i] + E[i] * (S[i] - smax) / (Qs + E[i] + 0.05)
            S[i] = smax
        elif S[i] < 0.1 * smax:
            Qs = Qs + Qs * (S[i] - 0.1 * smax) / (Qs + E[i] + 0.05)
            E[i]  = E[i] + E[i] * (S[i] - 0.1 * smax) / (Qs + E[i] + 0.05)
            S[i] = 0.1 * smax
        # 计算总径流
        Q[i] = Qs + Qg + Qsn
    return Q

@njit
def RCCCWBM_nn(x, params):
    Ks     = params[0]
    Kg     = params[1]
    alpha  = params[2]
    smax   = params[3]

    P   = x[:, 0]
    PET = x[:, 2]

    n = len(P)

    Q   = np.full(n, np.nan)
    S   = np.full(n, np.nan)
    E   = np.full(n, np.nan)

    S[0] = 0.5 * smax

    for i in range(1, n):
        Pr = P[i]
        # 计算地表径流
        Qs = Ks * Pr * S[i-1] / smax
        # 计算地下径流
        Qg = Kg * S[i-1]
        # 计算蒸散发
        E[i] = alpha * PET[i] * S[i-1] / smax
        # 计算土壤蓄水量
        S[i] = S[i-1] + Pr - Qs - Qg - E[i]
        if S[i] > smax:
            Qs = Qs + Qs * (S[i] - smax) / (Qs + E[i] + 0.05)
            E[i]  = E[i] + E[i] * (S[i] - smax) / (Qs + E[i] + 0.05)
            S[i] = smax
        elif S[i] < 0.1 * smax:
            Qs = Qs + Qs * (S[i] - 0.1 * smax) / (Qs + E[i] + 0.05)
            E[i]  = E[i] + E[i] * (S[i] - 0.1 * smax) / (Qs + E[i] + 0.05)
            S[i] = 0.1 * smax
        # 计算总径流
        Q[i] = Qs + Qg
    return Q