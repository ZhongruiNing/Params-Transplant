import numpy as np
import sys

sys.path.append('../')

from numba import njit
from Rewrite_Func import _concatenate, _nanmean_axis12
from Water_Blance_Model import RCCCWBM, water_capacity, YWBMnlS, mYWBMnlS

@njit
def GRWBMu(x, params):
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
            Qsim_M[:, ii, jj] = RCCCWBM(x_series, params)
    return _nanmean_axis12(Qsim_M)

@njit
def GYWBMu(x, params):
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
            Qsim_M[:, ii, jj] = YWBMnlS(x_series, params)
    return _nanmean_axis12(Qsim_M)

@njit
def GmYWBMu(x, params):
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
def GRWBM(x, params):
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

    smax_M  = mask * smax * np.ones_like(mask)

    Qsim_M    = np.full_like(PRE, np.nan)

    r, c = mask.shape
    for ii in range(r):
        for jj in range(c):
            if np.isnan(mask[ii, jj]):
                continue
            params = np.array([Ks, Kg, alpha, smax_M[ii, jj], Ksn])
            x_series = _concatenate(PRE[:, ii, jj], TMP[:, ii, jj], PET[:, ii, jj])
            Qsim_M[:, ii, jj] = RCCCWBM(x_series, params)
    return _nanmean_axis12(Qsim_M)

@njit
def GYWBM(x, params):
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

    smax_M  = mask * smax * np.ones_like(mask)

    Qsim_M    = np.full_like(PRE, np.nan)

    r, c = mask.shape
    for ii in range(r):
        for jj in range(c):
            if np.isnan(mask[ii, jj]):
                continue
            params = np.array([Ks, Kg, alpha, smax_M[ii, jj], Ksn])
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

    smax_M  = mask * smax * np.ones_like(mask)

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