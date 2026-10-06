# -*- coding: utf-8 -*-
"""
真实数据应用: 有界支持检验
三个数据集: 牙齿萌出, 糖尿病, 小鼠肿瘤
"""
import numpy as np
from scipy.optimize import minimize
from scipy.stats import chi2
import pandas as pd
import sys
sys.path.insert(0, r"G:\OPT\q-Weibull竞争风险论文\code")
from qweibull import log_sf as log_sf_q

def fit_qweibull(L, R, rc, q_fixed=None):
    def nll(par):
        if q_fixed is not None:
            q = q_fixed; ll, lb = par
        else:
            q, ll, lb = par
        lam = np.exp(ll); b = np.exp(lb)
        if q <= 0.2 or q >= 3.0: return 1e6
        try:
            Sl = np.exp(log_sf_q(q, lam, b, L[~rc]))
            Sr = np.exp(log_sf_q(q, lam, b, R[~rc]))
            p = Sl - Sr
            if np.any(p <= 0): return 1e6
            return -np.log(p).sum() - np.log(np.exp(log_sf_q(q, lam, b, L[rc]))).sum()
        except: return 1e6
    if q_fixed is not None:
        r = minimize(nll, [np.log(0.1), np.log(2)], method='Nelder-Mead',
                     options={'maxiter':2000, 'xatol':1e-4})
        return q_fixed, np.exp(r.x[0]), np.exp(r.x[1]), r.fun
    else:
        r = minimize(nll, [1.0, np.log(0.1), np.log(2)], method='Nelder-Mead',
                     options={'maxiter':2000, 'xatol':1e-4})
        return r.x[0], np.exp(r.x[1]), np.exp(r.x[2]), r.fun

# ===== 1. 牙齿萌出数据 =====
print("="*60)
print("数据集1: 牙齿萌出 (Tandmobiel)")
print("="*60)
df = pd.read_csv(r"G:\OPT\JSCS-WFC20261005\data\tooth24_clean.csv")
print(f"n = {len(df)}")
print(f"列名: {df.columns.tolist()}")

L = df['left'].values.astype(float)
R = df['right'].values.astype(float)
rc = df['censored'].values.astype(bool)
print(f"区间删失: {(~rc).sum()}, 右删失: {rc.sum()}")
print(f"L范围: [{L.min():.1f}, {L.max():.1f}]")
print(f"R范围: [{R.min():.1f}, {R.max():.1f}]")

q_hat, lam_hat, b_hat, nll_q = fit_qweibull(L, R, rc)
_, _, _, nll_w = fit_qweibull(L, R, rc, q_fixed=1.0)
LR = 2 * (nll_w - nll_q)
p_val = 1 - chi2.cdf(max(LR, 0), df=1)

print(f"q̂ = {q_hat:.3f}")
print(f"λ̂ = {lam_hat:.4f}")
print(f"β̂ = {b_hat:.3f}")
if q_hat < 1:
    T_max = (1/((1-q_hat)*lam_hat))**(1/b_hat)
    print(f"T_max 估计 = {T_max:.2f}")
else:
    print("q̂ >= 1, 无有界支持")
print(f"nll(q自由) = {nll_q:.1f}")
print(f"nll(q=1) = {nll_w:.1f}")
print(f"LR = {LR:.3f}, p = {p_val:.4f}")
print()

# ===== 2. 糖尿病数据 =====
print("="*60)
print("数据集2: 糖尿病")
print("="*60)
from scipy.io import loadmat
mat = loadmat(r"G:\OPT\JSCS-WFC20261005\data\IR_diabetes.RData")
# RData 格式需要用 pyreadr
try:
    import pyreadr
    result = pyreadr.read_r(r"G:\OPT\JSCS-WFC20261005\data\IR_diabetes.RData")
    for k, v in result.items():
        print(f"  {k}: shape={v.shape}, cols={v.columns.tolist()[:5]}")
except:
    print("  (无法读取 RData, 跳过)")

print()

# ===== 3. 小鼠肿瘤数据 =====
print("="*60)
print("数据集3: 小鼠肿瘤")
print("="*60)
try:
    import pyreadr
    result = pyreadr.read_r(r"G:\OPT\JSCS-WFC20261005\data\miceData.RData")
    for k, v in result.items():
        print(f"  {k}: shape={v.shape}, cols={v.columns.tolist()[:5]}")
        print(v.head())
except:
    print("  (无法读取 RData, 跳过)")
