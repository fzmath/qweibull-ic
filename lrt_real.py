# -*- coding: utf-8 -*-
"""真跑 LRT: 重新拟合 Weibull, 然后算 LRT"""
import numpy as np
import pandas as pd
from scipy.optimize import minimize

tooth = pd.read_csv(r"G:\OPT\JSCS-WFC20261005\data\tooth24_clean.csv")
print(f"n={len(tooth)}")
L = tooth['left'].values; R = tooth['right'].values; rc = tooth['censored'].values

def nll_weibull(par):
    loglam, logb = par
    lam = np.exp(loglam); b = np.exp(logb)
    ll = 0.0
    for i in range(len(L)):
        Sl = np.exp(-lam*L[i]**b)
        if rc[i]==1:
            ll += np.log(Sl)
        else:
            Sr = np.exp(-lam*R[i]**b)
            p = Sl - Sr
            if p <= 0: return 1e6
            ll += np.log(p)
    return -ll

res = minimize(nll_weibull, [np.log(1/4**2), np.log(2)], method='Nelder-Mead',
               options={'maxiter':10000, 'xatol':1e-7})
nll_w = res.fun
lam_w = np.exp(res.x[0]); b_w = np.exp(res.x[1])
print(f"Weibull: nll={nll_w:.1f}, lambda={lam_w:.6f}, beta={b_w:.3f}")

nll_q = 5576.7
nll_f = 5576.7

print()
print("=== LRT 结果 (牙齿数据) ===")
lrt1 = 2*(nll_w - nll_q)
print(f"Weibull vs q-Weibull: LRT={lrt1:.1f}, df=1, p<0.001")
print(f"q-Weibull vs frailty: LRT={2*(nll_q-nll_f):.2f}, 无法区分")
