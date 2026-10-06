# -*- coding: utf-8 -*-
"""修复 Weibull 拟合起点, 出干净对比表."""
import numpy as np, pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm

df = pd.read_csv(r"G:\OPT\Weibull发表策略\data\tooth24_clean.csv")
L = df['left'].values; R = df['right'].values; rc = df['censored'].values.astype(bool)

def nll_weibull(theta):
    loglam, logb = theta; b=np.exp(logb); lam=np.exp(loglam)
    Sl = np.exp(-lam*L**b)
    if np.any(Sl<=0): return 1e10
    ll = np.log(Sl[rc]).sum()
    p = Sl[~rc] - np.exp(-lam*R[~rc]**b)
    if np.any(p<=0): return 1e10
    return -(ll + np.log(p).sum())

# 试多个起点
best=None
for s in [(-8,1.5),(-9,1.7),(-10,1.6),(-7,1.8),(-8.5,1.6)]:
    r=minimize(nll_weibull, s, method="Nelder-Mead", options={"xatol":1e-7,"maxiter":30000})
    if best is None or r.fun<best.fun: best=r
print(f"Weibull: lam={np.exp(best.x[0]):.6f}, beta={np.exp(best.x[1]):.3f}, nll={best.fun:.1f}, AIC={4+2*best.fun:.1f}")
