# -*- coding: utf-8 -*-
"""
Profile likelihood CI for q on tooth data
"""
import numpy as np
from scipy.optimize import minimize
import pandas as pd
import sys
sys.path.insert(0, r"G:\OPT\q-Weibull竞争风险论文\code")
from qweibull import log_sf as log_sf_q

def nll_q(q_fixed, L, R, rc):
    def nll(par):
        ll, lb = par
        lam = np.exp(ll); b = np.exp(lb)
        if q_fixed <= 0.3 or q_fixed >= 1.95: return 1e6
        try:
            Sl = np.exp(log_sf_q(q_fixed, lam, b, L[~rc]))
            Sr = np.exp(log_sf_q(q_fixed, lam, b, R[~rc]))
            p = Sl - Sr
            if np.any(p <= 0): return 1e6
            return -np.log(p).sum() - np.log(np.exp(log_sf_q(q_fixed, lam, b, L[rc]))).sum()
        except: return 1e6
    r = minimize(nll, [np.log(0.1), np.log(2)], method='Nelder-Mead',
                 options={'maxiter':2000, 'xatol':1e-4})
    return r.fun

df = pd.read_csv(r"G:\OPT\JSCS-WFC20261005\data\tooth24_clean.csv")
L = df['left'].values.astype(float)
R = df['right'].values.astype(float)
rc = df['censored'].values.astype(bool)

# MLE
nll_min = nll_q(1.41, L, R, rc)
print(f"MLE q=1.41, nll={nll_min:.1f}")

# Profile likelihood CI (95%, chi2 df=1, threshold = 3.84/2 = 1.92)
threshold = nll_min + 1.92

# Find lower bound
print("Finding lower CI bound...")
q_low = 1.0
for q in np.arange(1.0, 1.41, 0.01):
    n = nll_q(q, L, R, rc)
    if n > threshold:
        q_low = q
        break

# Find upper bound
print("Finding upper CI bound...")
q_high = 1.9
for q in np.arange(1.9, 1.41, -0.01):
    n = nll_q(q, L, R, rc)
    if n > threshold:
        q_high = q
        break

print(f"\nProfile likelihood 95% CI for q: [{q_low:.2f}, {q_high:.2f}]")
print(f"MLE: 1.41")
print(f"CI 包含 1.0? {'是' if q_low < 1.0 < q_high else '否'}")
