# -*- coding: utf-8 -*-
"""
补: 不同样本量 + 右删失对比
"""
import numpy as np
from scipy.optimize import minimize
import sys
sys.path.insert(0, r"G:\OPT\q-Weibull竞争风险论文\code")
from qweibull import log_sf, log_pdf

def sim_one(q_true, lam, beta, n, w=1.0, rc_rate=0.37, right_censor=False, seed=0):
    rng = np.random.default_rng(seed)
    u = rng.random(n)
    if q_true < 1:
        t = ((1-(1-u)**((1-q_true)/(2-q_true)))/((1-q_true)*lam))**(1/beta)
    else:
        t = ((u**((q_true-1)/(q_true-2))-1)/((q_true-1)*lam))**(1/beta)
    
    if right_censor:
        # 右删失: 精确观测 t, 37% 删失
        rc = rng.random(n) < rc_rate
        obs_t = t.copy()
        obs_t[rc] = t[rc] * rng.uniform(0.5, 1.0, rc.sum())
        L = obs_t; R = obs_t; ic = False
    else:
        L = np.floor(t/w)*w; R = L + w
        rc = rng.random(n) < rc_rate
        R[rc] = np.nan
        ic = True
    
    def nll(par):
        q, ll, lb = par
        l = np.exp(ll); b = np.exp(lb)
        if q <= 0.5 or q >= 2.5: return 1e6
        try:
            if ic:
                Sl = np.exp(log_sf(q, l, b, L[~rc]))
                Sr = np.exp(log_sf(q, l, b, R[~rc]))
                p = Sl - Sr
                if np.any(p <= 0): return 1e6
                ll_val = np.log(p).sum() + np.log(np.exp(log_sf(q, l, b, L[rc]))).sum()
            else:
                ll_val = np.log(np.exp(log_pdf(q, l, b, obs_t[~rc]))).sum() + \
                         np.log(np.exp(log_sf(q, l, b, obs_t[rc]))).sum()
            return -ll_val
        except: return 1e6
    
    res = minimize(nll, [q_true, np.log(lam), np.log(beta)], method='Nelder-Mead',
                   options={'maxiter':2000, 'xatol':1e-4})
    return res.x[0]

# 不同样本量
print("=== 不同样本量 (q=1.2, 区间删失, 200 reps) ===")
lam = 1/4**2; beta = 2.0
for n in [200, 500, 1000]:
    qhats = [sim_one(1.2, lam, beta, n, seed=i) for i in range(200)]
    bias = np.mean(qhats) - 1.2
    rmse = np.sqrt(np.mean((np.array(qhats)-1.2)**2))
    print(f"n={n}: bias={bias:+.3f}, RMSE={rmse:.3f}")

print()
print("=== 右删失 vs 区间删失 (q=1.2, n=1000, 200 reps) ===")
q_ic = [sim_one(1.2, lam, beta, 1000, right_censor=False, seed=i) for i in range(200)]
q_rc = [sim_one(1.2, lam, beta, 1000, right_censor=True, seed=i) for i in range(200)]
print(f"区间删失: bias={np.mean(q_ic)-1.2:+.3f}, RMSE={np.sqrt(np.mean((np.array(q_ic)-1.2)**2)):.3f}")
print(f"右删失:   bias={np.mean(q_rc)-1.2:+.3f}, RMSE={np.sqrt(np.mean((np.array(q_rc)-1.2)**2)):.3f}")
