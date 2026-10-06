# -*- coding: utf-8 -*-
"""
正式仿真: 有界支持检验的 size 和 power
200 reps, bootstrap p 值
"""
import numpy as np
from scipy.optimize import minimize
from scipy.stats import chi2
import sys, time
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
        r = minimize(nll, [np.log(1/16), np.log(2)], method='Nelder-Mead',
                     options={'maxiter':1500, 'xatol':1e-3})
        return q_fixed, np.exp(r.x[0]), np.exp(r.x[1]), r.fun
    else:
        r = minimize(nll, [1.0, np.log(1/16), np.log(2)], method='Nelder-Mead',
                     options={'maxiter':1500, 'xatol':1e-3})
        return r.x[0], np.exp(r.x[1]), np.exp(r.x[2]), r.fun

def lr_stat(L, R, rc):
    q_hat, lam_hat, b_hat, nll_q = fit_qweibull(L, R, rc)
    _, _, _, nll_w = fit_qweibull(L, R, rc, q_fixed=1.0)
    LR = 2 * (nll_w - nll_q)
    return LR, q_hat, lam_hat, b_hat, nll_q

def simulate_one(q_true, n, rng, lam=1/16, beta=2.0, p_rc=0.37):
    u = rng.random(n)
    if q_true == 1.0:
        t = (-np.log(u)/lam)**(1/beta)
    else:
        t = ((u**((q_true-1)/(q_true-2))-1)/((q_true-1)*lam))**(1/beta)
    L = np.floor(t)*1.0; R = L + 1.0
    rc = rng.random(n) < p_rc
    return L, R, rc

def bootstrap_pvalue(L, R, rc, q_hat, lam_hat, b_hat, B=50, rng=None):
    """在 q=1 (Weibull) 零假设下 bootstrap"""
    if rng is None: rng = np.random.default_rng()
    n = len(L)
    lr_boot = []
    for b in range(B):
        u = rng.random(n)
        t_sim = (-np.log(u)/lam_hat)**(1/b_hat)
        L_sim = np.floor(t_sim)*1.0; R_sim = L_sim + 1.0
        rc_sim = rng.random(n) < rc.mean()
        lr_b, _, _, _, _ = lr_stat(L_sim, R_sim, rc_sim)
        lr_boot.append(lr_b)
    return np.mean(np.array(lr_boot) >= 0)  # placeholder

# ===== 主仿真 =====
print("开始仿真...")
t0 = time.time()

scenarios = [
    (1.0, 200, "Weibull n=200 (size)"),
    (0.8, 200, "q=0.8 n=200"),
    (0.7, 200, "q=0.7 n=200"),
    (0.6, 200, "q=0.6 n=200"),
    (1.0, 500, "Weibull n=500 (size)"),
    (0.7, 500, "q=0.7 n=500"),
]

R = 200
results = {}

for q_true, n, label in scenarios:
    rng = np.random.default_rng(42)
    lr_vals = []
    q_hats = []
    rejections_asymp = []  # 用卡方近似 (df=1)
    
    for rep in range(R):
        L, Rc, rc = simulate_one(q_true, n, rng)
        LR, q_hat, lam_hat, b_hat, nll_q = lr_stat(L, Rc, rc)
        lr_vals.append(LR)
        q_hats.append(q_hat)
        # 渐近 p 值 (卡方 df=1, 单侧)
        p_asymp = 1 - chi2.cdf(max(LR, 0), df=1)
        rejections_asymp.append(p_asymp < 0.05)
    
    lr_vals = np.array(lr_vals)
    q_hats = np.array(q_hats)
    
    results[label] = {
        'q_hat_mean': np.mean(q_hats),
        'q_hat_sd': np.std(q_hats),
        'lr_mean': np.mean(lr_vals),
        'lr_median': np.median(lr_vals),
        'rejection_rate': np.mean(rejections_asymp),
    }
    
    print(f"{label}:")
    print(f"  q̂ = {np.mean(q_hats):.3f} ± {np.std(q_hats):.3f}")
    print(f"  LR = {np.mean(lr_vals):.2f} (中位 {np.median(lr_vals):.2f})")
    print(f"  拒绝率 (α=0.05) = {np.mean(rejections_asymp):.3f}")
    print()

print(f"总耗时: {time.time()-t0:.1f}s")
