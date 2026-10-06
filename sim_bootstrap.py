# -*- coding: utf-8 -*-
"""
Bootstrap p 值验证 LRT size
对比渐近卡方 vs bootstrap
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
    _, lam_hat, b_hat, nll_q = fit_qweibull(L, R, rc)
    _, _, _, nll_w = fit_qweibull(L, R, rc, q_fixed=1.0)
    return 2 * (nll_w - nll_q), lam_hat, b_hat

def simulate_weibull(n, rng, lam=1/16, beta=2.0, p_rc=0.37):
    u = rng.random(n)
    t = (-np.log(u)/lam)**(1/beta)
    L = np.floor(t)*1.0; R = L + 1.0
    rc = rng.random(n) < p_rc
    return L, R, rc

# ===== Bootstrap size 验证 =====
print("Bootstrap p 值验证 (Weibull 零假设)")
print("="*50)

B = 50  # bootstrap 次数
R = 50  # 重复次数 (少一点, 因为 bootstrap 慢)
n = 200

rng = np.random.default_rng(42)
asymp_reject = 0
boot_reject = 0

for rep in range(R):
    L, Rc, rc = simulate_weibull(n, rng)
    LR_obs, lam_hat, b_hat = lr_stat(L, Rc, rc)
    
    # 渐近 p 值
    p_asymp = 1 - chi2.cdf(max(LR_obs, 0), df=1)
    if p_asymp < 0.05:
        asymp_reject += 1
    
    # Bootstrap p 值: 在拟合的 Weibull 下重采样
    lr_boot = []
    for b in range(B):
        u = rng.random(n)
        t_boot = (-np.log(u)/lam_hat)**(1/b_hat)
        L_boot = np.floor(t_boot)*1.0; R_boot = L_boot + 1.0
        rc_boot = rng.random(n) < rc.mean()
        lr_b, _, _ = lr_stat(L_boot, R_boot, rc_boot)
        lr_boot.append(lr_b)
    
    p_boot = np.mean(np.array(lr_boot) >= LR_obs)
    if p_boot < 0.05:
        boot_reject += 1
    
    if (rep+1) % 10 == 0:
        print(f"  rep {rep+1}/{R}: 渐近={asymp_reject/(rep+1):.3f}, bootstrap={boot_reject/(rep+1):.3f}")

print()
print(f"最终结果 (n={n}, R={R}, B={B}):")
print(f"  渐近卡方 size = {asymp_reject/R:.3f}")
print(f"  Bootstrap size = {boot_reject/R:.3f}")
print(f"  名义 = 0.050")
