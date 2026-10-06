# -*- coding: utf-8 -*-
"""
q-Weibull vs frailty 判别检验
检验统计量: 两个模型拟合的生存曲线最大差值
Bootstrap 算 p 值
"""
import numpy as np
from scipy.optimize import minimize
import sys
sys.path.insert(0, r"G:\OPT\q-Weibull竞争风险论文\code")
from qweibull import log_sf as log_sf_q

def log_sf_frailty(theta, lam, beta, t):
    return -1/theta * np.log(1 + theta*lam*t**beta)

def fit_qweibull(L, R, rc):
    def nll(par):
        q, ll, lb = par
        lam = np.exp(ll); b = np.exp(lb)
        if q <= 0.5 or q >= 2.5: return 1e6
        try:
            Sl = np.exp(log_sf_q(q, lam, b, L[~rc]))
            Sr = np.exp(log_sf_q(q, lam, b, R[~rc]))
            p = Sl - Sr
            if np.any(p <= 0): return 1e6
            ll_val = np.log(p).sum() + np.log(np.exp(log_sf_q(q, lam, b, L[rc]))).sum()
            return -ll_val
        except: return 1e6
    res = minimize(nll, [1.2, np.log(1/16), np.log(2)], method='Nelder-Mead',
                   options={'maxiter':2000, 'xatol':1e-4})
    return res.x

def fit_frailty(L, R, rc):
    def nll(par):
        theta, ll, lb = par
        lam = np.exp(ll); b = np.exp(lb)
        if theta <= 0.01: theta = 0.01
        try:
            Sl = np.exp(log_sf_frailty(theta, lam, b, L[~rc]))
            Sr = np.exp(log_sf_frailty(theta, lam, b, R[~rc]))
            p = Sl - Sr
            if np.any(p <= 0): return 1e6
            ll_val = np.log(p).sum() + np.log(np.exp(log_sf_frailty(theta, lam, b, L[rc]))).sum()
            return -ll_val
        except: return 1e6
    res = minimize(nll, [0.5, np.log(1/16), np.log(2)], method='Nelder-Mead',
                   options={'maxiter':2000, 'xatol':1e-4})
    return res.x

def test_statistic(L, R, rc):
    """计算检验统计量: 两个模型拟合的生存曲线最大差值"""
    p_q = fit_qweibull(L, R, rc)
    p_f = fit_frailty(L, R, rc)
    q, ll_q, lb_q = p_q; lam_q = np.exp(ll_q); b_q = np.exp(lb_q)
    theta, ll_f, lb_f = p_f; lam_f = np.exp(ll_f); b_f = np.exp(lb_f)
    
    t_grid = np.linspace(0.1, 10, 200)
    S_q = np.exp(log_sf_q(q, lam_q, b_q, t_grid))
    S_f = np.exp(log_sf_frailty(theta, lam_f, b_f, t_grid))
    
    D = np.max(np.abs(S_q - S_f))
    nll_q = -fit_qweibull.__wrapped__(L, R, rc) if hasattr(fit_qweibull, '__wrapped__') else None
    return D, p_q, p_f

# 测试: 生成 q-Weibull 数据, 看检验是否不显著
rng = np.random.default_rng(42)
n = 500
q_true = 1.3; lam = 1/16; beta = 2.0

u = rng.random(n)
t = ((u**((q_true-1)/(q_true-2))-1)/((q_true-1)*lam))**(1/beta)
L = np.floor(t)*1.0; R = L + 1.0
rc = rng.random(n) < 0.37

D_obs, p_q, p_f = test_statistic(L, R, rc)
print(f"q-Weibull 数据 (q=1.3):")
print(f"  拟合 q-Weibull: q̂={p_q[0]:.3f}")
print(f"  拟合 frailty: θ̂={p_f[0]:.3f}")
print(f"  检验统计量 D = {D_obs:.4f}")
print()

# 生成 frailty 数据, 看检验
theta_true = 0.5
u = rng.random(n)
t = (((1/u)**theta_true - 1)/(theta_true*lam))**(1/beta)
L2 = np.floor(t)*1.0; R2 = L2 + 1.0
rc2 = rng.random(n) < 0.37

D_obs2, p_q2, p_f2 = test_statistic(L2, R2, rc2)
print(f"Frailty 数据 (θ=0.5):")
print(f"  拟合 q-Weibull: q̂={p_q2[0]:.3f}")
print(f"  拟合 frailty: θ̂={p_f2[0]:.3f}")
print(f"  检验统计量 D = {D_obs2:.4f}")
