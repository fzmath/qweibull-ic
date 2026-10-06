# -*- coding: utf-8 -*-
"""
q-Weibull 有界支持检验
检验: 数据是否需要有上界的分布?
"""
import numpy as np
from scipy.optimize import minimize
import sys
sys.path.insert(0, r"G:\OPT\q-Weibull竞争风险论文\code")
from qweibull import log_sf as log_sf_q

def log_sf_frailty(theta, lam, beta, t):
    return -1/theta * np.log(1 + theta*lam*t**beta)

def fit_qweibull(L, R, rc, q_fixed=None):
    def nll(par):
        if q_fixed is not None:
            q = q_fixed
            ll, lb = par
        else:
            q, ll, lb = par
        lam = np.exp(ll); b = np.exp(lb)
        if q <= 0.3 or q >= 2.5: return 1e6
        try:
            Sl = np.exp(log_sf_q(q, lam, b, L[~rc]))
            Sr = np.exp(log_sf_q(q, lam, b, R[~rc]))
            p = Sl - Sr
            if np.any(p <= 0): return 1e6
            ll_val = np.log(p).sum() + np.log(np.exp(log_sf_q(q, lam, b, L[rc]))).sum()
            return -ll_val
        except: return 1e6
    if q_fixed is not None:
        res = minimize(nll, [np.log(1/16), np.log(2)], method='Nelder-Mead',
                       options={'maxiter':2000, 'xatol':1e-4})
        return q_fixed, res.x[0], res.x[1], res.fun
    else:
        res = minimize(nll, [1.0, np.log(1/16), np.log(2)], method='Nelder-Mead',
                       options={'maxiter':2000, 'xatol':1e-4})
        return res.x[0], res.x[1], res.x[2], res.fun

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
    return res.x[0], res.x[1], res.x[2], res.fun

def bounded_support_test(L, R, rc):
    """
    检验统计量: 
    比较 q-Weibull (自由 q) 和 q=1 (Weibull, 无界) 的似然比
    如果 q<1 显著改善拟合, 说明数据需要有界支持
    """
    q_hat, ll_q, lb_q, nll_q = fit_qweibull(L, R, rc)
    _, ll_1, lb_1, nll_w = fit_qweibull(L, R, rc, q_fixed=1.0)
    
    T_max = None
    if q_hat < 1:
        lam = np.exp(ll_q); b = np.exp(lb_q)
        T_max = (1/((1-q_hat)*lam))**(1/b)
    
    # 似然比统计量
    LR = 2 * (nll_w - nll_q)  # q自由比q=1好多少
    
    return q_hat, T_max, LR, nll_q, nll_w

# ===== 测试 =====
rng = np.random.default_rng(42)
n = 500

print("="*60)
print("场景1: q=0.7 (真有界)")
print("="*60)
q_true = 0.7; lam = 1/16; beta = 2.0
u = rng.random(n)
t = ((u**((q_true-1)/(q_true-2))-1)/((q_true-1)*lam))**(1/beta)
T_max_true = (1/((1-q_true)*lam))**(1/beta)
print(f"真实 T_max = {T_max_true:.2f}")
print(f"最大观测 t = {np.max(t):.2f}")

L = np.floor(t)*1.0; R = L + 1.0
rc = rng.random(n) < 0.37

q_hat, T_max_hat, LR, nll_q, nll_w = bounded_support_test(L, R, rc)
print(f"q̂ = {q_hat:.3f}")
print(f"T_max 估计 = {T_max_hat:.2f}" if T_max_hat else "T_max: q̂>=1, 无界")
print(f"LR 统计量 = {LR:.3f}")
print(f"nll(q自由) = {nll_q:.1f}")
print(f"nll(q=1) = {nll_w:.1f}")
print()

print("="*60)
print("场景2: Weibull (q=1, 无界)")
print("="*60)
q_true = 1.0
u = rng.random(n)
t = (-np.log(u)/lam)**(1/beta)
print(f"最大观测 t = {np.max(t):.2f}")

L = np.floor(t)*1.0; R = L + 1.0
rc = rng.random(n) < 0.37

q_hat, T_max_hat, LR, nll_q, nll_w = bounded_support_test(L, R, rc)
print(f"q̂ = {q_hat:.3f}")
print(f"T_max 估计 = {T_max_hat:.2f}" if T_max_hat else "T_max: q̂>=1, 无界")
print(f"LR 统计量 = {LR:.3f}")
print(f"nll(q自由) = {nll_q:.1f}")
print(f"nll(q=1) = {nll_w:.1f}")
