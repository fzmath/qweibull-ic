# -*- coding: utf-8 -*-
"""
补 log-logistic 和 log-normal 对比
补真实数据的置信区间
"""
import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm
import pandas as pd
import sys
sys.path.insert(0, r"G:\OPT\q-Weibull竞争风险论文\code")
from qweibull import log_sf as log_sf_q

def log_sf_frailty(theta, lam, beta, t):
    return -1/theta * np.log(1 + theta*lam*t**beta)

def fit_model(L, R, rc, model='qweibull'):
    """拟合各种模型"""
    def nll_qweibull(par):
        q, ll, lb = par
        lam = np.exp(ll); b = np.exp(lb)
        if q <= 0.3 or q >= 1.95: return 1e6
        try:
            Sl = np.exp(log_sf_q(q, lam, b, L[~rc]))
            Sr = np.exp(log_sf_q(q, lam, b, R[~rc]))
            p = Sl - Sr
            if np.any(p <= 0): return 1e6
            return -np.log(p).sum() - np.log(np.exp(log_sf_q(q, lam, b, L[rc]))).sum()
        except: return 1e6
    
    def nll_frailty(par):
        theta, ll, lb = par
        lam = np.exp(ll); b = np.exp(lb)
        if theta <= 0.01: theta = 0.01
        try:
            Sl = np.exp(log_sf_frailty(theta, lam, b, L[~rc]))
            Sr = np.exp(log_sf_frailty(theta, lam, b, R[~rc]))
            p = Sl - Sr
            if np.any(p <= 0): return 1e6
            return -np.log(p).sum() - np.log(np.exp(log_sf_frailty(theta, lam, b, L[rc]))).sum()
        except: return 1e6
    
    def nll_weibull(par):
        ll, lb = par
        lam = np.exp(ll); b = np.exp(lb)
        try:
            Sl = np.exp(-lam*L[~rc]**b)
            Sr = np.exp(-lam*R[~rc]**b)
            p = Sl - Sr
            if np.any(p <= 0): return 1e6
            return -np.log(p).sum() - (-lam*L[rc]**b).sum()
        except: return 1e6
    
    def nll_loglogistic(par):
        # log-logistic: S(t) = 1/(1 + (t/scale)^shape)
        ls, ls2 = par
        s = np.exp(ls); sh = np.exp(ls2)
        try:
            Sl = 1/(1 + (L[~rc]/s)**sh)
            Sr = 1/(1 + (R[~rc]/s)**sh)
            p = Sl - Sr
            if np.any(p <= 0): return 1e6
            return -np.log(p).sum() - np.log(1/(1 + (L[rc]/s)**sh)).sum()
        except: return 1e6
    
    def nll_lognormal(par):
        # log-normal: S(t) = 1 - Phi((ln(t)-mu)/sigma)
        mu, ls = par
        s = np.exp(ls)
        from scipy.stats import norm
        try:
            Sl = 1 - norm.cdf((np.log(L[~rc]) - mu)/s)
            Sr = 1 - norm.cdf((np.log(R[~rc]) - mu)/s)
            p = Sl - Sr
            if np.any(p <= 0): return 1e6
            return -np.log(p).sum() - np.log(1 - norm.cdf((np.log(L[rc]) - mu)/s)).sum()
        except: return 1e6
    
    if model == 'qweibull':
        r = minimize(nll_qweibull, [1.2, np.log(0.1), np.log(2)], method='Nelder-Mead',
                     options={'maxiter':2000, 'xatol':1e-4})
        return r.fun, r.x, 3
    elif model == 'frailty':
        r = minimize(nll_frailty, [0.5, np.log(0.1), np.log(2)], method='Nelder-Mead',
                     options={'maxiter':2000, 'xatol':1e-4})
        return r.fun, r.x, 3
    elif model == 'weibull':
        r = minimize(nll_weibull, [np.log(0.1), np.log(2)], method='Nelder-Mead',
                     options={'maxiter':2000, 'xatol':1e-4})
        return r.fun, r.x, 2
    elif model == 'loglogistic':
        r = minimize(nll_loglogistic, [np.log(5), np.log(2)], method='Nelder-Mead',
                     options={'maxiter':2000, 'xatol':1e-4})
        return r.fun, r.x, 2
    elif model == 'lognormal':
        r = minimize(nll_lognormal, [np.log(5), np.log(0.5)], method='Nelder-Mead',
                     options={'maxiter':2000, 'xatol':1e-4})
        return r.fun, r.x, 2

# ===== 牙齿数据 =====
print("="*60)
print("牙齿萌出数据 - 多模型对比")
print("="*60)
df = pd.read_csv(r"G:\OPT\JSCS-WFC20261005\data\tooth24_clean.csv")
L = df['left'].values.astype(float)
R = df['right'].values.astype(float)
rc = df['censored'].values.astype(bool)

models = ['weibull', 'qweibull', 'frailty', 'loglogistic', 'lognormal']
results = {}
for m in models:
    nll, params, k = fit_model(L, R, rc, m)
    aic = 2*nll + 2*k
    results[m] = {'nll': nll, 'aic': aic, 'k': k, 'params': params}
    print(f"{m:15s}: nll={nll:.1f}, AIC={aic:.1f}, k={k}")

print()
print("结论: q-Weibull 和 frailty nll 相同; log-logistic 和 log-normal 是额外的竞争者")
