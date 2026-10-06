# -*- coding: utf-8 -*-
"""
深入探索: q-Weibull vs Weibull+frailty 的可辨识性问题.
1. 数学关系: 两者 survival function 的联系
2. 仿真: frailty 数据 → q-Weibull 拟合, q 会被估成多少?
3. 仿真: q-Weibull 数据 → frailty 拟合, θ 会被估成多少?
"""
import numpy as np
from scipy.optimize import minimize, minimize_scalar
import sys
sys.path.insert(0, r"G:\OPT\q-Weibull竞争风险论文\code")
from qweibull import log_sf

# ============================================================
# 0. 数学关系
# ============================================================
print("=== 数学关系 ===")
print("q-Weibull (q>1): S(t) = [1+(q-1)λt^β]^{-(q-2)/(q-1)}")
print("Gamma frailty:   S(t) = (1+θλt^β)^{-1/θ}")
print("匹配条件: θ=q-1 且 1/θ=(q-2)/(q-1) → q=3, θ=2")
print("即: q=3 时两模型精确等价; 其他 q 值时是同一家族的近似.\n")

# ============================================================
# 1. frailty 数据 → q-Weibull 拟合
# ============================================================
def sim_frailty(eta, beta, theta, n, rng, rc_rate=0.37):
    """Weibull+gamma frailty 生成区间删失数据."""
    lam = eta**(-beta)
    # gamma frailty: mean=1, var=theta
    Z = rng.gamma(1/theta, scale=theta, size=n)
    u = rng.random(n)
    # conditional: S(t|Z) = exp(-Z*lam*t^beta)
    t = (-np.log(u)/(Z*lam))**(1/beta)
    L = np.floor(t); R = L + 1.0
    rc = rng.random(n) < rc_rate; R[rc] = np.nan
    return L, R, rc

def fit_qweibull(L, R, rc):
    def nll_q(qq):
        def nll(par):
            eta, b = par; lam = eta**(-b)
            Sl = np.exp(log_sf(qq, lam, b, L))
            if np.any(Sl<=0): return 1e10
            ll = np.log(Sl[rc]).sum()
            p = Sl[~rc] - np.exp(log_sf(qq, lam, b, R[~rc]))
            if np.any(p<=0): return 1e10
            return -(ll + np.log(p).sum())
        best=None
        for e0,b0 in [(4.0,2.0),(3.0,1.5)]:
            r=minimize(nll,[e0,b0],method="Nelder-Mead",options={"xatol":1e-4,"maxiter":800})
            if best is None or r.fun<best.fun: best=r
        return best.fun
    grid=np.linspace(0.6,1.8,25)
    nlls=np.array([nll_q(q) for q in grid])
    qg=grid[np.argmin(nlls)]
    r=minimize_scalar(lambda q: nll_q(q),bounds=(0.3,1.9),method="bounded",options={"xatol":1e-4})
    return r.x

print("=== 实验1: frailty 数据 → q-Weibull MLE ===")
print("(真值: Weibull+frailty, 无真正的 q-deformation)")
for theta in [0.3, 0.5, 1.0]:
    rng = np.random.default_rng(42)
    qs = []
    for rep in range(20):
        L,R,rc = sim_frailty(4.0, 2.0, theta, 1000, rng)
        qs.append(fit_qweibull(L,R,rc))
    qs=np.array(qs)
    print(f"  theta={theta}: q̂ 均值={qs.mean():.3f}, sd={qs.std():.3f}, 范围=[{qs.min():.2f},{qs.max():.2f}]")

# ============================================================
# 2. q-Weibull 数据 → frailty 拟合
# ============================================================
def fit_frailty(L, R, rc):
    def nll(par):
        loglam, logbeta, logtheta = par
        lam=np.exp(loglam); beta=np.exp(logbeta); theta=np.exp(logtheta)
        if theta<=0.01 or theta>=10: return 1e10
        def marg_S(t):
            return (1 + theta*lam*t**beta)**(-1/theta)
        Sl = marg_S(L)
        if np.any(Sl<=0): return 1e10
        ll = np.log(Sl[rc]).sum()
        p = Sl[~rc] - marg_S(R[~rc])
        if np.any(p<=0): return 1e10
        return -(ll + np.log(p).sum())
    best=None
    for t0 in [-8,1.5,-2]:
        r=minimize(nll,[t0,1.5,-2],method="Nelder-Mead",options={"xatol":1e-4,"maxiter":2000})
        if best is None or r.fun<best.fun: best=r
    return np.exp(best.x[2])  # theta

def sim_qweibull(q_true, eta, beta, n, rng, rc_rate=0.37):
    lam = eta**(-beta)
    u = rng.random(n)
    t = ((1-(1-u)**((1-q_true)/(2-q_true)))/((1-q_true)*lam))**(1/beta)
    L = np.floor(t); R = L+1.0
    rc = rng.random(n) < rc_rate; R[rc] = np.nan
    return L, R, rc

print("\n=== 实验2: q-Weibull 数据 → frailty MLE ===")
print("(真值: q-Weibull, 无 frailty)")
for q_true in [1.2, 1.4, 1.6]:
    rng = np.random.default_rng(int(q_true*100))
    thetas = []
    for rep in range(20):
        L,R,rc = sim_qweibull(q_true, 4.0, 2.0, 1000, rng)
        thetas.append(fit_frailty(L,R,rc))
    thetas=np.array(thetas)
    print(f"  q={q_true}: θ̂ 均值={thetas.mean():.3f}, sd={thetas.std():.3f}, 范围=[{thetas.min():.2f},{thetas.max():.2f}]")

print("\n=== 结论 ===")
print("如果实验1中 q̂ 显著>1, 实验2中 θ̂ 显著>0,")
print("说明两模型在区间删失下不可区分 — 这是论文的核心发现.")
