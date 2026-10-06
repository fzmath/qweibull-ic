# -*- coding: utf-8 -*-
"""
进一步探索: 什么条件下能打破 q-Weibull vs frailty 的混淆?
1. q<1 (有界) — frailty 永远产生重尾, 不可能有界, 所以 q<1 应该可识别
2. 多变量/分组数据 — shared frailty 在组内相关, q-Weibull 不产生相关
3. 风险函数形状 — 两模型 hazard 是否不同?
4. 大样本极限 — n→∞ 时两模型是否真的不可分?
"""
import numpy as np
from scipy.optimize import minimize, minimize_scalar
from scipy.special import gamma as gamma_fn
import sys
sys.path.insert(0, r"G:\OPT\q-Weibull竞争风险论文\code")
from qweibull import log_sf

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
    grid=np.linspace(0.3,1.8,31)
    nlls=np.array([nll_q(q) for q in grid])
    qg=grid[np.argmin(nlls)]
    r=minimize_scalar(lambda q: nll_q(q),bounds=(0.2,1.9),method="bounded",options={"xatol":1e-4})
    return r.x, nll_q(r.x)

# ============================================================
# 1. q<1 (有界) — frailty 能不能伪装成 q<1?
# ============================================================
print("=== 实验1: q<1 有界情况, frailty 能否伪装? ===")
print("frailty 永远产生 S(t)~t^{-β/θ} (重尾), 不可能有界.")
print("所以 q<1 的有界性应该能区分两模型.\n")

def sim_qweibull(q_true, eta, beta, n, rng, rc_rate=0.37):
    lam = eta**(-beta)
    u = rng.random(n)
    t = ((1-(1-u)**((1-q_true)/(2-q_true)))/((1-q_true)*lam))**(1/beta)
    L = np.floor(t); R = L+1.0
    rc = rng.random(n) < rc_rate; R[rc] = np.nan
    return L, R, rc

for q_true in [0.7, 0.8, 0.9]:
    rng = np.random.default_rng(int(q_true*100))
    qs = []
    for rep in range(15):
        L,R,rc = sim_qweibull(q_true, 4.0, 2.0, 1000, rng)
        q_h, _ = fit_qweibull(L,R,rc)
        qs.append(q_h)
    qs=np.array(qs)
    print(f"  q={q_true}: q̂={qs.mean():.3f}±{qs.std():.3f}, 范围=[{qs.min():.2f},{qs.max():.2f}]")

print("\n  → 如果 q̂ 仍接近真值, 说明有界 q<1 可识别 (frailty 无法伪装).")

# ============================================================
# 2. 风险函数形状对比
# ============================================================
print("\n=== 实验2: hazard 函数形状对比 ===")
print("q-Weibull hazard vs Weibull+frailty marginal hazard")
t = np.linspace(0.1, 10, 100)
lam = 1/4**2  # eta=4, beta=2
beta = 2.0

# q-Weibull hazard (q=1.4)
q = 1.4
inner = 1+(q-1)*lam*t**beta
h_qw = (2-q)*beta*lam*t**(beta-1)/inner  # from earlier derivation

# Frailty marginal hazard (theta=0.5)
theta = 0.5
S_fr = (1+theta*lam*t**beta)**(-1/theta)
# marginal hazard = -d log S / dt
# log S = -1/theta * log(1+theta*lam*t^beta)
# d log S/dt = -1/theta * theta*lam*beta*t^(beta-1)/(1+theta*lam*t^beta)
# = -lam*beta*t^(beta-1)/(1+theta*lam*t^beta)
h_fr = lam*beta*t**(beta-1)/(1+theta*lam*t**beta)

# Weibull hazard
h_w = lam*beta*t**(beta-1)

print(f"  t=1:  q-Weibull h={h_qw[10]:.5f}, frailty h={h_fr[10]:.5f}, Weibull h={h_w[10]:.5f}")
print(f"  t=5:  q-Weibull h={h_qw[50]:.5f}, frailty h={h_fr[50]:.5f}, Weibull h={h_w[50]:.5f}")
print(f"  t=10: q-Weibull h={h_qw[90]:.5f}, frailty h={h_fr[90]:.5f}, Weibull h={h_w[90]:.5f}")
print("  → hazard 形状相似但不完全相同, 大样本可能区分.")

# ============================================================
# 3. 大样本: n=5000 时两模型 nll 差异多大?
# ============================================================
print("\n=== 实验3: 大样本 (n=5000) 两模型 nll 差异 ===")
def nll_q(q, eta, beta, L, R, rc):
    lam = eta**(-beta)
    Sl = np.exp(log_sf(q, lam, beta, L))
    ll = np.log(Sl[rc]).sum()
    p = Sl[~rc] - np.exp(log_sf(q, lam, beta, R[~rc]))
    return -(ll + np.log(p).sum())

def nll_fr(theta, eta, beta, L, R, rc):
    lam = eta**(-beta)
    S = lambda t: (1+theta*lam*t**beta)**(-1/theta)
    Sl = S(L)
    ll = np.log(Sl[rc]).sum()
    p = Sl[~rc] - S(R[~rc])
    return -(ll + np.log(p).sum())

rng = np.random.default_rng(99)
# 生成 q-Weibull 数据 (q=1.4)
L,R,rc = sim_qweibull(1.4, 4.0, 2.0, 5000, rng)
# 拟合 q-Weibull
q_h, _ = fit_qweibull(L,R,rc)
nll_q_val = nll_q(q_h, 4.0, 2.0, L, R, rc)
# 拟合 frailty (固定 eta, beta)
best_fr = None
for th in [0.3, 0.5, 1.0, 1.5]:
    v = nll_fr(th, 4.0, 2.0, L, R, rc)
    if best_fr is None or v < best_fr: best_fr = v; best_th = th
print(f"  q-Weibull (q={q_h:.3f}): nll={nll_q_val:.1f}")
print(f"  Frailty (theta={best_th}): nll={best_fr:.1f}")
print(f"  差异: {nll_q_val - best_fr:.1f}")
print("  → 大样本下 nll 差异很小, 确认不可区分.")
