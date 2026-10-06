# -*- coding: utf-8 -*-
"""
加强实验:
1. 可辨识性边界: 间隔宽度 w × 样本量 n 对 q 估计的影响
2. Frailty 对比: Weibull+frailty vs q-Weibull (真实数据)
"""
import numpy as np
from scipy.optimize import minimize, minimize_scalar
import sys, time
sys.path.insert(0, r"G:\OPT\q-Weibull竞争风险论文\code")
from qweibull import log_sf
import pandas as pd

# ============================================================
# Part 1: 可辨识性边界
# ============================================================
def sim_ic(q_true, eta_true, beta_true, n, w, rng, rc_rate=0.37):
    """w = 区间宽度(年). L = floor(t/w)*w, R = L+w."""
    lam_true = eta_true**(-beta_true)
    u = rng.random(n)
    t = ((1-(1-u)**((1-q_true)/(2-q_true)))/((1-q_true)*lam_true))**(1/beta_true)
    L = np.floor(t/w)*w; R = L + w
    rc = rng.random(n) < rc_rate; R[rc] = np.nan
    return L, R, rc

def fit_q(L, R, rc):
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
    grid=np.linspace(0.6,1.6,21)
    nlls=np.array([nll_q(q) for q in grid])
    qg=grid[np.argmin(nlls)]
    r=minimize_scalar(lambda q: nll_q(q),bounds=(max(0.3,qg-0.15),min(1.8,qg+0.15)),
                      method="bounded",options={"xatol":1e-4})
    return r.x

print("=== Part 1: 可辨识性边界 (q_true=1.35) ===")
print(f"{'w':>4} {'n':>5} {'bias':>8} {'RMSE':>8} {'sd':>8}")
results = {}
for w in [0.5, 1.0, 2.0, 3.0]:
    for n in [200, 500, 1000]:
        rng = np.random.default_rng(int(w*1000+n))
        qs = []
        for rep in range(30):
            L,R,rc = sim_ic(1.35, 4.0, 2.0, n, w, rng)
            qs.append(fit_q(L,R,rc))
        qs=np.array(qs)
        b=qs.mean()-1.35; rmse=np.sqrt(((qs-1.35)**2).mean()); sd=qs.std()
        results[(w,n)]=(b,rmse,sd)
        print(f"{w:>4.1f} {n:>5d} {b:>8.4f} {rmse:>8.4f} {sd:>8.4f}")

# ============================================================
# Part 2: Frailty 对比 (真实数据)
# ============================================================
print("\n=== Part 2: Frailty 模型对比 (tooth data) ===")
df = pd.read_csv(r"G:\OPT\Weibull发表策略\data\tooth24_clean.csv")
L = df["left"].values; R = df["right"].values
rc = df["censored"].values==1
print(f"n={len(df)}, interval={len(df)-rc.sum()}, right-censored={rc.sum()}")

# Weibull + gamma shared frailty (simple marginal: variance component)
# Use numerical integration over frailty distribution
from scipy.integrate import quad
from scipy.special import gamma as gamma_fn

def weibull_S(lam, beta, t):
    return np.exp(-lam*t**beta)

def nll_frailty(par, L, R, rc):
    """Weibull + gamma frailty with variance theta."""
    loglam, logbeta, logtheta = par
    lam=np.exp(loglam); beta=np.exp(logbeta); theta=np.exp(logtheta)
    def marg_S(t):
        # E_frailty[exp(-f*lam*t^beta)] = (1 + theta*lam*t^beta)^(-1/theta)
        return (1 + theta*lam*t**beta)**(-1/theta)
    Sl = marg_S(L)
    if np.any(Sl<=0): return 1e10
    ll = np.log(Sl[rc]).sum()
    p = Sl[~rc] - marg_S(R[~rc])
    if np.any(p<=0): return 1e10
    return -(ll + np.log(p).sum())

best=None
for t0 in [-8, 1.5, -2]:
    r=minimize(nll_frailty,[t0,1.5,-2],args=(L,R,rc),method="Nelder-Mead",options={"xatol":1e-5,"maxiter":3000})
    if best is None or r.fun<best.fun: best=r
lam_f=np.exp(best.x[0]); beta_f=np.exp(best.x[1]); theta_f=np.exp(best.x[2])
nll_f=best.fun
aic_f=2*(nll_f+3)
print(f"Weibull+frailty: lam={lam_f:.6f}, beta={beta_f:.3f}, theta={theta_f:.3f}")
print(f"  nll={nll_f:.2f}, AIC={aic_f:.2f}")
print(f"  q-Weibull (no cov): nll=5576.68, AIC=11159.4")
print(f"  Weibull (no frailty): nll=5597.0, AIC=11198.0")
