# -*- coding: utf-8 -*-
"""仿真 v4: 网格+局部优化, 50 reps, 快速稳健."""
import numpy as np
from scipy.optimize import minimize_scalar, minimize
import sys, time
sys.path.insert(0, r"G:\OPT\q-Weibull竞争风险论文\code")
from qweibull import log_sf

def profile_nll(q, L, R, rc):
    """固定 q, 优化 eta, beta."""
    def nll(par):
        eta, b = par
        lam = eta**(-b)
        Sl = np.exp(log_sf(q, lam, b, L))
        if np.any(Sl<=0): return 1e10
        ll = np.log(Sl[rc]).sum()
        p = Sl[~rc] - np.exp(log_sf(q, lam, b, R[~rc]))
        if np.any(p<=0): return 1e10
        return -(ll + np.log(p).sum())
    best=None
    for e0,b0 in [(4.0,2.0),(3.0,1.5),(5.0,2.5)]:
        r=minimize(nll,[e0,b0],method="Nelder-Mead",options={"xatol":1e-4,"maxiter":1000})
        if best is None or r.fun<best.fun: best=r
    return best.fun

def fit(L, R, rc):
    grid=np.linspace(0.5,1.5,21)
    nlls=np.array([profile_nll(q,L,R,rc) for q in grid])
    q_best=grid[np.argmin(nlls)]
    # 局部细化
    r=minimize_scalar(lambda q: profile_nll(q,L,R,rc),bounds=(max(0.3,q_best-0.15),min(1.7,q_best+0.15)),
                      method="bounded",options={"xatol":1e-4})
    return r.x

def sim_one(q_true, eta_true, beta_true, n, rng, rc_rate=0.37):
    lam_true=eta_true**(-beta_true)
    u=rng.random(n)
    t=((1-(1-u)**((1-q_true)/(2-q_true)))/((1-q_true)*lam_true))**(1/beta_true)
    L=np.floor(t); R=L+1.0
    rc=rng.random(n)<rc_rate; R[rc]=np.nan
    return L,R,rc

results={}
for q_true in [0.8,1.0,1.2,1.4]:
    rng=np.random.default_rng(int(q_true*100))
    qs=[]; t0=time.time()
    for rep in range(50):
        L,R,rc=sim_one(q_true,4.0,2.0,1000,rng)
        qs.append(fit(L,R,rc))
    qs=np.array(qs)
    lo,hi=np.percentile(qs,[2.5,97.5])
    results[q_true]=(qs.mean()-q_true,np.sqrt(((qs-q_true)**2).mean()),qs.std(),lo,hi)
    print(f"q={q_true}: bias={qs.mean()-q_true:.4f}, RMSE={np.sqrt(((qs-q_true)**2).mean()):.4f}, "
          f"sd={qs.std():.4f}, 95%=[{lo:.3f},{hi:.3f}] ({time.time()-t0:.0f}s)")

print("\n=== Simulation (n=1000, 50 reps, interval-censored) ===")
print(f"{'q_true':>7} {'bias':>8} {'RMSE':>8} {'sd':>8} {'2.5%':>7} {'97.5%':>7}")
for q,(b,r,s,lo,hi) in results.items():
    print(f"{q:>7.1f} {b:>8.4f} {r:>8.4f} {s:>8.4f} {lo:>7.3f} {hi:>7.3f}")
