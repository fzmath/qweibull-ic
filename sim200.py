# -*- coding: utf-8 -*-
"""快速仿真: 200 reps, 无profile CI, 只报bias/RMSE/sd."""
import numpy as np
from scipy.optimize import minimize, minimize_scalar
import sys, time
sys.path.insert(0, r"G:\OPT\q-Weibull竞争风险论文\code")
from qweibull import log_sf

def nll_q(qq, L, R, rc):
    def nll(par):
        la, lb = par
        eta=np.exp(la); b=np.exp(lb); lam=eta**(-b)
        Sl = np.exp(log_sf(qq, lam, b, L))
        if np.any(Sl<=0): return 1e10
        ll = np.log(Sl[rc]).sum()
        p = Sl[~rc] - np.exp(log_sf(qq, lam, b, R[~rc]))
        if np.any(p<=0): return 1e10
        return -(ll + np.log(p).sum())
    best=None
    for e0,b0 in [(np.log(4.0),0.5),(np.log(3.0),0.8)]:
        r=minimize(nll,[e0,b0],method="Nelder-Mead",options={"xatol":1e-3,"maxiter":300})
        if best is None or r.fun<best.fun: best=r
    return best.fun

def fit(L, R, rc):
    grid=np.linspace(0.5,1.7,13)
    nlls=np.array([nll_q(q,L,R,rc) for q in grid])
    qg=grid[np.argmin(nlls)]
    r=minimize_scalar(lambda q: nll_q(q,L,R,rc),
                      bounds=(max(0.3,qg-0.12),min(1.9,qg+0.12)),
                      method="bounded",options={"xatol":1e-3})
    return r.x

def sim(q_true, eta, beta, n, rng, rc_rate=0.37):
    lam=eta**(-beta)
    u=rng.random(n)
    t=((1-(1-u)**((1-q_true)/(2-q_true)))/((1-q_true)*lam))**(1/beta)
    L=np.floor(t); R=L+1.0
    rc=rng.random(n)<rc_rate; R[rc]=np.nan
    return L,R,rc

results={}
for q_true in [0.8,1.0,1.2,1.4]:
    rng=np.random.default_rng(int(q_true*100))
    qs=[]; t0=time.time()
    for rep in range(200):
        L,R,rc=sim(q_true,4.0,2.0,1000,rng)
        qs.append(fit(L,R,rc))
    qs=np.array(qs)
    results[q_true]=(qs.mean()-q_true, np.sqrt(((qs-q_true)**2).mean()), qs.std())
    print(f"q={q_true}: bias={qs.mean()-q_true:.4f} RMSE={np.sqrt(((qs-q_true)**2).mean()):.4f} sd={qs.std():.4f} ({time.time()-t0:.0f}s)")

print("\n=== Final Table (n=1000, 200 reps) ===")
print(f"{'q':>5} {'bias':>8} {'RMSE':>8} {'sd':>8}")
for q,(b,r,s) in results.items():
    print(f"{q:>5.1f} {b:>8.4f} {r:>8.4f} {s:>8.4f}")
