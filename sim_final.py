# -*- coding: utf-8 -*-
"""正式仿真 v5: 快速, 100 reps, 修q=1, 覆盖率."""
import numpy as np
from scipy.optimize import minimize, minimize_scalar
import sys, time
sys.path.insert(0, r"G:\OPT\q-Weibull竞争风险论文\code")
from qweibull import log_sf

def nll_q_fixed(qq, L, R, rc):
    """固定q, 优化eta, beta."""
    def nll(par):
        logeta, logb = par
        eta=np.exp(logeta); b=np.exp(logb); lam=eta**(-b)
        Sl = np.exp(log_sf(qq, lam, b, L))
        if np.any(Sl<=0): return 1e10
        ll = np.log(Sl[rc]).sum()
        p = Sl[~rc] - np.exp(log_sf(qq, lam, b, R[~rc]))
        if np.any(p<=0): return 1e10
        return -(ll + np.log(p).sum())
    best=None
    for e0,b0 in [(np.log(4.0),0.5),(np.log(3.0),0.8),(np.log(5.0),0.3)]:
        r=minimize(nll,[e0,b0],method="Nelder-Mead",options={"xatol":1e-4,"maxiter":600})
        if best is None or r.fun<best.fun: best=r
    return best.fun

def fit_q(L, R, rc):
    grid=np.linspace(0.4,1.8,29)
    nlls=np.array([nll_q_fixed(q,L,R,rc) for q in grid])
    qg=grid[np.argmin(nlls)]
    r=minimize_scalar(lambda q: nll_q_fixed(q,L,R,rc),
                      bounds=(max(0.2,qg-0.1),min(1.9,qg+0.1)),
                      method="bounded",options={"xatol":1e-4})
    q_hat=r.x; nll_min=r.fun
    # profile CI
    grid2=np.linspace(max(0.2,q_hat-0.3),min(1.9,q_hat+0.3),21)
    nlls2=np.array([nll_q_fixed(q,L,R,rc) for q in grid2])
    in_ci = grid2[nlls2 - nll_min < 1.92]
    lo,hi = (in_ci[0],in_ci[-1]) if len(in_ci)>0 else (np.nan,np.nan)
    return q_hat, lo, hi

def sim(q_true, eta, beta, n, rng, rc_rate=0.37):
    lam=eta**(-beta)
    u=rng.random(n)
    t=((1-(1-u)**((1-q_true)/(2-q_true)))/((1-q_true)*lam))**(1/beta)
    L=np.floor(t); R=L+1.0
    rc=rng.random(n)<rc_rate; R[rc]=np.nan
    return L,R,rc

print("q_true  bias  RMSE   sd   cover")
results={}
for q_true in [0.8,1.0,1.2,1.4]:
    rng=np.random.default_rng(int(q_true*100))
    qs=[]; cover=0
    t0=time.time()
    for rep in range(100):
        L,R,rc=sim(q_true,4.0,2.0,1000,rng)
        q_h,lo,hi=fit_q(L,R,rc)
        qs.append(q_h)
        if lo<q_true<hi: cover+=1
    qs=np.array(qs)
    b=qs.mean()-q_true; rmse=np.sqrt(((qs-q_true)**2).mean()); sd=qs.std()
    results[q_true]=(b,rmse,sd,cover/100)
    print(f"q={q_true}: bias={b:.4f} RMSE={rmse:.4f} sd={sd:.4f} cover={cover/100:.3f} ({time.time()-t0:.0f}s)")

print("\n=== Table: q-Weibull MLE recovery (n=1000, 100 reps) ===")
print(f"{'q':>5} {'bias':>8} {'RMSE':>8} {'sd':>8} {'cover95':>8}")
for q,(b,r,s,c) in results.items():
    print(f"{q:>5.1f} {b:>8.4f} {r:>8.4f} {s:>8.4f} {c:>8.3f}")
