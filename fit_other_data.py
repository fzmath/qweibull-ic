import numpy as np
from scipy.optimize import minimize
import pyreadr, sys
sys.path.insert(0, r"G:\OPT\q-Weibull竞争风险论文\code")
from qweibull import log_sf

def fit_qw_fr(L, R, rc, label):
    n = len(L)
    # q-Weibull
    def nll_qw(par):
        q, logeta, logb = par
        if q<0.3 or q>1.9: return 1e10
        eta=np.exp(logeta); b=np.exp(logb); lam=eta**(-b)
        Sl = np.exp(log_sf(q, lam, b, L))
        if np.any(Sl<=0): return 1e10
        ll = np.log(Sl[rc]).sum()
        p = Sl[~rc] - np.exp(log_sf(q, lam, b, R[~rc]))
        if np.any(p<=0): return 1e10
        return -(ll + np.log(p).sum())
    best=None
    for q0 in [1.0, 1.2, 1.4]:
        r=minimize(nll_qw,[q0,np.log(np.median(L[~rc])),0.5],method="Nelder-Mead",options={"xatol":1e-5,"maxiter":3000})
        if best is None or r.fun<best.fun: best=r
    q_h=best.x[0]; nll_qw=best.fun; k_qw=3

    # Weibull
    def nll_w(par):
        logeta, logb = par
        eta=np.exp(logeta); b=np.exp(logb); lam=eta**(-b)
        Sl = np.exp(log_sf(1.0, lam, b, L))
        ll = np.log(Sl[rc]).sum()
        p = Sl[~rc] - np.exp(log_sf(1.0, lam, b, R[~rc]))
        return -(ll + np.log(p).sum())
    r=minimize(nll_w,[np.log(np.median(L[~rc])),0.5],method="Nelder-Mead",options={"xatol":1e-5,"maxiter":2000})
    nll_w=r.fun; k_w=2

    # Frailty
    def nll_fr(par):
        logeta, logb, logth = par
        eta=np.exp(logeta); b=np.exp(logb); lam=eta**(-b); th=np.exp(logth)
        if th<0.01: return 1e10
        S = lambda t: (1+th*lam*t**b)**(-1/th)
        Sl = S(L)
        if np.any(Sl<=0): return 1e10
        ll = np.log(Sl[rc]).sum()
        p = Sl[~rc] - S(R[~rc])
        if np.any(p<=0): return 1e10
        return -(ll + np.log(p).sum())
    best2=None
    for th0 in [-1, 0, 1]:
        r=minimize(nll_fr,[np.log(np.median(L[~rc])),0.5,th0],method="Nelder-Mead",options={"xatol":1e-5,"maxiter":3000})
        if best2 is None or r.fun<best2.fun: best2=r
    th_h=np.exp(best2.x[2]); nll_fr=best2.fun; k_fr=3

    print(f"\n=== {label} (n={n}) ===")
    print(f"  Weibull:      nll={nll_w:.1f}, AIC={2*(nll_w+k_w):.1f}")
    print(f"  q-Weibull:    q={q_h:.3f}, nll={nll_qw:.1f}, AIC={2*(nll_qw+k_qw):.1f}")
    print(f"  Frailty:      theta={th_h:.3f}, nll={nll_fr:.1f}, AIC={2*(nll_fr+k_fr):.1f}")
    print(f"  qW vs fr nll差: {nll_qw-nll_fr:.1f}")

# Diabetes
d = pyreadr.read_r(r"G:\OPT\Weibull发表策略\data\icenReg\data\IR_diabetes.RData")
df = list(d.values())[0]
L = df["left"].values.astype(float).copy(); R = df["right"].values.astype(float).copy()
rc = np.zeros(len(L), dtype=bool)  # all interval
R[L==R] = L[L==R] + 0.01  # exact point → tiny interval
fit_qw_fr(L, R, rc, "Diabetes onset (n=731)")

# Mice
d2 = pyreadr.read_r(r"G:\OPT\Weibull发表策略\data\icenReg\data\miceData.RData")
df2 = list(d2.values())[0]
L2 = df2["l"].values.astype(float).copy(); R2 = df2["u"].values.astype(float).copy()
rc2 = np.zeros(len(L2), dtype=bool)
L2[L2==0] = 0.01
fit_qw_fr(L2, R2, rc2, "Mice tumor (n=144)")
