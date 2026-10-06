# -*- coding: utf-8 -*-
"""
q-Weibull 区间删失 MLE: 用 tooth24 数据看 q 是否偏离 1.

区间删失似然:
  区间 (L,R]:  S(L) - S(R)
  右删失 L:    S(L)
S(t) = [1-(1-q)*lam*t^beta]^((2-q)/(1-q)), q<1 有界, q=1 标准Weibull.
"""
import numpy as np, pandas as pd
from scipy.optimize import minimize
import sys
sys.path.insert(0, r"G:\OPT\q-Weibull竞争风险论文\code")
from qweibull import log_sf

df = pd.read_csv(r"G:\OPT\Weibull发表策略\data\tooth24_clean.csv")
L = df['left'].values
R = df['right'].values
rc = df['censored'].values  # 1=right-censored

def nll(theta):
    q, loglam, logb = theta
    if q <= 0.2 or q >= 1.95: return 1e10
    lam = np.exp(loglam); b = np.exp(logb)
    ll = 0.0
    for i in range(len(L)):
        sl = log_sf(q, lam, b, L[i])
        if rc[i] == 1:
            ll += sl
        else:
            sr = log_sf(q, lam, b, R[i])
            # S(L) - S(R) = exp(sl) - exp(sr)
            p = np.exp(sl) - np.exp(sr)
            if p <= 0: return 1e10
            ll += np.log(p)
    return -ll

# 初始值: 标准 Weibull (q=1)
best = None
for q0 in [0.7, 0.9, 1.0, 1.2, 1.4]:
    r = minimize(nll, [q0, np.log(1/4.0**2), np.log(2.0)],
                 method="Nelder-Mead", options={"xatol":1e-5, "maxiter":5000})
    if best is None or r.fun < best.fun:
        best = r

q_hat, loglam_hat, logb_hat = best.x
lam_hat = np.exp(loglam_hat); b_hat = np.exp(logb_hat)
nll_q = best.fun

# 固定 q=1 (标准Weibull)
def nll_weibull(theta):
    loglam, logb = theta
    return nll([1.0, loglam, logb])
r1 = minimize(nll_weibull, [np.log(1/4.0**2), np.log(2.0)],
              method="Nelder-Mead", options={"xatol":1e-5})
nll_w = r1.fun
lam_w = np.exp(r1.x[0]); b_w = np.exp(r1.x[1])

# LR test for q=1
lr = 2*(nll_w - nll_q)
out = (f"q-Weibull MLE: q={q_hat:.3f}, lam={lam_hat:.5f}, beta={b_hat:.3f}, nll={nll_q:.2f}\n"
       f"Weibull(q=1):   lam={lam_w:.5f}, beta={b_w:.3f}, nll={nll_w:.2f}\n"
       f"LR test (q=1): 2*(nll_W - nll_q) = {lr:.2f}  "
       f"({'拒绝q=1' if lr > 3.84 else '不拒绝q=1'})")
print(out)
open(r"G:\OPT\Weibull发表策略\data\mle_result.txt","w").write(out)
