# -*- coding: utf-8 -*-
"""q-Weibull 区间删失贝叶斯后验, 自适应 MH."""
import numpy as np, pandas as pd
import sys
sys.path.insert(0, r"G:\OPT\q-Weibull竞争风险论文\code")
from qweibull import log_sf

df = pd.read_csv(r"G:\OPT\Weibull发表策略\data\tooth24_clean.csv")
L = df['left'].values; R = df['right'].values; rc = df['censored'].values.astype(bool)

def log_post(theta):
    q, loglam, logb = theta
    if q <= 0.2 or q >= 1.95: return -np.inf
    lam = np.exp(loglam); b = np.exp(logb)
    sl = log_sf(q, lam, b, L)
    p_int = np.exp(sl[~rc]) - np.exp(log_sf(q, lam, b, R[~rc]))
    if np.any(p_int <= 0): return -np.inf
    ll = np.log(p_int).sum() + sl[rc].sum()
    lp = (-0.5*((q-1.0)/0.5)**2
          -0.5*((loglam-np.log(3e-5))/0.3)**2
          -0.5*((logb-np.log(6.3))/0.3)**2)
    return ll + lp

# 从 MLE 起步
theta = np.array([1.348, np.log(3e-5), np.log(6.3)])
rng = np.random.default_rng(42)
# 初始提议尺度
step = np.array([0.02, 0.02, 0.05])
N = 30000; burn = 10000
chain = np.zeros((N, 3)); accept = 0
for i in range(N):
    prop = theta + rng.normal(0, step)
    lr = log_post(prop) - log_post(theta)
    if np.log(rng.random()) < lr:
        theta = prop; accept += 1
    chain[i] = theta
    # 自适应: 每500步调整
    if i < burn and i % 500 == 0 and i > 0:
        ar = accept / (i+1)
        if ar > 0.35: step *= 1.2
        elif ar < 0.15: step *= 0.8
post = chain[burn:]
out = (f"accept(final) ~ {accept/N:.3f}\n"
       f"q: mean={post[:,0].mean():.3f} sd={post[:,0].std():.3f} "
       f"95%CI=[{np.percentile(post[:,0],2.5):.3f},{np.percentile(post[:,0],97.5):.3f}]\n"
       f"P(q>1)={(post[:,0]>1).mean():.4f}\n"
       f"beta: mean={np.exp(post[:,2]).mean():.3f} sd={np.exp(post[:,2]).std():.3f}\n"
       f"lam: mean={np.exp(post[:,1]).mean():.6f}")
print(out)
open(r"G:\OPT\Weibull发表策略\data\bayes_result.txt","w").write(out)
np.save(r"G:\OPT\Weibull发表策略\data\post_chain.npy", post)
