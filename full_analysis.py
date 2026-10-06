# -*- coding: utf-8 -*-
"""
q-Weibull 区间删失: 协变量模型 + 模型对比 + PPC + 仿真.
AFT: eta_i = eta0 * exp(g1*girl_i + g2*dmf_i)
"""
import numpy as np, pandas as pd
from scipy.optimize import minimize
import sys
sys.path.insert(0, r"G:\OPT\q-Weibull竞争风险论文\code")

df = pd.read_csv(r"G:\OPT\Weibull发表策略\data\tooth24_clean.csv")
L = df['left'].values; R = df['right'].values; rc = df['censored'].values.astype(bool)
girl = df['girl'].values.astype(float); dmf = df['dmf_pos'].values.astype(float)
n = len(L)

def qweibull_S(q, lam, beta, t):
    inner = 1.0 - (1.0-q)*lam*t**beta
    return np.where(inner>0, inner**((2-q)/(1-q)), 0.0)

# ---------- 1. q-Weibull AFT ----------
def nll_qw(theta, cov=True):
    q, logeta0, logb, g1, g2 = theta
    if q<=0.2 or q>=1.95: return 1e10
    b = np.exp(logb)
    eta = np.exp(logeta0) * np.exp(g1*girl + g2*dmf) if cov else np.full(n, np.exp(logeta0))
    lam = eta**(-b)
    Sl = qweibull_S(q, lam, b, L)
    if not np.all(Sl>0): return 1e10
    ll_rc = np.log(Sl[rc]).sum()
    Rf = R[~rc]; S_l = Sl[~rc]
    Sr = qweibull_S(q, lam[~rc], b, Rf)
    p = S_l - Sr
    if np.any(p<=0): return 1e10
    return -(ll_rc + np.log(p).sum())

# 无协变量
r0 = minimize(lambda t: nll_qw(t, False), [1.348, np.log(4.0), np.log(6.3), 0, 0],
              method="Nelder-Mead", options={"xatol":1e-6,"maxiter":20000})
# 有协变量
r1 = minimize(lambda t: nll_qw(t, True), [1.348, np.log(4.0), np.log(6.3), 0.1, 0.0],
              method="Nelder-Mead", options={"xatol":1e-6,"maxiter":20000})
nll_qw0, nll_qw1 = r0.fun, r1.fun
aic_qw0 = 2*3 + 2*nll_qw0; aic_qw1 = 2*5 + 2*nll_qw1
print(f"q-Weibull 无协变量: nll={nll_qw0:.1f} AIC={aic_qw0:.1f}")
print(f"q-Weibull 有协变量: nll={nll_qw1:.1f} AIC={aic_qw1:.1f}")
print(f"  q={r1.x[0]:.3f}, eta0={np.exp(r1.x[1]):.2f}, beta={np.exp(r1.x[2]):.3f}")
print(f"  g_girl={r1.x[3]:.3f}, g_dmf={r1.x[4]:.3f}")
lr_cov = 2*(nll_qw0 - nll_qw1)
print(f"  协变量 LR={lr_cov:.2f} (chi2 df=2, 显著={lr_cov>5.99})")

# ---------- 2. 其他模型 ----------
def nll_weibull(theta):
    loglam, logb = theta; b=np.exp(logb); lam=np.exp(loglam)
    Sl = np.exp(-lam*L**b)
    ll = np.log(Sl[rc]).sum()
    p = Sl[~rc] - np.exp(-lam*R[~rc]**b)
    if np.any(p<=0): return 1e10
    return -(ll + np.log(p).sum())

def nll_lognormal(theta):
    mu, logsig = theta; sig=np.exp(logsig)
    from scipy.stats import norm
    Sl = 1-norm.cdf(np.log(L), mu, sig)
    ll = np.log(Sl[rc]).sum()
    p = Sl[~rc] - (1-norm.cdf(np.log(R[~rc]), mu, sig))
    if np.any(p<=0): return 1e10
    return -(ll + np.log(p).sum())

def nll_loglogistic(theta):
    logalpha, logg = theta; a=np.exp(logalpha); g=np.exp(logg)
    Sl = 1/(1+(L/a)**g)
    ll = np.log(Sl[rc]).sum()
    p = Sl[~rc] - 1/(1+(R[~rc]/a)**g)
    if np.any(p<=0): return 1e10
    return -(ll + np.log(p).sum())

rw = minimize(nll_weibull, [np.log(1/4.0**2), np.log(5.3)], method="Nelder-Mead", options={"xatol":1e-6,"maxiter":20000})
rn = minimize(nll_lognormal, [np.log(4.5), 0.0], method="Nelder-Mead", options={"xatol":1e-6,"maxiter":20000})
rl = minimize(nll_loglogistic, [np.log(4.5), 0.0], method="Nelder-Mead", options={"xatol":1e-6,"maxiter":20000})

models = [
    ("Weibull", 2, rw.fun),
    ("Log-normal", 2, rn.fun),
    ("Log-logistic", 2, rl.fun),
    ("q-Weibull (no cov)", 3, nll_qw0),
    ("q-Weibull (with cov)", 5, nll_qw1),
]
print("\n=== 模型对比 ===")
print(f"{'Model':<22}{'k':>3}{'nll':>10}{'AIC':>10}")
for name,k,nll in models:
    print(f"{name:<22}{k:>3}{nll:>10.1f}{2*k+2*nll:>10.1f}")

# ---------- 3. PPC: 后验预测复制区间中点分布 ----------
rng = np.random.default_rng(0)
q_hat, logeta0, logb, g1, g2 = r1.x
b=np.exp(logb); eta=np.exp(logeta0)*np.exp(g1*girl+g2*dmf); lam=eta**(-b)
# 观测: 区间中点
obs_mid = np.where(rc, L, (L+R)/2)
# 预测: 从 S(t|q,lam,b) 反解
def predict_t(q, lam_i, b):
    u = rng.random(len(lam_i))
    # S(t)=u => t = [(1-u**((1-q)/(2-q)))/((1-q)lam)]^(1/b)
    return ((1-u**((1-q)/(2-q)))/((1-q)*lam_i))**(1/b)
pred_mid = predict_t(q_hat, lam, b)
print(f"\n=== PPC ===")
print(f"观测区间中点: mean={obs_mid.mean():.2f}, sd={obs_mid.std():.2f}")
print(f"预测寿命:     mean={pred_mid.mean():.2f}, sd={pred_mid.std():.2f}")
print(f"男孩: 观测mid mean={obs_mid[girl==0].mean():.2f}, 预测={predict_t(q_hat,lam[girl==0],b).mean():.2f}")
print(f"女孩: 观测mid mean={obs_mid[girl==1].mean():.2f}, 预测={predict_t(q_hat,lam[girl==1],b).mean():.2f}")

# ---------- 4. 仿真验证 ----------
print("\n=== 仿真验证 (q=1.35, n=1000, R=20) ===")
true_q, true_beta, true_eta = 1.35, 6.3, 4.0
true_lam = true_eta**(-true_beta)
qs=[]
for rep in range(20):
    u = rng.random(1000)
    t = ((1-u**((1-true_q)/(2-true_q)))/((1-true_q)*true_lam))**(1/true_beta)
    # 区间删失: 每年体检
    Ls = np.floor(t); Rs = Ls+1
    rc_sim = rng.random(1000) < 0.37
    Rs[rc_sim] = np.nan
    def nll_s(th):
        q, lle, llb = th
        if q<=0.2 or q>=1.95: return 1e10
        bb=np.exp(llb); ll2=np.exp(lle)
        lam_i = ll2**(-bb)
        Sl = qweibull_S(q, lam_i, bb, Ls)
        if np.any(Sl<=0): return 1e10
        ll = np.log(Sl[rc_sim]).sum()
        p = Sl[~rc_sim] - qweibull_S(q, lam_i, bb, Rs[~rc_sim])
        if np.any(p<=0): return 1e10
        return -(ll+np.log(p).sum())
    rs = minimize(nll_s, [1.2, np.log(true_eta), np.log(true_beta)], method="Nelder-Mead", options={"xatol":1e-5,"maxiter":10000})
    qs.append(rs.x[0])
qs=np.array(qs)
print(f"q 恢复: 真值=1.35, 均值={qs.mean():.3f}, sd={qs.std():.3f}, 95%CI覆盖={np.mean((qs-1.96*0.02<1.35)&(qs+1.96*0.02>1.35))*100:.0f}%")
