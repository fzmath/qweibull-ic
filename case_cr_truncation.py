# -*- coding: utf-8 -*-
"""
真实案例扩展: 滚珠轴承数据 + 人为竞争截断, 看弱可辨识下先验借用/逃逸.

做法:
  现场批 = Lieblein-Zelen 轴承寿命.
  竞争截断 = 寿命超过 t_cut 的个体视为被第二原因截走(目标删失), 只保留 t<=t_cut 的目标失效.
  然后:
    A) 不同截断强度 t_cut 下, q 的 profile 似然有多宽;
    B) 加先验 N(q_hist, sigma^2), 比较"相容历史批(q=1.36)" vs "漂移历史批(q=0.6 有界)";
    C) 报告后验在 q>=1 (重尾) vs q<1 (有界) 的概率质量, 看是否"逃逸/翻向".
"""
import numpy as np
from scipy.optimize import minimize
import sys
sys.path.insert(0, r"G:\OPT\q-Weibull竞争风险论文\code")
from qweibull import log_pdf, log_sf

bearings = np.array([
    17.88,28.92,33.00,41.52,42.12,45.60,48.40,51.84,51.96,54.12,
    55.56,67.80,68.64,68.64,68.88,84.12,93.12,98.64,105.12,105.84,
    127.92,128.04,173.40
])
TRUE_Q = 1.36  # 全数据 MLE, 当作"真实"现场分布

def make_truncated(t_cut):
    """返回 (t, cause): cause=1 目标失效, cause=0 竞争删失."""
    t = np.where(bearings <= t_cut, bearings, t_cut)
    cause = np.where(bearings <= t_cut, 1, 0)
    return t, cause

def cens_nll(theta, t, cause):
    q, loglam, logb = theta
    if q <= 0.2 or q >= 1.95: return 1e10
    lam = np.exp(loglam); b = np.exp(logb)
    ll = 0.0
    for i in range(len(t)):
        if cause[i] == 1:
            ll += log_pdf(q, lam, b, t[i])
        else:
            ll += log_sf(q, lam, b, t[i])
    return -ll

def fit(t, cause, q_fixed=None):
    def obj(p):
        if q_fixed is None:
            q, loglam, logb = p
            if q<=0.2 or q>=1.95: return 1e10
        else:
            q = q_fixed; loglam, logb = p
        lam=np.exp(loglam); b=np.exp(logb)
        ll=0.0
        for i in range(len(t)):
            ll += log_pdf(q,lam,b,t[i]) if cause[i]==1 else log_sf(q,lam,b,t[i])
        return -ll
    if q_fixed is None:
        best=None
        for q0 in [0.6,1.0,1.3,1.6]:
            r=minimize(obj,[q0,np.log(1/80.0**2),np.log(2.0)],method="Nelder-Mead",options={"xatol":1e-5})
            if best is None or r.fun<best.fun: best=r
        return best.x[0], best.fun
    else:
        r=minimize(obj,[np.log(1/80.0**2),np.log(2.0)],method="Nelder-Mead",options={"xatol":1e-5})
        return q_fixed, r.fun

for t_cut in [80.0, 60.0]:
    t, cause = make_truncated(t_cut)
    n_fail = (cause==1).sum()
    n_cens = (cause==0).sum()
    print(f"\n===== 截断 t_cut={t_cut:.0f}: 目标失效 {n_fail}, 竞争删失 {n_cens} =====")
    q_mle, nll_mle = fit(t, cause)
    print(f"MLE q = {q_mle:.3f}  (全数据真实 q≈{TRUE_Q})")

    grid = np.linspace(0.3, 1.8, 31)
    prof=[]
    for q0 in grid:
        _, nl = fit(t, cause, q_fixed=q0)
        prof.append(-nl)
    prof=np.array(prof)
    prof-=np.nanmax(prof)
    print("profile 2*DeltaLL:")
    for q,p in zip(grid,prof):
        p=-np.inf if not np.isfinite(p) else p
        bar="#"*int(max(0,min(15,-p))/1.5)
        print(f"  q={q:.2f}  {-2*p:6.2f}  {bar}")

    # 先验借用: 相容 q_hist=1.36 vs 漂移 q_hist=0.6
    def post_mode(q_hist, sigma=0.3, delta=2.0):
        best=None
        for q0 in [0.5,0.9,1.2,1.6]:
            def obj(q):
                if q<=0.2 or q>=1.95: return 1e10
                _,nl=fit(t,cause,q_fixed=float(np.atleast_1d(q)[0]))
                return nl+delta*(q-q_hist)**2/(2*sigma**2)
            r=minimize(obj,q0,method="Nelder-Mead",options={"xatol":1e-5})
            if best is None or r.fun<best.fun: best=r
        return best.x[0]

    qp_compat = post_mode(1.36)
    qp_drift  = post_mode(0.6)
    print(f"后验众数: 相容历史批(q=1.36) -> {qp_compat:.3f}; 漂移历史批(q=0.6) -> {qp_drift:.3f}")

    # 后验网格: 重尾(q>=1) vs 有界(q<1) 的质量比
    logpost = prof - 0.0  # 无先验时
    # 加漂移先验 q_hist=0.6
    lp = prof - 2.0*(grid-0.6)**2/(2*0.3**2)
    lp -= np.nanmax(lp)
    w = np.exp(lp); w/=np.nansum(w)
    p_heavy = w[grid>=1.0].sum()
    print(f"加漂移先验(q_hist=0.6)后: P(q>=1) = {p_heavy:.2f}  (真实现象: 后验是否被推向有界)")
