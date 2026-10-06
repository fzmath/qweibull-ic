# -*- coding: utf-8 -*-
"""
最小探针: q<1 有界 + 强竞争截断(端点未观测) 下, 历史先验借用会不会把后验拉偏/逃逸到 q>=1.

机制:
  目标原因 q=0.6 (有界), 竞争原因 q=1 小尺度 -> 多数个体早死, 目标寿命尾部观测不到.
  T_max 由 (1-q) 决定; 截断强时 max(t) << T_max.
探针指标:
  1) profile log-likelihood(q) 在 q in [0.3, 1.5] 上有多宽/多平;
  2) 加 N(q_hist, sigma^2) 历史先验(强度 delta)后, 后验众数 q_map;
  3) q_hist = 0.6(相容) / 1.0(漂移) / 1.3(重尾) 三种借用情形.
"""
import numpy as np
from scipy.optimize import minimize_scalar
import sys, os
sys.path.insert(0, r"G:\OPT\q-Weibull竞争风险论文\code")
from qweibull import log_pdf, log_sf, support_upper

rng = np.random.default_rng(20261005)

def sim_cr(n, q_t, eta_t, beta_t, q_c, eta_c, beta_c):
    """k=2 竞争风险: cause1=目标, cause2=竞争. 返回 (t, cause), cause 0=删失."""
    lam_t = eta_t ** (-beta_t)
    lam_c = eta_c ** (-beta_c)
    t1 = rng.weibull(beta_t, n) * eta_t  # 近似 q-Weibull 难直接采样, 用反函数法:
    # 用逆 CDF 采样 q-Weibull: u~U(0,1), F=u => t=[(1-u**((1-q)/(2-q)))/( (1-q) lam)]**(1/beta)
    u = rng.random(n)
    F = u
    # t target (q-Weibull)
    base = 1.0 - (1.0 - F) ** ((1.0 - q_t) / (2.0 - q_t))
    t1 = (base / ((1.0 - q_t) * lam_t)) ** (1.0 / beta_t)
    # competing (standard Weibull q=1)
    t2 = eta_c * (-np.log(rng.random(n))) ** (1.0 / beta_c)
    t = np.minimum(t1, t2)
    cause = np.where(t1 <= t2, 1, 2)
    return t, cause

# 真值: 目标 q=0.6 (有界), eta=1000, beta=2
q_t, eta_t, beta_t = 0.6, 1000.0, 2.0
lam_t = eta_t ** (-beta_t)
Tmax = support_upper(q_t, lam_t, beta_t)
# 竞争原因: q=1, eta 很小 -> 几乎所有个体早早被竞争原因截走, 目标尾部完全观测不到
q_c, eta_c, beta_c = 1.0, 120.0, 2.0

print(f"True target q={q_t}, eta={eta_t}, beta={beta_t}, T_max={Tmax:.1f}")

rows = []
for rep in range(30):
    t, cause = sim_cr(200, q_t, eta_t, beta_t, q_c, eta_c, beta_c)
    # 只看目标原因失效 + 竞争原因造成的删失信息(竞争风险似然)
    n_target = (cause == 1).sum()
    n_comp = (cause == 2).sum()
    tmax_obs = t.max()
    frac_endpoint = tmax_obs / Tmax

    # profile loglik for q: 固定 eta_t=1000, beta_t=2, 竞争原因参数固定真值
    lam_c_true = eta_c ** (-beta_c)
    def prof_ll(q):
        ll = 0.0
        for i in range(len(t)):
            ti = t[i]; ci = cause[i]
            if ci == 1:  # target failure
                ll += log_pdf(q, lam_t, beta_t, ti) + log_sf(1.0, lam_c_true, beta_c, ti)
            else:  # competing failure: target survives
                ll += log_sf(1.0, lam_c_true, beta_c, ti)
        return ll

    # 找 MLE q
    grid = np.linspace(0.3, 1.5, 121)
    lls = np.array([prof_ll(q) for q in grid])
    q_mle = grid[np.argmax(lls)]

    # 后验众数: ll + log prior, 先验 N(q_hist, sigma^2), 强度 delta
    def post_mode(q_hist, sigma, delta):
        def negpost(q):
            if q <= 0.2 or q >= 1.9:
                return 1e10
            return -(prof_ll(q) - delta * (q - q_hist) ** 2 / (2 * sigma ** 2))
        res = minimize_scalar(negpost, bounds=(0.25, 1.6), method="bounded",
                              options={"xatol": 1e-4})
        return res.x

    q_map_compat = post_mode(0.6, 0.3, 1.0)   # 历史批相容 q_hist=0.6
    q_map_shift  = post_mode(1.0, 0.3, 1.0)   # 历史批漂移 q_hist=1.0 (Weibull)
    q_map_heavy  = post_mode(1.3, 0.3, 1.0)   # 历史批重尾 q_hist=1.3
    rows.append([n_target, n_comp, tmax_obs, frac_endpoint, q_mle,
                 q_map_compat, q_map_shift, q_map_heavy])

rows = np.array(rows)
print("\n=== 30 次重复 (n=200, 强竞争截断) ===")
print(f"目标原因失效数: 均值 {rows[:,0].mean():.0f} (竞争原因 {rows[:,1].mean():.0f})")
print(f"max(t)/T_max: 均值 {rows[:,3].mean():.3f}  -> 端点{'几乎未观测' if rows[:,3].mean()<0.5 else '部分观测'}")
print(f"q_MLE:            均值 {rows[:,4].mean():.3f}  std {rows[:,4].std():.3f}")
print(f"q_MAP(历史 q=0.6): 均值 {rows[:,5].mean():.3f}  std {rows[:,5].std():.3f}")
print(f"q_MAP(历史 q=1.0): 均值 {rows[:,6].mean():.3f}  std {rows[:,6].std():.3f}")
print(f"q_MAP(历史 q=1.3): 均值 {rows[:,7].mean():.3f}  std {rows[:,7].std():.3f}")
print(f"\n后验逃逸到 q>=1 的比例:")
print(f"  q_MLE:            {(rows[:,4]>=1.0).mean()*100:.0f}%")
print(f"  q_MAP(q_hist=1.0): {(rows[:,6]>=1.0).mean()*100:.0f}%")
print(f"  q_MAP(q_hist=1.3): {(rows[:,7]>=1.0).mean()*100:.0f}%")

# 单次 profile 似然形状
print("\n=== 单次 profile log-likelihood 形状 ===")
best = int(np.argmax(lls))
for q, ll in zip(grid[::10], lls[::10]):
    print(f"  q={q:.2f}  ll={ll:8.1f}  " + ("#"*int((lls.max()-ll)/5+1)))
print(f"  MLE q={q_mle:.2f}, 真值 q=0.6")
