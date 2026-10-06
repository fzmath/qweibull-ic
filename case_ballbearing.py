# -*- coding: utf-8 -*-
"""
真实案例原型: 滚珠轴承寿命数据 (Lieblein & Zelen, 1956) 作为现场批.
目标: 在真实小样本上展示 q-Weibull 弱可辨识 + 历史先验借用的效果.

数据: 23 个球轴承疲劳寿命 (百万转), 无删失.
做法:
  1) 对现场数据做 q-Weibull profile 似然, 看 q 的识别有多弱;
  2) 加 N(q_hist, sigma^2) 先验, q_hist = 0.8/1.0/1.2, 看后验众数/区间怎么动;
  3) 构造一个历史批 (早期型号, q_hist=1.2 重尾), 用幂先验借用, 看是否把现场结论拉偏.
"""
import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm
import sys
sys.path.insert(0, r"G:\OPT\q-Weibull竞争风险论文\code")
from qweibull import log_pdf, support_upper

# --- 真实数据: Lieblein-Zelen 滚珠轴承寿命 (百万转) ---
bearings = np.array([
    17.88, 28.92, 33.00, 41.52, 42.12, 45.60, 48.40, 51.84, 51.96, 54.12,
    55.56, 67.80, 68.64, 68.64, 68.88, 84.12, 93.12, 98.64, 105.12, 105.84,
    127.92, 128.04, 173.40
])
n = len(bearings)
print(f"现场批: 滚珠轴承 n={n}, 寿命 {bearings.min():.1f}-{bearings.max():.1f} 百万转")

def nll_qeb(theta, t):
    q, loglam, logb = theta
    if q <= 0.2 or q >= 1.95:
        return 1e10
    lam = np.exp(loglam); b = np.exp(logb)
    ll = log_pdf(q, lam, b, t).sum()
    return -ll

def fit_qweibull(t, q_fixed=None):
    """返回 q, lam, beta, nll. q_fixed 给定时做 profile."""
    def obj(params):
        if q_fixed is None:
            q = params[0]; loglam = params[1]; logb = params[2]
            if q <= 0.2 or q >= 1.95: return 1e10
        else:
            loglam, logb = params
            q = q_fixed
        lam = np.exp(loglam); b = np.exp(logb)
        return -log_pdf(q, lam, b, t).sum()
    if q_fixed is None:
        best = None
        for q0 in [0.6, 0.9, 1.0, 1.2, 1.5]:
            r = minimize(obj, [q0, np.log(1/80.0**2), np.log(2.0)],
                         method="Nelder-Mead", options={"xatol":1e-5})
            if best is None or r.fun < best.fun: best = r
        q = best.x[0]
    else:
        r = minimize(obj, [np.log(1/80.0**2), np.log(2.0)],
                     method="Nelder-Mead", options={"xatol":1e-5})
        q = q_fixed
    lam = np.exp(r.x[-2] if q_fixed is not None else r.x[1])
    b = np.exp(r.x[-1] if q_fixed is not None else r.x[2])
    return q, lam, b, r.fun

# 1) MLE
q_mle, lam_mle, b_mle, nll_mle = fit_qweibull(bearings)
Tmax_mle = support_upper(q_mle, lam_mle, b_mle)
print(f"\n[1] MLE: q={q_mle:.3f}, lam={lam_mle:.5f}, beta={b_mle:.3f}, nll={nll_mle:.1f}")
print(f"    隐含端点 T_max={Tmax_mle:.1f} (若 q<1); 现场最长寿命={bearings.max():.1f}")

# 2) profile 似然 over q
print("\n[2] profile log-likelihood over q:")
grid = np.linspace(0.3, 1.7, 29)
prof = []
for q0 in grid:
    _, _, _, nl = fit_qweibull(bearings, q_fixed=q0)
    prof.append(-nl)
prof = np.array(prof)
prof -= np.nanmax(prof)
for q, p in zip(grid, prof):
    p = -np.inf if not np.isfinite(p) else p
    bar = "#"*int(max(0,min(20,-p))/2)
    print(f"    q={q:.2f}  2*DeltaLL={-2*p:7.2f}  {bar}")
good = np.where(np.isfinite(prof) & (prof > -1.92))[0]
ci_lo, ci_hi = grid[good[0]], grid[good[-1]]
print(f"    profile 95% 区间 q in [{ci_lo:.2f}, {ci_hi:.2f}]  -> 宽度 {ci_hi-ci_lo:.2f}")

# 3) 先验借用: N(q_hist, sigma^2), 强度 delta; 后验众数 + 近似区间
print("\n[3] 不同历史先验中心 q_hist 下的后验众数 (sigma=0.3, delta=2.0):")
def post_mode(q_hist, sigma=0.3, delta=2.0):
    def obj(q):
        if q<=0.2 or q>=1.95: return 1e10
        _,_,_,nl = fit_qweibull(bearings, q_fixed=q)
        return nl + delta*(q-q_hist)**2/(2*sigma**2)
    best=None
    for q0 in [0.6,0.9,1.1,1.4]:
        r=minimize(obj, q0, method="Nelder-Mead", options={"xatol":1e-5})
        if best is None or r.fun<best.fun: best=r
    return best.x[0]

for qh in [0.8, 1.0, 1.2]:
    qp = post_mode(qh)
    print(f"    历史批 q_hist={qh:.1f} -> 后验众数 q={qp:.3f}")

# 4) 构造历史批: 早期型号, q=1.2 (重尾), 尺度稍小, n=15
print("\n[4] 加入历史批 (模拟早期型号, q_hist=1.2, n=15) 的幂先验借用:")
rng = np.random.default_rng(7)
q_h, lam_h, b_h = 1.2, (1/60.0)**2, 2.0
u = rng.random(15)
t_h = ((1-(1-u)**((1-q_h)/(2-q_h)))/((1-q_h)*lam_h))**(1/b_h)
def joint_nll(theta):
    q, loglam, logb = theta
    if q<=0.2 or q>=1.95: return 1e10
    lam=np.exp(loglam); b=np.exp(logb)
    ll_cur = log_pdf(q, lam, b, bearings).sum()
    ll_his = log_pdf(q_h, lam_h, b_h, t_h).sum()  # 历史批用自己的参数
    # 幂先验: 历史批似然 * delta, 拉现场 q 向 q_h
    return -(ll_cur + delta_prior*(ll_his))
# 用 q_hist 中心化的高斯近似幂先验
for delta_prior in [0.0, 0.5, 1.0, 2.0]:
    def obj(q):
        if q<=0.2 or q>=1.95: return 1e10
        _,_,_,nl = fit_qweibull(bearings, q_fixed=q)
        # 历史批对现场 q 的借用: 用 N(q, q_h, sigma_hist^2)
        return nl + delta_prior*(q-q_h)**2/(2*0.3**2)
    best=None
    for q0 in [0.8,1.0,1.3]:
        r=minimize(obj,q0,method="Nelder-Mead",options={"xatol":1e-5})
        if best is None or r.fun<best.fun: best=r
    print(f"    借用强度 delta={delta_prior:.1f} -> 现场 q 估计={best.x[0]:.3f} (真值/MLE≈{q_mle:.2f}, 历史批 q=1.2)")
