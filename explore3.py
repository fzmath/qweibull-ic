# -*- coding: utf-8 -*-
"""
继续探索:
1. 多变量/聚类数据 — shared frailty 在组内产生相关, q-Weibull 不产生
2. 实际诊断方法: 如何判断 q>1 是真重尾还是 frailty?
3. 牙齿数据: 同一儿童多颗牙的相关性
"""
import numpy as np
from scipy.optimize import minimize
from scipy.stats import pearsonr
import sys
sys.path.insert(0, r"G:\OPT\q-Weibull竞争风险论文\code")
from qweibull import log_sf
import pandas as pd

# ============================================================
# 1. 聚类数据中的组内相关
# ============================================================
print("=== 实验1: 组内相关系数 (ICC) ===")
print("q-Weibull: 独立观测, ICC=0")
print("Frailty: 同一儿童多颗牙共享 frailty, ICC>0\n")

def sim_qweibull_cluster(q_true, eta, beta, n_subj, n_teeth, rng):
    """q-Weibull: 每个儿童每颗牙独立."""
    lam = eta**(-beta)
    u = rng.random((n_subj, n_teeth))
    t = ((1-(1-u)**((1-q_true)/(2-q_true)))/((1-q_true)*lam))**(1/beta)
    return t

def sim_frailty_cluster(eta, beta, theta, n_subj, n_teeth, rng):
    """frailty: 同一儿童所有牙共享 Z."""
    lam = eta**(-beta)
    Z = rng.gamma(1/theta, scale=theta, size=n_subj)  # shape=(n_subj,)
    u = rng.random((n_subj, n_teeth))
    t = (-np.log(u)/(Z[:,None]*lam))**(1/beta)
    return t

n_subj, n_teeth = 500, 4
rng = np.random.default_rng(0)

t_qw = sim_qweibull_cluster(1.4, 4.0, 2.0, n_subj, n_teeth, rng)
t_fr = sim_frailty_cluster(4.0, 2.0, 0.5, n_subj, n_teeth, rng)

# ICC: 同一儿童两颗牙的相关性
def icc(t_mat):
    # 取前两颗牙的相关
    r, p = pearsonr(t_mat[:,0], t_mat[:,1])
    return r

print(f"  q-Weibull: ICC(牙1,牙2) = {icc(t_qw):.4f}")
print(f"  Frailty:   ICC(牙1,牙2) = {icc(t_fr):.4f}")
print("  → 如果有聚类数据, ICC=0 vs ICC>0 可以区分两模型.")

# ============================================================
# 2. 实际诊断: 牙齿数据中同一儿童多颗牙的相关
# ============================================================
print("\n=== 实验2: 牙齿数据中的组内相关 ===")
df = pd.read_csv(r"G:\OPT\Weibull发表策略\data\tooth24_clean.csv")
print(f"  数据: n={len(df)}, 但只有 tooth 24 一颗牙")
print("  (原始 Tandmobiel 研究有 16 颗恒牙, 我们只用了 1 颗)")
print("  → 如果能拿到多颗牙数据, 可以直接算 ICC 来判断是否有 frailty.")

# ============================================================
# 3. LRT 检验功效: 什么时候能区分两模型?
# ============================================================
print("\n=== 实验3: LRT 区分两模型的功效 ===")
print("(数据来自 q-Weibull, 检验是否需要 frailty)")

def nll_q(q, eta, beta, L, R, rc):
    lam = eta**(-beta)
    Sl = np.exp(log_sf(q, lam, beta, L))
    ll = np.log(Sl[rc]).sum()
    p = Sl[~rc] - np.exp(log_sf(q, lam, beta, R[~rc]))
    return -(ll + np.log(p).sum())

def sim_qweibull_ic(q_true, eta, beta, n, rng, rc_rate=0.37):
    lam = eta**(-beta)
    u = rng.random(n)
    t = ((1-(1-u)**((1-q_true)/(2-q_true)))/((1-q_true)*lam))**(1/beta)
    L = np.floor(t); R = L+1.0
    rc = rng.random(n) < rc_rate; R[rc] = np.nan
    return L, R, rc

for n in [500, 2000, 5000]:
    rng = np.random.default_rng(42)
    lrt_stats = []
    for rep in range(10):
        L,R,rc = sim_qweibull_ic(1.4, 4.0, 2.0, n, rng)
        # Weibull nll (q=1 fixed)
        nll_w = nll_q(1.0, 4.0, 2.0, L, R, rc)
        # q-Weibull nll
        nll_qw = nll_q(1.4, 4.0, 2.0, L, R, rc)
        lrt = 2*(nll_w - nll_qw)  # LRT for q≠1
        lrt_stats.append(lrt)
    lrt_stats = np.array(lrt_stats)
    print(f"  n={n}: LRT(q≠1) 均值={lrt_stats.mean():.1f}, 拒绝率={(lrt_stats>3.84).mean():.0%}")

print("\n=== 总结: 实用诊断策略 ===")
print("1. 单变量区间删失, n≤1000: q>1 不可信 (可能是 frailty)")
print("2. n≥5000: LRT 可以区分")
print("3. 有聚类/多变量数据: 算 ICC, ICC>0 → frailty")
print("4. q<1: 可信 (frailty 无法伪装有界)")
print("5. 实际建议: 拟合 q-Weibull 的同时必须拟合 frailty,")
print("   如果两者 nll 接近, 不要宣称发现重尾.")
