# -*- coding: utf-8 -*-
"""
数值 Fisher information: q-Weibull 区间删失模型.
计算 (q, logλ, logβ) 的 Fisher 信息矩阵, 展示:
1. q 的信息小
2. q 和 λ/β 高度相关
3. 这解释仿真中的 bias 和 RMSE
"""
import numpy as np
from scipy.integrate import quad
import sys
sys.path.insert(0, r"G:\OPT\q-Weibull竞争风险论文\code")
from qweibull import log_pdf, log_sf, pdf, sf

def fisher_information(q_true, lam_true, beta_true, n, w=1.0, rc_rate=0.37):
    """
    数值计算 Fisher information matrix for (q, logλ, logβ).
    区间删失: L=floor(t/w)*w, R=L+w, 37% right-censored.
    """
    # 用蒙特卡洛近似: 生成大量数据, 算 score 的外积
    rng = np.random.default_rng(42)
    N = 50000  # 大样本近似期望信息
    u = rng.random(N)
    t = ((1-(1-u)**((1-q_true)/(2-q_true)))/((1-q_true)*lam_true))**(1/beta_true)
    L = np.floor(t/w)*w; R = L + w
    rc = rng.random(N) < rc_rate
    R[rc] = np.nan

    # 数值 score: 对每个参数求导
    eps = 1e-4
    def nll(par):
        q, loglam, logb = par
        lam = np.exp(loglam); b = np.exp(logb)
        Sl = np.exp(log_sf(q, lam, b, L))
        ll = np.log(Sl[rc]).sum()
        p = Sl[~rc] - np.exp(log_sf(q, lam, b, R[~rc]))
        return -(ll + np.log(p).sum())

    par0 = [q_true, np.log(lam_true), np.log(beta_true)]
    scores = np.zeros((N, 3))
    for j in range(3):
        p_plus = par0.copy(); p_minus = par0.copy()
        p_plus[j] += eps; p_minus[j] -= eps
        # 数值导数: 每个观测的 score
        def nll_i(i, par):
            q, loglam, logb = par
            lam = np.exp(loglam); b = np.exp(logb)
            Sl = np.exp(log_sf(q, lam, b, L[i:i+1]))
            if rc[i]:
                return -np.log(Sl[0])
            else:
                p = Sl[0] - np.exp(log_sf(q, lam, b, R[i:i+1]))
                return -np.log(p[0])
        # 向量化: 对所有 i 算 score
        s_plus = np.zeros(N); s_minus = np.zeros(N)
        for i in range(N):
            s_plus[i] = nll_i(i, p_plus)
            s_minus[i] = nll_i(i, p_minus)
        scores[:, j] = -(s_plus - s_minus) / (2*eps)  # negative nll derivative = score

    # Fisher information = E[score * score^T]
    FI = (scores.T @ scores) / N
    return FI

# 计算不同 q 下的 Fisher information
print("=== Fisher Information Matrix (n=1, 每观测) ===")
print("参数: (q, logλ, logβ)\n")

for q_true in [1.0, 1.2, 1.4]:
    lam_true = 1/4**2; beta_true = 2.0
    FI = fisher_information(q_true, lam_true, beta_true, 1000)
    # 相关矩阵
    sd = np.sqrt(np.diag(FI))
    corr = FI / np.outer(sd, sd)
    print(f"q={q_true}:")
    print(f"  diag (信息): q={FI[0,0]:.4f}, logλ={FI[1,1]:.4f}, logβ={FI[2,2]:.4f}")
    print(f"  corr(q, logλ) = {corr[0,1]:.3f}")
    print(f"  corr(q, logβ) = {corr[0,2]:.3f}")
    print(f"  corr(logλ, logβ) = {corr[1,2]:.3f}")
    # 条件信息: q 的边际信息
    # Schur complement: I_q·ψ = I_qq - I_qψ I_ψψ^{-1} I_ψq
    I_qq = FI[0,0]
    I_qψ = FI[0,1:]; I_ψψ = FI[1:,1:]
    I_q_cond = I_qq - I_qψ @ np.linalg.inv(I_ψψ) @ I_qψ
    print(f"  条件信息 I(q|logλ,logβ) = {I_q_cond:.4f}")
    print(f"  相对信息比 = {I_q_cond/I_qq:.3f}")
    print()
