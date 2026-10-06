# -*- coding: utf-8 -*-
"""
真算 frailty 模型的 Fisher information matrix.
参数: (θ, logλ, logβ)
对比 q-Weibull 的 (q, logλ, logβ)
"""
import numpy as np
import sys
sys.path.insert(0, r"G:\OPT\q-Weibull竞争风险论文\code")
from qweibull import log_sf as log_sf_q

def log_sf_frailty(theta, lam, beta, t):
    """frailty survival: S(t) = (1+θλt^β)^{-1/θ}"""
    return -1/theta * np.log(1 + theta*lam*t**beta)

def compute_fisher_3param(log_sf_func, p0, n_sim=200000, w=1.0, rc_rate=0.37):
    """
    数值计算 Fisher info matrix for 3 parameters.
    p0 = [p1, logλ, logβ] 其中 p1 是 θ 或 q
    """
    rng = np.random.default_rng(123)
    N = n_sim
    p1_true, loglam, logb = p0
    lam = np.exp(loglam); b = np.exp(logb)
    
    # 生成数据 (从 frailty 生成)
    u = rng.random(N)
    if p1_true > 0:
        # frailty: S(t) = (1+θλt^β)^{-1/θ}
        # 反解: t = ((1/u)^θ - 1)/(θλ))^{1/β}
        t = (((1/u)**p1_true - 1)/(p1_true*lam))**(1/b)
    L = np.floor(t/w)*w; R = L + w
    rc = rng.random(N) < rc_rate
    R[rc] = np.nan
    
    eps = 1e-4
    scores = np.zeros((N, 3))
    for j in range(3):
        p_plus = list(p0); p_minus = list(p0)
        p_plus[j] += eps; p_minus[j] -= eps
        
        def nll_i(i, p):
            p1, ll, lb = p
            l = np.exp(ll); bb = np.exp(lb)
            Sl = np.exp(log_sf_frailty(p1, l, bb, np.array([L[i]])))[0]
            if rc[i]:
                return -np.log(Sl)
            else:
                Sr = np.exp(log_sf_frailty(p1, l, bb, np.array([R[i]])))[0]
                diff = Sl - Sr
                if diff <= 0: return 100
                return -np.log(diff)
        
        s_p = np.zeros(N); s_m = np.zeros(N)
        for i in range(N):
            s_p[i] = nll_i(i, p_plus)
            s_m[i] = nll_i(i, p_minus)
        scores[:, j] = -(s_p - s_m) / (2*eps)
    
    FI = (scores.T @ scores) / N
    return FI

# 算 frailty θ=0.5 的 Fisher info
print("=== Frailty Fisher Information (θ=0.5) ===")
p0 = [0.5, np.log(1/4**2), np.log(2.0)]
FI_f = compute_fisher_3param(log_sf_frailty, p0, n_sim=50000)

sd_f = np.sqrt(np.diag(FI_f))
corr_f = FI_f / np.outer(sd_f, sd_f)

print(f"Diagonal: I(θ,θ)={FI_f[0,0]:.3f}, I(logλ,logλ)={FI_f[1,1]:.3f}, I(logβ,logβ)={FI_f[2,2]:.3f}")
print(f"corr(θ, logλ) = {corr_f[0,1]:.3f}")
print(f"corr(θ, logβ) = {corr_f[0,2]:.3f}")
print(f"corr(logλ, logβ) = {corr_f[1,2]:.3f}")

# 条件信息
I_qq = FI_f[0,0]
I_qψ = FI_f[0,1:]; I_ψψ = FI_f[1:,1:]
I_cond = I_qq - I_qψ @ np.linalg.inv(I_ψψ) @ I_qψ
print(f"条件信息 I(θ|logλ,logβ) = {I_cond:.3f}")
print(f"相对信息比 = {I_cond/I_qq:.3f}")

print()
print("=== 对比 q-Weibull q=1.2 (之前算的) ===")
print(f"q-Weibull: corr(q,logλ)=-0.88, corr(q,logβ)=-0.96, 条件信息=6.7%")
print(f"Frailty:   corr(θ,logλ)={corr_f[0,1]:.2f}, corr(θ,logβ)={corr_f[0,2]:.2f}, 条件信息={I_cond/I_qq*100:.1f}%")
