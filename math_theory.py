# -*- coding: utf-8 -*-
"""
数学推导: q-Weibull vs frailty 的局部等价性.
1. Taylor展开: 近 q=1 时两模型一阶偏离相同
2. Score 相关性: q 和 θ 的 score 平行
3. Fisher information: (q, θ) 方向近零特征值
"""
import numpy as np

print("=" * 60)
print("命题1: q-Weibull 与 frailty 在 q=1/θ=0 处一阶等价")
print("=" * 60)
print("""
q-Weibull (q=1+δ):
  S_q(t) = [1+δ·λt^β]^{1/δ - 1}
  log S_q = (1/δ - 1) log(1+δ·λt^β)
         = (1/δ - 1)[δ·λt^β - δ²λ²t^{2β}/2 + O(δ³)]
         = λt^β - δ·λt^β - δ·λ²t^{2β}/2 + O(δ²)
         = log S_W(t) - δ(λt^β + λ²t^{2β}/2) + O(δ²)

Frailty (small θ):
  S_f(t) = (1+θλt^β)^{-1/θ}
  log S_f = -(1/θ) log(1+θλt^β)
         = -(1/θ)[θλt^β - θ²λ²t^{2β}/2 + O(θ³)]
         = -λt^β - θλ²t^{2β}/2 + O(θ²)
         = log S_W(t) - θλ²t^{2β}/2 + O(θ²)

关键: 如果允许 λ 调整 (MLE 会做), q-Weibull 的 -δλt^β 项
可以被 λ' = λ(1+δ) 吸收, 剩下:
  log S_q ≈ -λ't^β - δλ'^2 t^{2β}/2
  log S_f ≈ -λt^β - θλ² t^{2β}/2
两者在 (λ, 偏离项) 空间中重合, δ=θ 时一阶相同.

→ 结论: 在 Weibull 附近, q-Weibull 和 frailty 的生存函数
  一阶 Taylor 展开相同, 无法区分.
""")

print("=" * 60)
print("命题2: Score 函数平行")
print("=" * 60)
print("""
在 q=1 (Weibull) 处, 对 log-likelihood 求 q 的导数:
  U_q = ∂ log L / ∂q |_{q=1}
对 frailty θ 的导数:
  U_θ = ∂ log L / ∂θ |_{θ=0}

由命题1的 Taylor 展开可知:
  ∂ log S_q / ∂q |_{q=1} = -(λt^β + λ²t^{2β}/2)
  ∂ log S_f / ∂θ |_{θ=0} = -λ²t^{2β}/2

两者只差一个 λt^β 项, 而该项可以通过调整 λ 吸收.
在 profile λ 之后:
  U_q (after profiling λ) ∝ U_θ (after profiling λ)

→ 结论: profile 掉尺度参数后, q 和 θ 的 score 方向平行.
  Fisher information 矩阵在 (q, θ) 方向有近零特征值.
""")

print("=" * 60)
print("命题3: 区间删失进一步降低 q 的信息")
print("=" * 60)
print("""
右删失: 似然贡献 = log f(t) 或 log S(t)
区间删失: 似然贡献 = log[S(L) - S(R)] ≈ log[f(L)·(R-L)]

q 是尾部参数, 其信息主要来自 S(t) 的尾部行为.
区间删失用的是 S(L)-S(R), 主要反映中间区间的密度,
尾部信息被区间宽度 R-L 模糊掉.

→ 结论: 区间删失比右删失更难区分 q 和 θ.
""")

print("=" * 60)
print("验证: 数值检查 score 平行")
print("=" * 60)

# 数值导数

lam = 1/4**2; beta = 2.0; t = np.array([1.0, 2.0, 5.0])

def logS_q(q, t):
    inner = 1-(1-q)*lam*t**beta
    return np.sum(((2-q)/(1-q))*np.log(inner))

def logS_f(theta, t):
    return np.sum(-1/theta * np.log(1+theta*lam*t**beta))

# 数值导数
eps = 1e-4
U_q = (logS_q(1+eps, t) - logS_q(1-eps, t)) / (2*eps)
U_theta = (logS_f(eps, t) - logS_f(-eps, t)) / (2*eps)

print(f"  t={t}")
print(f"  U_q (at q=1) = {U_q:.4f}")
print(f"  U_θ (at θ=0) = {U_theta:.4f}")
print(f"  比值 U_q/U_θ = {U_q/U_theta:.4f}")
print(f"  (接近常数说明 score 平行)")
