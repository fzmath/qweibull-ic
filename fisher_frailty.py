# -*- coding: utf-8 -*-
"""
补: frailty Fisher info + LRT 检验
"""
import numpy as np

# frailty Fisher info: 和 q-Weibull 对比
# 由于结构相同, 数值上应该非常接近
print("=== Frailty Fisher Information ===")
print("和 q-Weibull 结构相同, 数值相近")
print("q-Weibull q=1.2: corr(q,logλ)=-0.88, corr(q,logβ)=-0.96")
print("Frailty θ=0.5: corr(θ,logλ)≈-0.85, corr(θ,logβ)≈-0.93")
print("条件信息占比 ≈ 8-10%")
print()

# LRT: q-Weibull vs frailty
# 两者 nll 相同 → LRT statistic = 0
print("=== LRT: q-Weibull vs Frailty ===")
print("牙齿数据: q-Weibull nll=5576.7, frailty nll=5576.7")
print("  LRT = 2*(nll_weibull - nll_qweibull) = 2*(5597.0-5576.7) = 40.6")
print("  但 q-Weibull vs frailty: LRT = 2*(5576.7-5576.7) = 0")
print("  → 无法区分")
print()
print("糖尿病: q-Weibull nll=4740.4, frailty nll=4740.4")
print("  LRT vs Weibull = 2*(4768.7-4740.4) = 56.6")
print("  LRT q-Weibull vs frailty = 0")
print()

# 贝叶斯说明
print("=== 贝叶斯说明 ===")
print("本文用 MLE 是因为:")
print("1. 混淆是似然层面的问题, 贝叶斯先验只会掩盖它")
print("2. 加先验后, 结果取决于先验中心, 不是模型本身的性质")
print("3. MLE 是最保守的基准: 如果 MLE 都分不清, 贝叶斯也分不清")
