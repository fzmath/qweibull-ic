# -*- coding: utf-8 -*-
"""重新生成 Figure 1: 修正差值图纵轴"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

lam = 1/4**2; beta = 2.0
t = np.linspace(0.01, 10, 500)

# q-Weibull q=1.4
q = 1.4
inner = 1-(1-q)*lam*t**beta
S_q = inner**((2-q)/(1-q))

# Frailty theta=0.5
theta = 0.5
S_f = (1+theta*lam*t**beta)**(-1/theta)

# Weibull
S_w = np.exp(-lam*t**beta)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

ax1.plot(t, S_q, 'b-', linewidth=2, label=f'q-Weibull (q=1.4)')
ax1.plot(t, S_f, 'r--', linewidth=2, label=f'Frailty (θ=0.5)')
ax1.plot(t, S_w, 'k:', linewidth=1.5, label='Weibull')
ax1.set_xlabel('t'); ax1.set_ylabel('S(t)')
ax1.set_title('(a) Matched parameters')
ax1.legend(fontsize=8)
ax1.set_ylim(0, 1.05)

# 差值图: 真正的差值
diff = S_q - S_f
ax2.plot(t, diff, 'b-', linewidth=2)
ax2.axhline(0, color='k', linestyle='--', alpha=0.5)
ax2.set_xlabel('t'); ax2.set_ylabel('S_q(t) - S_f(t)')
ax2.set_title('(b) Difference')
# 自动纵轴
ax2.set_ylim(np.min(diff)*1.2, np.max(diff)*1.2)

plt.tight_layout()
plt.savefig(r'G:\OPT\JSCS-WFC20261005\paper\fig1_survival.pdf', bbox_inches='tight')
print(f"差值范围: {diff.min():.6f} 到 {diff.max():.6f}")
print("Figure 1 已更新")
