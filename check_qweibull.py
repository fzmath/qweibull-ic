# -*- coding: utf-8 -*-
"""数值校验 qweibull.py: 归一化、边界、导数、竞争风险似然."""
import numpy as np, sys
sys.path.insert(0, r"G:\OPT\q-Weibull竞争风险论文\code")
from qweibull import log_pdf, log_sf, pdf, sf, support_upper, cr_loglik

rng = np.random.default_rng(0)
print("=== 1. 归一化: f 积分应为 1 ===")
for q, lam, beta in [(0.5, 1e-5, 2.0), (0.8, 1e-5, 2.0), (1.0, 1e-5, 2.0), (1.3, 1e-5, 2.0), (1.5, 1e-5, 3.0)]:
    Tmax = support_upper(q, lam, beta)
    t = np.linspace(0.001, Tmax if Tmax<np.inf else 5000, 100000)
    dt = t[1]-t[0]
    integ = np.trapezoid(pdf(q, lam, beta, t), t)
    print(f"  q={q:.1f}, lam={lam}, beta={beta}: integral={integ:.4f}, S(0)={sf(q,lam,beta,0.0):.4f}, S(end)={sf(q,lam,beta,t[-1]):.6f}")

print("\n=== 2. 生存函数边界 ===")
for q, lam, beta in [(0.5, 1e-5, 2.0), (1.3, 1e-5, 2.0)]:
    Tmax = support_upper(q, lam, beta)
    print(f"  q={q}: T_max={Tmax:.1f}, S(Tmax)={sf(q,lam,beta,Tmax if Tmax<np.inf else 1e10):.6e}")

print("\n=== 3. f = -dS/dt (数值导数校验) ===")
q, lam, beta = 0.7, 1e-5, 2.0
t = np.array([100.0, 500.0, 1000.0])
num_f = -(np.exp(log_sf(q,lam,beta,t+1)) - np.exp(log_sf(q,lam,beta,t-1)))/2
ana_f = pdf(q, lam, beta, t)
print(f"  t={t}: numerical f={num_f}, analytical f={ana_f}, ratio={num_f/ana_f}")

print("\n=== 4. 竞争风险: 无竞争原因时退化为单因 ===")
# k=1, all failures
t = np.array([100.0, 200.0, 300.0])
cause = np.array([1, 1, 1])
ll_single = cr_loglik(np.array([q]), np.array([lam]), np.array([beta]), t, cause)
ll_manual = log_pdf(q, lam, beta, t).sum()
print(f"  cr_loglik={ll_single:.4f}, manual sum log_pdf={ll_manual:.4f}, match={np.isclose(ll_single, ll_manual)}")

print("\n=== 5. 右删失: cause=0 贡献 log S ===")
cause2 = np.array([0, 1, 0])
ll2 = cr_loglik(np.array([q]), np.array([lam]), np.array([beta]), t, cause2)
ll2_manual = log_sf(q,lam,beta,100) + log_pdf(q,lam,beta,200) + log_sf(q,lam,beta,300)
print(f"  cr_loglik={ll2:.4f}, manual={ll2_manual:.4f}, match={np.isclose(ll2, ll2_manual)}")

print("\n=== 6. q=1 退化校验 ===")
t3 = np.array([100.0, 200.0])
print(f"  log_pdf(q=1)={log_pdf(1.0, lam, beta, t3)}, Weibull={np.log(beta)+np.log(lam)+(beta-1)*np.log(t3)-lam*t3**beta}")
print(f"  log_sf(q=1)={log_sf(1.0, lam, beta, t3)}, Weibull={-lam*t3**beta}")
