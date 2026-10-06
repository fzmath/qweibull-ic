# -*- coding: utf-8 -*-
"""Generate figures for JSCS paper."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 11, "font.family": "serif"})

OUT = r"G:\OPT\JSCS-WFC20261005\paper"

# ============================================================
# Fig 1: Survival curves — q-Weibull vs frailty
# ============================================================
t = np.linspace(0.01, 10, 200)
lam = 1/4**2; beta = 2.0

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

# Left: matched parameters
ax = axes[0]
# q=1.4 Weibull
S_qw = (1+(1.4-1)*lam*t**beta)**(-(1.4-2)/(1.4-1))  # = [1+0.4 lam t^2]^(-1.5)
# frailty theta=0.5
S_fr = (1+0.5*lam*t**beta)**(-2.0)
# Weibull
S_w = np.exp(-lam*t**beta)
ax.plot(t, S_qw, "b-", lw=2, label="q-Weibull (q=1.4)")
ax.plot(t, S_fr, "r--", lw=2, label="Frailty (θ=0.5)")
ax.plot(t, S_w, "k:", lw=1.5, label="Weibull")
ax.set_xlabel("t"); ax.set_ylabel("Survival S(t)")
ax.set_title("(a) Matched parameters")
ax.legend(fontsize=9)

# Right: residual difference
ax = axes[1]
diff = S_qw - S_fr
ax.plot(t, diff, "b-", lw=2)
ax.axhline(0, color="k", ls=":", lw=0.8)
ax.set_xlabel("t"); ax.set_ylabel("$S_{qw}(t) - S_{fr}(t)$")
ax.set_title("(b) Difference")
ax.set_ylim(-0.02, 0.02)

plt.tight_layout()
plt.savefig(f"{OUT}/fig1_survival.pdf", bbox_inches="tight")
plt.close()
print("Fig 1 done")

# ============================================================
# Fig 2: Bidirectional confounding
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

# Left: frailty data → q̂
ax = axes[0]
theta = np.array([0.3, 0.5, 1.0])
qhat = np.array([1.32, 1.44, 1.61])
ax.errorbar(theta, qhat, yerr=[0.07, 0.06, 0.03], fmt="bo-", lw=2, capsize=4)
ax.axhline(1.0, color="k", ls=":", label="Weibull (q=1)")
ax.set_xlabel("True frailty θ"); ax.set_ylabel("Estimated q̂")
ax.set_title("(a) Frailty data → q-Weibull fit")
ax.legend(fontsize=9)

# Right: q-Weibull data → θ̂
ax = axes[1]
qtrue = np.array([1.2, 1.4, 1.6])
thetahat = np.array([0.36, 1.17, 2.34])
ax.errorbar(qtrue, thetahat, yerr=[0.14, 0.17, 0.28], fmt="rs-", lw=2, capsize=4)
ax.axhline(0, color="k", ls=":", label="Weibull (θ=0)")
ax.set_xlabel("True q"); ax.set_ylabel("Estimated θ̂")
ax.set_title("(b) q-Weibull data → frailty fit")
ax.legend(fontsize=9)

plt.tight_layout()
plt.savefig(f"{OUT}/fig2_confounding.pdf", bbox_inches="tight")
plt.close()
print("Fig 2 done")

# ============================================================
# Fig 3: Hazard functions
# ============================================================
fig, ax = plt.subplots(figsize=(6, 4))
h_qw = (2-1.4)*beta*lam*t**(beta-1)/(1+0.4*lam*t**beta)
h_fr = lam*beta*t**(beta-1)/(1+0.5*lam*t**beta)
h_w = lam*beta*t**(beta-1)
ax.plot(t, h_qw, "b-", lw=2, label="q-Weibull (q=1.4)")
ax.plot(t, h_fr, "r--", lw=2, label="Frailty (θ=0.5)")
ax.plot(t, h_w, "k:", lw=1.5, label="Weibull")
ax.set_xlabel("t"); ax.set_ylabel("Hazard h(t)")
ax.set_title("Hazard functions differ in shape")
ax.legend(fontsize=9)
ax.set_xlim(0, 10)
plt.tight_layout()
plt.savefig(f"{OUT}/fig3_hazard.pdf", bbox_inches="tight")
plt.close()
print("Fig 3 done")

print("All figures saved to", OUT)
