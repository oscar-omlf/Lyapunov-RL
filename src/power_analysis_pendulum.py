#!/usr/bin/env python3
import numpy as np
import pandas as pd
from pathlib import Path
import warnings
from statsmodels.stats.power import TTestIndPower
from statsmodels.tools.sm_exceptions import ConvergenceWarning

# -------------------------- USER CONFIG -------------------------- #
RUN_NUMBER  = 5                # Which run_<n> to read
LOG_ROOT    = Path("logs/InvertedPendulum/EVALUATION")
AGENTS      = ["LAC", "LDP", "TD3", "LQR"]
ALPHA       = 0.05             # Family-wise α
POWER       = 0.95             # Desired power (1-β)
MIN_SAMPLES = 2                # Minimum runs per group for trivial effects
# ----------------------------------------------------------------- #

# Suppress convergence & runtime warnings
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# 1. Point to your run directory
run_dir = LOG_ROOT / f"run_{RUN_NUMBER}"
if not run_dir.exists():
    raise FileNotFoundError(f"Run directory {run_dir} not found")

# 2. Load per-run mean returns for all agents
returns = {}
for agent in AGENTS:
    csv_path = run_dir / agent / "returns.csv"
    data = np.loadtxt(csv_path, delimiter=",", skiprows=1)
    returns[agent] = data.mean(axis=1)

# 3. Compute pairwise Cohen's d (returns)
pairs = [(i,j) for idx,i in enumerate(AGENTS) for j in AGENTS[idx+1:]]
d_vals = {}
for i,j in pairs:
    xi, xj = returns[i], returns[j]
    ni, nj = len(xi), len(xj)
    si, sj = xi.std(ddof=1), xj.std(ddof=1)
    pooled = np.sqrt(((ni-1)*si**2 + (nj-1)*sj**2)/(ni+nj-2))
    d_vals[(i,j)] = abs(xi.mean() - xj.mean())/pooled

# 4. Bonferroni-adjusted alpha for returns
m = len(d_vals)
alpha_adj = ALPHA / m

# 5. Solve required n per group for each return comparison
tool = TTestIndPower()
req_n_return = {}
for pair,d in d_vals.items():
    try:
        n = tool.solve_power(effect_size=d,
                              nobs1=None,
                              alpha=alpha_adj,
                              power=POWER,
                              ratio=1.0,
                              alternative='two-sided')
        req_n = max(int(np.ceil(n)), MIN_SAMPLES)
    except Exception:
        req_n = MIN_SAMPLES
    req_n_return[pair] = req_n

# 6. Steps-to-stabilize (LDP & LQR only)
st_ldp = np.loadtxt(run_dir/"LDP"/"steps_to_stabilize.csv", delimiter=",", skiprows=1)
st_lqr = np.loadtxt(run_dir/"LQR"/"steps_to_stabilize.csv", delimiter=",", skiprows=1)
steps_ldp = np.nanmean(st_ldp, axis=1)
steps_lqr = np.nanmean(st_lqr, axis=1)

# Cohen's d for steps
n1,n2 = len(steps_ldp), len(steps_lqr)
s1,s2 = steps_ldp.std(ddof=1), steps_lqr.std(ddof=1)
pooled_sd = np.sqrt(((n1-1)*s1**2 + (n2-1)*s2**2)/(n1+n2-2))
d_steps = abs(steps_ldp.mean() - steps_lqr.mean())/pooled_sd

# Solve required n for steps (no multiple comparisons)
try:
    n_st = tool.solve_power(effect_size=d_steps,
                             nobs1=None,
                             alpha=ALPHA,
                             power=POWER,
                             ratio=1.0,
                             alternative='two-sided')
    req_n_steps = max(int(np.ceil(n_st)), MIN_SAMPLES)
except Exception:
    req_n_steps = MIN_SAMPLES

# 7. Summarise results
df_rows = []
for (i,j),n in req_n_return.items():
    df_rows.append({
        "Metric":      "Return",
        "Comparison":  f"{i} vs {j}",
        "Cohen_d":     round(d_vals[(i,j)],3),
        "Alpha_adj":   round(alpha_adj,4),
        "Power":       POWER,
        "Req_n/group": n
    })
# add steps row
df_rows.append({
    "Metric":      "Steps",
    "Comparison":  "LDP vs LQR",
    "Cohen_d":     round(d_steps,3),
    "Alpha_adj":   ALPHA,
    "Power":       POWER,
    "Req_n/group": req_n_steps
})

df = pd.DataFrame(df_rows)
print("\n=== Prior Power Analysis (95% power, Bonferroni-adjusted) ===")
print(df.to_string(index=False))