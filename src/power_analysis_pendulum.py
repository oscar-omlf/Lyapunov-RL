import os
import numpy as np
import pandas as pd
from statsmodels.stats.power import TTestIndPower

RUN_NUMBER = 5
LOG_ROOT = "logs/InvertedPendulum/EVALUATION"
AGENTS = ["LAC", "LDP", "TD3", "LQR"]
ALPHA = 0.05
POWER = 0.95
MIN_SAMPLES = 2

run_dir = os.path.join(LOG_ROOT, f"run_{RUN_NUMBER}")

returns = {}
for agent in AGENTS:
    csv_path = run_dir / agent / "returns.csv"
    data = np.loadtxt(csv_path, delimiter=",", skiprows=1)
    returns[agent] = data.mean(axis=1)

pairs = [(i,j) for idx,i in enumerate(AGENTS) for j in AGENTS[idx+1:]]
d_vals = {}
for i,j in pairs:
    xi, xj = returns[i], returns[j]
    ni, nj = len(xi), len(xj)
    si, sj = xi.std(ddof=1), xj.std(ddof=1)
    pooled = np.sqrt(((ni-1)*si**2 + (nj-1)*sj**2)/(ni+nj-2))
    d_vals[(i,j)] = abs(xi.mean() - xj.mean())/pooled

m = len(d_vals)
alpha_adj = ALPHA / m

tool = TTestIndPower()
req_n_return = {}
for pair,d in d_vals.items():
    try:
        n = tool.solve_power(
            effect_size=d,
            nobs1=None,
            alpha=alpha_adj,
            power=POWER,
            ratio=1.0,
            alternative='two-sided')
        req_n = max(int(np.ceil(n)), MIN_SAMPLES)
    except Exception:
        req_n = MIN_SAMPLES
    req_n_return[pair] = req_n

st_ldp = np.loadtxt(run_dir/"LDP"/"steps_to_stabilize.csv", delimiter=",", skiprows=1)
st_lqr = np.loadtxt(run_dir/"LQR"/"steps_to_stabilize.csv", delimiter=",", skiprows=1)
steps_ldp = np.nanmean(st_ldp, axis=1)
steps_lqr = np.nanmean(st_lqr, axis=1)

n1,n2 = len(steps_ldp), len(steps_lqr)
s1,s2 = steps_ldp.std(ddof=1), steps_lqr.std(ddof=1)
pooled_sd = np.sqrt(((n1-1)*s1**2 + (n2-1)*s2**2)/(n1+n2-2))
d_steps = abs(steps_ldp.mean() - steps_lqr.mean())/pooled_sd

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
