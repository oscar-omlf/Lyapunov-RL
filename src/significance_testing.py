#!/usr/bin/env python3
import os, numpy as np, itertools, json, sys
from pathlib import Path

import matplotlib.pyplot as plt
import scipy.stats as stats

RUN_NUMBER = 7
LOG_ROOT   = Path("logs/InvertedPendulum/EVALUATION")
ALPHA      = 0.05

BETAS = [round(0.1 * i, 1) for i in range(1, 10)]
AGENTS = [f"LDP_{beta}" for beta in BETAS] \
       + [f"LAS_TD3_{beta}" for beta in BETAS] \
       + ["LAC", "TD3", "LQR"]

run_dir = LOG_ROOT / f"run_{RUN_NUMBER}"
if not run_dir.exists():
    raise FileNotFoundError(f"Run directory {run_dir} not found")

def load_csv_as_array(csv_path: Path, nanaware=True):
    """Load csv (runs x episodes). Returns ndarray, skips header."""
    return np.loadtxt(csv_path, delimiter=",", skiprows=1)

def per_run_mean(arr, nanaware=True):
    if nanaware:
        return np.nanmean(arr, axis=1)
    return arr.mean(axis=1)

# --- load data -----------------------------------
returns_data = {}
steps_data   = {}
for agent in AGENTS:
    agent_dir = run_dir / agent
    ret_csv   = agent_dir / "returns.csv"
    step_csv  = agent_dir / "steps_to_stabilize.csv"

    if ret_csv.exists():
        arr_ret = load_csv_as_array(ret_csv)
        returns_data[agent] = per_run_mean(arr_ret, nanaware=False)

    if step_csv.exists():
        arr_step = load_csv_as_array(step_csv)
        steps_data[agent] = per_run_mean(arr_step, nanaware=True)

# --- print raw means ± SD ------------------------
print("\nRaw mean returns (ascending):")
stats_returns = {}
for a, v in returns_data.items():
    mean = v.mean()
    sd   = v.std(ddof=1)
    stats_returns[a] = (mean, sd)
for a, (m, s) in sorted(stats_returns.items(), key=lambda x: x[1][0]):
    print(f"  {a}: {m:.2f} ± {s:.2f}")

print("\nRaw mean steps-to-stabilize (ascending = faster):")
stats_steps = {}
for a, v in steps_data.items():
    mean = np.nanmean(v)
    sd   = np.nanstd(v, ddof=1)
    stats_steps[a] = (mean, sd)
for a, (m, s) in sorted(stats_steps.items(), key=lambda x: x[1][0]):
    print(f"  {a}: {m:.1f} ± {s:.1f}")

# --- plot vs β for LDP only ---------------------
betas = np.array(BETAS)
mean_returns = np.array([returns_data[f"LDP_{b}"].mean() for b in betas])
mean_steps   = np.array([np.nanmean(steps_data[f"LDP_{b}"]) for b in betas])

# Mean Return vs Beta
plt.figure(figsize=(6,4))
plt.plot(betas, mean_returns, marker='o')
plt.xlabel("β")
plt.ylabel("Mean Return")
plt.title("Mean Return vs β")
plt.grid(True)
plt.tight_layout()
plt.savefig(run_dir / "return_vs_beta.png")
plt.close()

# Mean Steps vs Beta
plt.figure(figsize=(6,4))
plt.plot(betas, mean_steps, marker='o')
plt.xlabel("β")
plt.ylabel("Mean Steps to Stabilize")
plt.title("Mean Steps vs β")
plt.grid(True)
plt.tight_layout()
plt.savefig(run_dir / "steps_vs_beta.png")
plt.close()

print(f"\nSaved plots: {run_dir/'return_vs_beta.png'} and {run_dir/'steps_vs_beta.png'}")

# ---------------- Normality check (Shapiro) -------------------------
def normality_table(data_dict):
    res = {}
    for k, v in data_dict.items():
        stat, p = stats.shapiro(v)
        res[k] = (stat, p)
    return res

shapiro_returns = normality_table(returns_data)
shapiro_steps   = normality_table({k:v for k,v in steps_data.items() if np.isfinite(v).any()})

lev_stat, lev_p = stats.levene(*returns_data.values())
print(f"\nLevene's test - Returns: stat={lev_stat:.3f}, p={lev_p:.4g}")

valid_runs = [v[~np.isnan(v)] for v in steps_data.values() if np.isfinite(v).any()]
if len(valid_runs) >= 2:
    lev_s, lev_ps = stats.levene(*valid_runs)
    print(f"Levene's test - Steps:   stat={lev_s:.3f}, p={lev_ps:.4g}")
else:
    print("Levene's test - Steps:   not enough groups with data to run Levene's test")

# ---------------- One-way tests -------------------------------------
def one_way_tests(data_dict):
    samples = list(data_dict.values())
    F, p_anova = stats.f_oneway(*samples)
    H, p_kw    = stats.kruskal(*samples)
    return {"ANOVA_F": F, "ANOVA_p": p_anova,
            "Kruskal_H": H, "Kruskal_p": p_kw}

anova_ret   = one_way_tests(returns_data)
anova_steps = one_way_tests({k:v for k,v in steps_data.items() if np.isfinite(v).any()})

# ---------------- Pairwise Welch t-tests + Holm correction ----------
import itertools
def pairwise_tests(data_dict):
    pairs = list(itertools.combinations(data_dict.keys(), 2))
    raw_p = {}
    for a, b in pairs:
        _, p = stats.ttest_ind(data_dict[a], data_dict[b], equal_var=False, nan_policy='omit')
        raw_p[(a, b)] = p
    m = len(raw_p)
    sorted_p = sorted(raw_p.items(), key=lambda x: x[1])
    adjusted = {}
    for i, ((a, b), pval) in enumerate(sorted_p, start=1):
        adjusted[(a, b)] = min((m - i + 1) * pval, 1.0)
    return raw_p, adjusted

raw_p_ret, adj_p_ret = pairwise_tests(returns_data)
valid_steps = {k:v for k,v in steps_data.items() if np.isfinite(v).any()}
raw_p_steps, adj_p_steps = ({}, {}) if len(valid_steps)<2 else pairwise_tests(valid_steps)

# ---------------- Print summary -------------------------------------
def fmt_pairwise(pdict):
    return "\n".join(f"{a} vs {b}: p={p:.4g}" for (a,b),p in sorted(pdict.items()))

print("\n=== Normality (Returns) ===")
for k,(s,p) in shapiro_returns.items():
    print(f"  {k}: stat={s:.3f}, p={p:.4g}")

print("\n=== Normality (Steps) ===")
for k,(s,p) in shapiro_steps.items():
    print(f"  {k}: stat={s:.3f}, p={p:.4g}")

print("\n=== One-way tests - Returns ===")
print(json.dumps(anova_ret, indent=2))
print("\n=== One-way tests - Steps ===")
print(json.dumps(anova_steps, indent=2))

print("\n=== Pairwise Welch t-tests - Returns (raw p) ===")
print(fmt_pairwise(raw_p_ret))
print("\n=== Pairwise Welch t-tests - Returns (Holm-adjusted) ===")
print(fmt_pairwise(adj_p_ret))

if raw_p_steps:
    print("\n=== Pairwise Welch t-tests - Steps (raw p) ===")
    print(fmt_pairwise(raw_p_steps))
    print("\n=== Pairwise Welch t-tests - Steps (Holm-adjusted) ===")
    print(fmt_pairwise(adj_p_steps))
else:
    print("\nNo valid pairwise step comparisons (only one agent produced finite data).")
