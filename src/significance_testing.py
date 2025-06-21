import os, numpy as np, pandas as pd, itertools, math, warnings, json, pprint, textwrap, sys
from pathlib import Path

import scipy.stats as stats

RUN_NUMBER = 6
LOG_ROOT   = Path("logs/InvertedPendulum/EVALUATION")
AGENTS     = ["LAC", "LDP", "TD3", "LQR"]
ALPHA      = 0.05

run_dir = LOG_ROOT / f"run_{RUN_NUMBER}"
if not run_dir.exists():
    raise FileNotFoundError(f"Run directory {run_dir} not found")

def load_csv_as_array(csv_path: Path, nanaware=True):
    """Load csv (runs × episodes). Returns ndarray, skips header."""
    arr = np.loadtxt(csv_path, delimiter=",", skiprows=1)
    return arr

def per_run_mean(arr, nanaware=True):
    if nanaware:
        return np.nanmean(arr, axis=1)
    return arr.mean(axis=1)

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
        steps_data[agent] = per_run_mean(arr_step, nanaware=True)  # nan-aware for steps

# ---------------- Normality check (Shapiro) ------------------------- #
def normality_table(data_dict):
    res = {}
    for k, v in data_dict.items():
        stat, p = stats.shapiro(v)
        res[k] = (stat, p)
    return res

shapiro_returns = normality_table(returns_data)
shapiro_steps   = normality_table({k: v for k, v in steps_data.items() if np.isfinite(v).any()})

# ---------------- One-way tests ------------------------------------- #
def one_way_tests(data_dict, nanaware=False):
    keys = list(data_dict.keys())
    samples = [data_dict[k] for k in keys]

    # Parametric
    F, p_anova = stats.f_oneway(*samples)

    # Non-parametric
    H, p_kw = stats.kruskal(*samples)

    return {"ANOVA_F": F, "ANOVA_p": p_anova,
            "Kruskal_H": H, "Kruskal_p": p_kw}

anova_ret   = one_way_tests(returns_data)
anova_steps = one_way_tests({k:v for k,v in steps_data.items() if np.isfinite(v).any()})

# ---------------- Pairwise Welch t-tests + Holm correction ---------- #
def pairwise_tests(data_dict):
    pairs = list(itertools.combinations(data_dict.keys(), 2))
    raw_p = {}
    for a, b in pairs:
        t, p = stats.ttest_ind(data_dict[a], data_dict[b], equal_var=False, nan_policy='omit')
        raw_p[(a, b)] = p

    # Holm–Bonferroni correction
    m = len(raw_p)
    sorted_p = sorted(raw_p.items(), key=lambda x: x[1])  # ascending by p
    adjusted = {}
    for i, ((a, b), pval) in enumerate(sorted_p, start=1):
        adj_p = min((m - i + 1) * pval, 1.0)   # Holm step-down
        adjusted[(a, b)] = adj_p

    return raw_p, adjusted

raw_p_ret, adj_p_ret = pairwise_tests(returns_data)
# steps: only include agents with finite means (LDP & LQR)
valid_steps = {k: v for k, v in steps_data.items() if np.isfinite(v).any()}
if len(valid_steps) >= 2:
    raw_p_steps, adj_p_steps = pairwise_tests(valid_steps)
else:
    raw_p_steps, adj_p_steps = {}, {}

# ---------------- Print summary ------------------------------------- #
def fmt_pairwise(pdict):
    out = []
    for (a, b), p in sorted(pdict.items()):
        out.append(f"{a} vs {b}: p = {p:.4g}")
    return "\n".join(out)

print("\n=== Normality check (Shapiro-Wilk) – Returns ===")
for k, (s, p) in shapiro_returns.items():
    print(f"{k:>4}: stat={s:.3f}  p={p:.4g}")

print("\n=== Normality check (Shapiro-Wilk) – Steps ===")
for k, (s, p) in shapiro_steps.items():
    print(f"{k:>4}: stat={s:.3f}  p={p:.4g}")

print("\n=== One-way tests – Returns ===")
print(json.dumps(anova_ret, indent=2))

print("\n=== One-way tests – Steps ===")
print(json.dumps(anova_steps, indent=2))

print("\n=== Pairwise Welch t-tests – Returns (raw p-values) ===")
print(fmt_pairwise(raw_p_ret))
print("\n=== Pairwise Welch t-tests – Returns (Holm-adjusted) ===")
print(fmt_pairwise(adj_p_ret))

if raw_p_steps:
    print("\n=== Pairwise Welch t-tests – Steps (raw p-values) ===")
    print(fmt_pairwise(raw_p_steps))
    print("\n=== Pairwise Welch t-tests – Steps (Holm-adjusted) ===")
    print(fmt_pairwise(adj_p_steps))
else:
    print("\nNo valid pairwise step comparisons (only one agent produced finite data).")
