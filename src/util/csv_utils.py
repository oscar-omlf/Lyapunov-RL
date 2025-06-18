import os
import csv
import numpy as np


def _write_2d_csv(arr: np.ndarray, path: str, header_prefix: str = "ep") -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        header = [f"{header_prefix}{i}" for i in range(arr.shape[1])]
        writer.writerow(header)
        writer.writerows(arr)


def _write_counts_csv(counts: np.ndarray, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["run", "episodes_stabilized"])
        for run_idx, c in enumerate(counts):
            writer.writerow([run_idx, int(c)])
