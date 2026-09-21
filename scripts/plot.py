"""
Plot LER over the swept parameter for one or more runs.

    python scripts/plot.py results/delta/asced-ms_cold_W5-F1@11c4c5
    python scripts/plot.py results/circuit-level-over-p/*

Each path is either a run directory (containing `data.csv`) or a curve
directory, in which case its latest run directory is used.
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

#
#
# Run resolution
#
#


def resolve_run_dir(path: Path) -> Path:
    if (path / "data.csv").exists():
        return path

    run_dirs = [data_path.parent for data_path in path.glob("*/data.csv")]
    if not run_dirs:
        raise FileNotFoundError(f"no run with data.csv under {path}")

    return max(run_dirs)


def find_axis(data: pd.DataFrame, meta: dict) -> str:
    fixed = {"LER", "num_trials", *meta["params"], *meta["circuit"]}
    candidates = [column for column in data.columns if column not in fixed]
    if len(candidates) != 1:
        raise ValueError(f"expected exactly one swept column, got {candidates}")

    return candidates[0]


def build_labels(run_dirs: list[Path]) -> list[str]:
    curve_names = [run_dir.parent.name for run_dir in run_dirs]
    labels = [name.split("@")[0] for name in curve_names]

    return labels if len(set(labels)) == len(labels) else curve_names


#
#
# Plotting
#
#


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--per-round-LER", action="store_true", dest="per_round_ler")
    args = parser.parse_args()

    run_dirs = [resolve_run_dir(path) for path in args.paths]
    labels = build_labels(run_dirs)

    plt.figure()
    axis = None

    for run_dir, label in zip(run_dirs, labels):
        data = pd.read_csv(run_dir / "data.csv")
        with open(run_dir / "meta.json") as handle:
            meta = json.load(handle)

        run_axis = find_axis(data, meta)
        if axis is not None and run_axis != axis:
            raise ValueError(f"runs sweep different axes: {axis} and {run_axis}")
        axis = run_axis

        data = data.sort_values(axis)
        ler = data["LER"].to_numpy(dtype=float)
        sigma = np.sqrt(ler * (1 - ler) / data["num_trials"].to_numpy())

        if args.per_round_ler:
            rounds = data["num_rounds"].to_numpy(dtype=float)
            if not np.all(np.isfinite(rounds) & (rounds > 0)):
                raise ValueError(f"num_rounds must be positive in {run_dir}")

            survival = 1 - ler
            derivative = np.zeros_like(ler)
            np.power(
                survival,
                1 / rounds - 1,
                out=derivative,
                where=survival > 0,
            )
            sigma = derivative * sigma / rounds
            ler = 1 - survival ** (1 / rounds)

        plt.errorbar(
            data[axis],
            ler,
            yerr=3 * sigma,
            marker="o",
            capsize=4,
            label=label,
        )

    plt.xlabel(axis)
    plt.ylabel("LER per round" if args.per_round_ler else "LER")
    plt.yscale("log")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
