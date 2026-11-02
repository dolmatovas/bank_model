"""Plot gamma-by-eta policy heatmaps from a sweep's summary.csv."""

import argparse
import csv
from pathlib import Path

import numpy as np


OTHER_AXES = ("spread_gap", "deposit_beta", "rate_vol", "amortization", "deposits")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()
    with args.summary.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError("The sweep summary is empty")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    groups = sorted({tuple(row[key] for key in OTHER_AXES) for row in rows})
    for group_id, group in enumerate(groups):
        subset = [row for row in rows if tuple(row[key] for key in OTHER_AXES) == group]
        gammas = sorted({float(row["gamma"]) for row in subset})
        etas = sorted({float(row["eta"]) for row in subset})
        figures = []
        for metric, title in (("f0", "Доля fixed в начальном состоянии"),
                              ("mean_fixed_share", "Средняя доля fixed на траекториях")):
            values = np.full((len(gammas), len(etas)), np.nan)
            for row in subset:
                if row["status"] == "complete" and row[metric]:
                    values[gammas.index(float(row["gamma"])),
                           etas.index(float(row["eta"]))] = float(row[metric])
            figures.append((values, title))

        fig, axes = plt.subplots(1, 2, figsize=(max(9, 2.2 * len(etas) + 5),
                                             max(4, .85 * len(gammas) + 2)),
                                 constrained_layout=True)
        for axis, (values, title) in zip(axes, figures):
            plot = axis.imshow(np.ma.masked_invalid(values), vmin=0, vmax=1,
                               cmap="viridis", aspect="auto")
            axis.set_xticks(range(len(etas)), [f"{eta * 10000:g}" for eta in etas])
            axis.set_yticks(range(len(gammas)), [f"{gamma:g}" for gamma in gammas])
            axis.set_xlabel("η, б.п. на единицу доли float")
            axis.set_ylabel("γ")
            axis.set_title(title)
            for i in range(len(gammas)):
                for j in range(len(etas)):
                    if np.isfinite(values[i, j]):
                        axis.text(j, i, f"{values[i, j]:.2f}", ha="center", va="center",
                                  color="white" if values[i, j] < .5 else "black")
            fig.colorbar(plot, ax=axis, label="fixed share")
        parameters = ", ".join(f"{key}={value}" for key, value in zip(OTHER_AXES, group))
        fig.suptitle(parameters)
        output = args.summary.parent / f"policy_heatmap_{group_id:03d}.png"
        fig.savefig(output, dpi=160)
        plt.close(fig)
        print(output)


if __name__ == "__main__":
    main()
