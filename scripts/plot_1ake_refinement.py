"""Plot the measured discovery and held-out particle diagnostics."""
from pathlib import Path
import argparse
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, "/home/nyh/.codex/skills/nature-figure/scripts")
from audit_panel_alignment import require_matplotlib_panel_alignment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    args = parser.parse_args()
    root = args.input
    discovery = pd.read_csv(root / "stages.csv")
    validation = pd.read_csv(root / "validation128/stages.csv")
    cross = pd.read_csv(root / "validation128/cross_scores.csv")
    methods = ["growing_k8", "freeze17_k8"]
    labels = ["Expand radius 12 → 32", "Cap radius at 17"]
    colors = ["#397ba8", "#b78245"]
    rng = np.random.default_rng(20261011)
    summaries, endpoints = [], []
    for cohort, table in (("Discovery (32)", discovery), ("Validation (128)", validation)):
        table = table[(table.condition == "observed") & (table.stage == 4)
                      & table.variant.isin(methods)]
        wide = table.pivot(index="particle", columns="variant", values="angle_deg")
        # Resample particle IDs jointly across methods; these are sampling intervals.
        draws = rng.integers(0, len(wide), size=(10000, len(wide)))
        for method in methods:
            values = wide[method].to_numpy()
            ci = np.quantile(values[draws].mean(axis=1), [.025, .975])
            summaries.append(dict(cohort=cohort, method=method, n=len(values),
                                  mean_deg=values.mean(), lower=ci[0], upper=ci[1]))
        difference = (wide[methods[1]] - wide[methods[0]]).to_numpy()
        interval = np.quantile(difference[draws].mean(axis=1), [.025, .975])
        endpoints.append(dict(cohort=cohort, mean_paired_difference_deg=difference.mean(),
                              lower=interval[0], upper=interval[1]))
    pd.DataFrame(summaries).to_csv(root / "figure_endpoints.csv", index=False)
    score = cross[(cross.condition == "observed") & (cross.evaluated_radius == 32)].pivot(
        index="particle", columns="candidate_stage", values="cosine")
    errors = validation[(validation.condition == "observed") & (validation.variant == methods[0])].pivot(
        index="particle", columns="stage", values="angle_deg")
    changes = pd.DataFrame({"score_gain": score[4] - score[1],
                            "angle_change_deg": errors[4] - errors[1]})
    changes.to_csv(root / "figure_paired_changes.csv")
    gain = changes.score_gain > 0
    harm = changes.angle_change_deg > 0
    evidence = dict(paired_cap_minus_growing=endpoints,
                    validation_score_gain_count=int(gain.sum()),
                    validation_score_gain_and_angle_increase_count=int((gain & harm).sum()),
                    validation_angle_increase_count=int(harm.sum()),
                    validation_mean_angle_change=float(changes.angle_change_deg.mean()),
                    validation_mean_score_gain=float(changes.score_gain.mean()),
                    interval="95% percentile bootstrap over paired particles; 10000 draws; seed 20261011")
    (root / "figure_evidence.json").write_text(json.dumps(evidence, indent=2))
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"], "font.size": 7,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.linewidth": .7, "pdf.fonttype": 42, "svg.fonttype": "none"})
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.9))
    ax = axes[0]
    for method_index, (method, label, color) in enumerate(zip(methods, labels, colors)):
        records = [r for r in summaries if r["method"] == method]
        x = np.arange(2) + (method_index - .5) * .32
        y = np.array([r["mean_deg"] for r in records])
        lower = np.array([r["lower"] for r in records])
        upper = np.array([r["upper"] for r in records])
        ax.bar(x, y, width=.29, color=color, label=label)
        ax.errorbar(x, y, yerr=np.stack([y - lower, upper - y]), fmt="none",
                    color="#333333", capsize=2, lw=.8)
        for xx, yy, hi in zip(x, y, upper):
            ax.text(xx, hi+1.9, f"{yy:.2f}°", ha="center", fontsize=7)
    ax.set_xticks(range(2), ["Discovery\nn = 32", "Fresh validation\nn = 128"])
    ax.set_ylim(0, 72)
    ax.set_ylabel("Final mean SO(3) angle error (°)")
    ax.set_title("Bandwidth effect across particle subsets", fontsize=8, pad=10)
    ax = axes[1]
    for selection, color in ((~(gain & harm), "#88949d"), (gain & harm, "#bd7065")):
        ax.scatter(changes.loc[selection, "score_gain"], changes.loc[selection, "angle_change_deg"],
                   s=12, color=color, alpha=.75, linewidths=0)
    ax.axvline(0, color="#c0c0c0", lw=.6, zorder=0)
    ax.axhline(0, color="#c0c0c0", lw=.6, zorder=0)
    ax.set_xlabel("Image-score change at fixed radius 32")
    ax.set_ylabel("SO(3) angle-error change (°)")
    ax.set_title("Refinement 1 → 4 · validation particles", fontsize=8, pad=10)
    ax.set_ylim(-185, 195)
    ax.set_yticks([-150, -75, 0, 75, 150])
    for i, ax in enumerate(axes):
        ax.text(-.17, 1.07, chr(97+i), transform=ax.transAxes, weight="bold", fontsize=10)
    fig.suptitle("1AKE: image-fit gains coexist with pose errors", fontsize=10, y=.99)
    fig.legend(*axes[0].get_legend_handles_labels(), ncol=2, loc="upper center",
               bbox_to_anchor=(.5, .94), frameon=False, fontsize=7)
    fig.text(.5, .035,
             "Fixed true-class density; centered images; raw full SO(3) angles. Both searches use beam 8.\n"
             "a: means with 95% paired-particle bootstrap intervals. b: 128 particles; red marks 60 with score and error increases.",
             ha="center", fontsize=6.4)
    fig.subplots_adjust(left=.09, right=.985, top=.77, bottom=.25, wspace=.42)
    require_matplotlib_panel_alignment(fig, json_out=str(root / "refinement_diagnosis.alignment.json"), strict=True)
    fig.savefig(root / "refinement_diagnosis.png", dpi=600)
    fig.savefig(root / "refinement_diagnosis.pdf")
    fig.savefig(root / "refinement_diagnosis.svg")


if __name__ == "__main__":
    main()
