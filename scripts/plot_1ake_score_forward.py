"""Measured 2x2 numerical controls and a clean-data closure comparison."""
import argparse
import json
from pathlib import Path
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
    p = args.input
    data = pd.read_csv(p / "stages.csv")
    observed = data[(data.condition == "observed") & (data.stage == 4)].pivot(
        index="particle", columns="variant", values="angle_deg")
    clean = data[(data.condition == "clean_native") & (data.variant == "native_cosine") & (data.stage == 4)]
    variants = ["native_cosine", "native_weighted", "padded_cosine", "padded_weighted"]
    rng = np.random.default_rng(20261011)
    # These indices bootstrap measured rows; the plotted data come from search results.
    indices = rng.integers(0, len(observed), (10000, len(observed)))
    records = []
    for variant in variants:
        values = observed[variant].to_numpy()
        delta = values - observed.native_cosine.to_numpy()
        interval = np.quantile(values[indices].mean(1), [.025, .975])
        delta_interval = np.quantile(delta[indices].mean(1), [.025, .975])
        records.append(dict(variant=variant, mean_deg=values.mean(), lower=interval[0], upper=interval[1],
                            paired_change=delta.mean(), change_lower=delta_interval[0], change_upper=delta_interval[1]))
    pd.DataFrame(records).to_csv(p / "figure_endpoints.csv", index=False)
    baseline = observed.native_cosine
    tail = baseline > 90
    evidence = dict(n=len(observed), above90_count=int(tail.sum()), above90_error_share=float(baseline[tail].sum()/baseline.sum()),
                    median_deg=float(baseline.median()), bootstrap="paired particle IDs;10000 percentile draws;seed20261011",
                    intervals=records)
    (p / "figure_evidence.json").write_text(json.dumps(evidence, indent=2))
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
                         "font.size": 7, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.linewidth": .7, "svg.fonttype": "none", "pdf.fonttype": 42})
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.85))
    colors = ["#397ba8", "#b78245"]
    for k, (kind, label, color) in enumerate(zip(["cosine", "weighted"], ["Original score", "Corner-PSD weighted"], colors)):
        rows = [r for r in records if r["variant"].endswith(kind)]
        x = np.arange(2)+(k-.5)*.32
        y = np.array([r["mean_deg"] for r in rows])
        lo = np.array([r["lower"] for r in rows]);hi = np.array([r["upper"] for r in rows])
        axs[0].bar(x, y, width=.29, color=color, label=label)
        axs[0].errorbar(x, y, yerr=[y-lo, hi-y], fmt="none", color="#333333", capsize=2, lw=.8)
        for xx, yy, upper in zip(x, y, hi):
            axs[0].text(xx, upper+1.2, f"{yy:.2f}°", ha="center", fontsize=7)
    axs[0].set_xticks([0, 1], ["Native FFT\n128³", "Zero-padded FFT\n256³"])
    axs[0].set_ylim(0, 53)
    axs[0].set_ylabel("Mean SO(3) angle error (°)")
    axs[0].set_title("Scoring × Fourier interpolation", fontsize=8, pad=10)
    axs[0].legend(loc="upper left", frameon=False, fontsize=6.5)
    for values, label, color in [(baseline.to_numpy(), "Original images", colors[0]),
                                 (clean.angle_deg.to_numpy(), "Matched noiseless images", "#33928c")]:
        x = np.sort(values);y = np.arange(1, len(x)+1)/len(x)*100
        axs[1].step(x, y, where="post", color=color, lw=1.5, label=label)
    axs[1].set_xlim(-3, 185);axs[1].set_ylim(0, 106)
    axs[1].set_xticks([0, 45, 90, 135, 180]);axs[1].set_yticks([0, 25, 50, 75, 100])
    axs[1].set_xlabel("Full SO(3) angle error (°)")
    axs[1].set_ylabel("Particles below angle (%)")
    axs[1].set_title("Original search · same 128 particles", fontsize=8, pad=10)
    axs[1].legend(loc="lower right", bbox_to_anchor=(1, .08), frameon=False, fontsize=6.5)
    axs[1].text(.97, .49, f"{int(tail.sum())}/128 particles above 90°\n{evidence['above90_error_share']:.1%} of summed angle error",
                ha="right", transform=axs[1].transAxes, fontsize=7)
    for i, ax in enumerate(axs):
        ax.text(-.17, 1.07, chr(97+i), transform=ax.transAxes, weight="bold", fontsize=10)
    fig.suptitle("1AKE: controlled scoring and interpolation tests", fontsize=10, y=.98)
    fig.text(.5, .04,
             "Fixed true-class density; zero translation; raw SO(3) error. Identical grid, beam and frequency schedule.\n"
             "a: means and 95% paired-particle bootstrap intervals. Image size remains 128² in every condition.",
             ha="center", fontsize=6.4)
    fig.subplots_adjust(left=.09, right=.985, top=.79, bottom=.25, wspace=.42)
    require_matplotlib_panel_alignment(fig, json_out=str(p / "score_forward.alignment.json"), strict=True)
    fig.savefig(p / "score_forward.png", dpi=600)
    fig.savefig(p / "score_forward.svg")
    fig.savefig(p / "score_forward.pdf")


if __name__ == "__main__":
    main()
