"""Measured pose error through all HPS stages and Adam passes."""
import argparse
import importlib.util
import json
from pathlib import Path
import pickle
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cryodyna.optpose import volume_metrics
from cryodyna.optpose.plotting import save_figure

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--truth", type=Path, required=True)
    p.add_argument("--evaluator", type=Path, help="Optional evaluator override")
    p.add_argument("--search-run", type=Path, help="Search trace directory; defaults to --run")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(exist_ok=True, parents=True)
    evaluator = volume_metrics
    if args.evaluator:
        spec = importlib.util.spec_from_file_location("paper_metric", args.evaluator)
        evaluator = importlib.util.module_from_spec(spec); spec.loader.exec_module(evaluator)
    evaluator.validate()
    truth = np.asarray(pickle.load(args.truth.open("rb")), dtype=float)
    search_run = args.search_run or args.run
    paths = [("Grid search", int(path.stem[5:]), path) for path in sorted((search_run/"search").glob("stage*.npz"))]
    paths += [("Adam", int(path.stem[5:]), path) for path in sorted((args.run/"evaluation").glob("epoch*.npz")) if int(path.stem[5:]) > 0]
    if not paths or not any(phase == "Grid search" for phase, _, _ in paths):
        raise ValueError("Recorded search stages are required")
    final_stage = max(stage for phase, stage, _ in paths if phase == "Grid search")
    with np.load(search_run/"search"/f"stage{final_stage:02d}.npz") as final_search, np.load(args.run/"evaluation/epoch000.npz") as initial_adam:
        for key in ("rotations", "particle_ids"):
            np.testing.assert_array_equal(final_search[key], initial_adam[key])
    rows, gauges = [], []
    ids_previous = None
    for phase, stage, path in paths:
        with np.load(path) as saved:
            ids = saved["particle_ids"]
            pred = saved["rotations"].astype(float)
        if ids_previous is not None:
            np.testing.assert_array_equal(ids_previous, ids)
        ids_previous = ids.copy()
        target = truth[ids]
        raw_view, raw_so3 = evaluator.angles(pred, target)
        metric = evaluator.fit(pred, target)["paper_mean_frobenius"]
        rows.append(dict(phase=phase, stage=stage, n=len(ids),
                         view_mean_deg=metric["view_mean_deg"], so3_mean_deg=metric["so3_mean_deg"],
                         view_p90_deg=metric["view_p90_deg"], raw_view_mean_deg=float(raw_view.mean()),
                         raw_so3_mean_deg=float(raw_so3.mean()), checkpoint=str(path.resolve())))
        gauges.append(dict(phase=phase, stage=stage, **metric))
    frame = pd.DataFrame(rows)
    frame.to_csv(args.output/"source_data.csv", index=False)
    (args.output/"alignment_metrics.json").write_text(json.dumps(gauges, indent=2)+"\n")
    grid = frame[frame.phase == "Grid search"]
    adam = frame[frame.phase == "Adam"]
    bridge = grid.iloc[-1:].copy(); bridge["stage"] = 0
    adam = pd.concat([bridge, adam])
    matplotlib.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
        "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 9, "legend.fontsize": 8,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "pdf.fonttype": 42, "svg.fonttype": "none",
        "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
        "axes.linewidth": .7, "lines.linewidth": 1.3})
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.3))
    fig.subplots_adjust(left=.085, right=.975, bottom=.21, top=.76, wspace=.30)
    colors = ["#0072B2", "#009E73"]
    for column, label, color, style in zip(
            ["view_mean_deg", "so3_mean_deg"], ["Viewing-direction error", "Full SO(3) error"],
            colors, ["-", "--"]):
        axes[0].plot(grid.stage, grid[column], marker="o", markersize=3, color=color, linestyle=style, label=label)
        axes[1].plot(adam.stage, adam[column], marker="o", markersize=2, color=color, linestyle=style, label=label)
    axes[0].set(title="Grid search", xlabel="Search stage", ylabel="Mean angular error (°)",
                xticks=grid.stage, xticklabels=["Coarse"] + [f"Refine {i}" for i in grid.stage.iloc[1:]],
                ylim=(0, grid.so3_mean_deg.max()*1.15))
    axes[1].set(title="Adam refinement", xlabel="Adam pass", ylabel="Mean angular error (°)",
                xlim=(-.5, float(adam.stage.max())+.5),
                ylim=(0, adam.so3_mean_deg.max()*1.15))
    for ax in axes:
        ax.grid(axis="y", color="#E5E7EB", linewidth=.5)
        ax.set_axisbelow(True)
    fig.suptitle(f"Optpose with fixed 80S density  |  {len(ids_previous):,} particles", fontsize=10, y=.98)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", ncol=2, bbox_to_anchor=(.53, .91))
    fig.text(.085, .035, "Adam pass 0 repeats the final search result. Error-axis ranges differ between panels.", fontsize=7)
    save_figure(fig, args.output/"error_curve")
    caption = (f"Fixed-volume Optpose control on the first {len(ids_previous)} prepared synthetic 80S particles. "
               "Every particle is retained at every stage. The left panel shows the best candidate after coarse Hopf search "
               "and each of four local refinements (Fourier radii 12, 17, 22, 27, 32 pixels). The right panel shows the "
               f"last search result at pass 0 and all {int(adam.stage.max())} Adam passes, fitting rotations and translations "
               "with a fixed external density prior. Metrics are means across particles after one global alignment and "
               "global hand selection per stage, using the min(100, N) candidate mean-Frobenius rule. "
               "One deterministic trajectory is shown; uncertainty across independent runs remains outside this figure. "
               "Ground-truth rotations are accessed only by this evaluation script. The two vertical axes use different ranges. "
               "The final search rotations and IDs equal the saved Adam initialization. Consult the run manifest for training arguments.")
    (args.output/"caption.md").write_text(caption+"\n")
    print(frame[["phase", "stage", "view_mean_deg", "so3_mean_deg"]].to_string(index=False))


if __name__ == "__main__": main()
