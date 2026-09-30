"""Measured learning curves and endpoint bars in the saved two-panel layout."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cryodyna.optpose.plotting import save_figure

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--csv", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--methods", nargs="+", required=True)
    p.add_argument("--metric-label", required=True)
    p.add_argument("--x-label", default="Training pass")
    p.add_argument("--value-column", default="value")
    p.add_argument("--output", required=True)
    p.add_argument("--caption", required=True, help="Provenance, particle count, seeds, alignment and uncertainty")
    args = p.parse_args()
    frame = pd.read_csv(args.csv).rename(columns={args.value_column: "value"})
    if not {"method", "epoch", "value"}.issubset(frame):
        raise ValueError("CSV requires method, epoch, value")
    if len(set(args.methods)) != len(args.methods):
        raise ValueError("Each method must occur once in the method list")
    selected = frame[frame.method.isin(args.methods)].copy()
    if set(selected.method) != set(args.methods):
        raise ValueError("Measured data required for every requested method")
    if selected.duplicated(["method", "epoch"]).any():
        raise ValueError("Provide one explicit aggregate per method and epoch")
    if not np.isfinite(selected[["epoch", "value"]].to_numpy()).all() or (selected.value < 0).any():
        raise ValueError("Finite epochs and nonnegative error values required")
    has_sd = "sd" in selected
    if has_sd and (not np.isfinite(selected.sd).all() or (selected.sd < 0).any()):
        raise ValueError("SD must be finite and nonnegative for every displayed observation")
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    matplotlib.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
        "font.size": 7, "axes.labelsize": 7, "legend.fontsize": 7, "xtick.labelsize": 6,
        "ytick.labelsize": 6, "pdf.fonttype": 42, "svg.fonttype": "none",
        "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
        "axes.linewidth": .6, "lines.linewidth": 1.1})
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2))
    fig.subplots_adjust(left=.085, right=.985, bottom=.23, top=.82, wspace=.32)
    named = {"CryoDyna-optpose": "#0072B2", "Optpose": "#0072B2", "Opt": "#0072B2",
             "Random SO(3)": "#D55E00", "DRGN-AI": "#009E73"}
    colors = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#6C757D", "#E69F00"]
    for i, name in enumerate(args.methods):
        data = selected[selected.method == name].sort_values("epoch")
        color = named.get(name, colors[i % len(colors)])
        if len(data) == 1:
            axes[0].axhline(data.value.iloc[0], color=color, linestyle="--", label=name)
        else:
            axes[0].plot(data.epoch, data.value, color=color, label=name)
        if has_sd:
            axes[0].fill_between(data.epoch.to_numpy(), (data.value-data.sd).clip(lower=0).to_numpy(),
                                 (data.value+data.sd).to_numpy(), color=color, alpha=.15)
        last = data.iloc[-1]
        axes[1].bar(i, last.value, color=color, width=.65,
                    yerr=last.sd if has_sd else None, capsize=2)
    upper = max(float((selected.value + (selected.sd if has_sd else 0)).max())*1.1, .05)
    axes[0].set(xlabel=args.x_label, ylabel=args.metric_label, ylim=(0, upper))
    axes[1].set(xticks=np.arange(len(args.methods)), xticklabels=[x.replace("CryoDyna-optpose", "Optpose") for x in args.methods],
                ylabel="Final " + args.metric_label[0].lower()+args.metric_label[1:], ylim=(0, upper))
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", ncol=min(3, len(args.methods)), bbox_to_anchor=(.5, .99))
    save_figure(fig, out/"figure")
    selected.to_csv(out/"source_data.csv", index=False)
    (out/"caption.md").write_text(args.dataset+". "+args.caption+"\n")
    (out/"figure.json").write_text(json.dumps(dict(dataset=args.dataset, methods=args.methods,
        metric=args.metric_label, x_label=args.x_label, rows_input=len(frame), rows_displayed=len(selected),
        selection="requested method names", endpoint="last recorded epoch per method", caption=args.caption), indent=2)+"\n")


if __name__ == "__main__": main()
