"""Quantitative grids: convergence, matched method comparison, ab-initio accuracy.

Particle means are descriptive; a single seed has no across-seed error bars.
Exports have editable text, a 183 mm width, source tables and measured layout.
"""
import json
import os
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

COLORS = ["#6C757D", "#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00"]
matplotlib.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
    "font.size": 7, "axes.titlesize": 8, "axes.labelsize": 7, "legend.fontsize": 7,
    "xtick.labelsize": 6, "ytick.labelsize": 6, "pdf.fonttype": 42, "svg.fonttype": "none",
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
    "axes.linewidth": .6, "lines.linewidth": 1.1})


def save_figure(fig, stem):
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.canvas.draw()
    # Public reproduction uses this dependency-free equal-grid rectangle gate.
    boxes = np.array([a.get_position().bounds for a in fig.axes])
    sizes = boxes[:, 2:] * np.array(fig.get_size_inches()) * 72
    if len(sizes) > 1 and np.max(np.ptp(sizes, axis=0)) > 1.5:
        raise ValueError("Comparable plot rectangles differ by over 1.5 pt")
    layout = {"rectangles_fraction": boxes.tolist(), "tolerance_pt": 1.5, "equal_sizes_pass": True}
    Path(str(stem)+".alignment.json").write_text(json.dumps(layout, indent=2))
    # Optional publication audit; core exports stay portable outside this machine.
    qa = os.environ.get("CRYODYNA_FIGURE_QA")
    if qa:
        sys.path.insert(0, qa)
        from audit_panel_alignment import require_matplotlib_panel_alignment
        require_matplotlib_panel_alignment(fig, json_out=str(stem)+".alignment.json",
                                          tolerance_pt=1.5, gutter_tolerance_pt=1.5, strict=True)
    fig.savefig(str(stem)+".png", dpi=600, facecolor="white")
    fig.savefig(str(stem)+".svg", facecolor="white")
    fig.savefig(str(stem)+".pdf", facecolor="white")
    plt.close(fig)


def load_frames(outdir):
    from .benchmark import collect_runs
    rows = []
    for path, manifest in collect_runs(outdir):
        frame = pd.read_csv(path / "metrics.csv")
        frame["method"] = manifest["method"]
        frame["angle"] = manifest["angle"]
        frame["run"] = str(path)
        rows.append(frame)
    return pd.concat(rows, ignore_index=True)


def plot_learning(outdir, output):
    frame = load_frames(outdir)
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output / "learning_curves.csv", index=False)
    angles = sorted(frame.angle.unique())
    methods = list(frame.method.unique())
    fig, axes = plt.subplots(3, len(angles), figsize=(max(7.2, 1.44*len(angles)), 6.2), squeeze=False, sharex=True, sharey="row")
    fig.subplots_adjust(left=.10, right=.98, bottom=.09, top=.88, hspace=.28, wspace=.20)
    for col, angle in enumerate(angles):
        for row, (key, label) in enumerate((("loss", "Image correlation loss"), ("rmsd_A", "Cα RMSD (Å)"), ("pose_deg", "Rotation error (°)"))):
            ax = axes[row, col]
            for j, method in enumerate(methods):
                data = frame[(frame.angle == angle) & (frame.method == method)].sort_values("epoch")
                ax.plot(data.epoch, data[key], label=method, color=COLORS[j % len(COLORS)])
            if col == 0: ax.set_ylabel(label)
            if row == 0: ax.set_title(f"{angle:g}° perturbation")
            if row == 2: ax.set_xlabel("Epoch")
            ax.grid(axis="y", alpha=.18, linewidth=.5)
    # Set shared limits after every column has contributed its data.
    for row, key in ((1, "rmsd_A"), (2, "pose_deg")):
        axes[row, 0].set_ylim(0, max(float(frame[key].max()) * 1.05, .05))
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.5, .99), ncol=len(methods))
    save_figure(fig, output / "learning_curves")
    (output / "learning_curves.caption.md").write_text(
        "Particle means at each saved epoch. Columns show exact random-axis SO(3) perturbations. "
        "Rows show evaluation image correlation loss, per-particle Cα Kabsch RMSD, and full SO(3) geodesic error. "
        "Evaluation IDs are fixed before optimization; pose/structure truth is read only by the evaluator. "
        "Epoch 0 follows reference-only initialization. Single-seed curves have no across-seed uncertainty band.\n")


def plot_comparison(specs, output):
    from .benchmark import collect_runs, write_json
    frames, contracts, budgets = [], [], []
    labels = []
    for spec in specs:
        if "=" not in spec: raise ValueError("Use LABEL=OUTDIR")
        label, path = spec.split("=", 1)
        if not label or label in labels: raise ValueError("Use unique, nonempty method labels")
        labels.append(label)
        runs = collect_runs(path)
        for directory, manifest in runs:
            contracts.append((manifest["particle_ids"], manifest["evaluation_ids"], manifest["truth_sha256"], manifest["pose_init"]))
            arguments = manifest.get("arguments")
            budgets.append({k: arguments.get(k) for k in ("batch_size", "eval_batch_size", "init_steps", "sample_seed", "epochs")}
                           if arguments else None)
            frame = pd.read_csv(directory / "metrics.csv")
            last = frame.sort_values("epoch").iloc[-1].to_dict()
            last.update(label=label, angle=manifest["angle"], seed=manifest["seed"], source=str(directory))
            frames.append(last)
    if not frames: raise ValueError("No evaluated runs found")
    if any(c != contracts[0] for c in contracts[1:]):
        raise ValueError("Comparison requires identical particles, evaluation split, truth and pose mode")
    if all(b is not None for b in budgets) and any(b != budgets[0] for b in budgets[1:]):
        raise ValueError("Matched comparisons require equal batching, sampling, warmup and epoch budgets")
    frame = pd.DataFrame(frames)
    if frame.duplicated(["label", "angle", "seed"]).any():
        raise ValueError("Each label must identify one method; select its run directories")
    angles = sorted(frame.angle.unique())
    expected = set(angles)
    if any(set(frame[frame.label == label].angle) != expected for label in labels):
        raise ValueError("Each method must contain the same perturbation angles")
    seed_groups = [set(map(tuple, frame[frame.label == label][["angle", "seed"]].to_numpy())) for label in labels]
    if any(group != seed_groups[0] for group in seed_groups[1:]):
        raise ValueError("Each method must contain matched angles and seeds")
    if frame.epoch.nunique() != 1: raise ValueError("Endpoint comparison requires the same epoch budget")
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output / "comparison.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1))
    fig.subplots_adjust(left=.085, right=.985, bottom=.19, top=.80, wspace=.30)
    x = np.arange(len(angles)); width = .8 / len(labels)
    for ax, metric, ylabel in zip(axes, ["rmsd_A", "pose_deg"], ["Cα RMSD (Å)", "Rotation error (°)"]):
        for i, label in enumerate(labels):
            grouped = frame[frame.label == label].groupby("angle")[metric]
            means = grouped.mean().reindex(angles)
            spread = grouped.std().reindex(angles)
            ax.bar(x+(i-(len(labels)-1)/2)*width, means, width=width, color=COLORS[i % len(COLORS)],
                   label=label, yerr=spread.to_numpy() if spread.notna().all() else None, capsize=2)
        ax.set(xticks=x, xticklabels=[f"{a:g}" for a in angles], xlabel="Initial perturbation (°)", ylabel=ylabel)
        ax.set_ylim(bottom=0)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", ncol=len(labels), bbox_to_anchor=(.5, .99))
    save_figure(fig, output / "method_comparison")
    write_json(output / "comparison_contract.json", {"epoch": int(frame.epoch.iloc[0]),
        "center": "mean of per-run particle means", "error_bars": "sample SD across seeds when >=2 seeds for every angle",
        "budget_verification": "verified" if all(b is not None for b in budgets) else "historical budgets require manual verification",
        "angles": angles, "labels": labels})


def plot_abinit(frame, output):
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2))
    fig.subplots_adjust(left=.085, right=.985, bottom=.23, top=.82, wspace=.32)
    last = []
    for i, (name, group) in enumerate(frame.groupby("method", sort=False)):
        group = group.sort_values("epoch")
        last.append((name, group.iloc[-1].pose_deg))
        color = COLORS[(i+1) % len(COLORS)]
        if len(group) == 1:
            axes[0].axhline(group.pose_deg.iloc[0], color=color, linestyle="--", label=name)
        else:
            axes[0].plot(group.epoch, group.pose_deg, color=color, label=name)
        axes[1].bar(i, group.iloc[-1].pose_deg, color=color, width=.65)
    axes[0].set(xlabel="Training pass", ylabel="Aligned rotation error (°)", ylim=(0, None))
    axes[1].set(xticks=np.arange(len(last)), xticklabels=[v[0].replace("CryoDyna-optpose", "Optpose") for v in last],
                ylabel="Final aligned error (°)", ylim=(0, None))
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", ncol=3, bbox_to_anchor=(.5, .99))
    save_figure(fig, Path(output) / "abinit_comparison")
    (Path(output)/"abinit_comparison.caption.md").write_text(
        "Full SO(3) error after fitting one global object-frame transformation and handedness on the fixed 20% calibration IDs; "
        "means are calculated on the other 80% IDs. Random rotations are Haar-uniform on SO(3), with a fixed seed. "
        "DRGN-AI and optpose retain their actual training-pass counts. Optpose uses a structural prior; DRGN-AI uses a learned volume. "
        "These prior and training-budget differences are part of the comparison. Bars show each method's final saved pass.\n")
