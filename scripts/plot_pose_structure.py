#!/usr/bin/env python
"""Plot completed four-way diagnostic runs, retaining their oracle labels."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--run", required=True)
parser.add_argument("--output", required=True)
parser.add_argument("--audit-tools", help="Optional publication geometry audit toolkit")
args = parser.parse_args()
run, out = Path(args.run), Path(args.output)
out.mkdir(parents=True, exist_ok=True)
protocol = json.loads((run / "protocol.json").read_text())
if protocol["status"] != "completed":
    raise RuntimeError("Complete every scheduled control before plotting")
rows = list(csv.DictReader((run / "curves.csv").open()))
angles = sorted({float(r["angle"]) for r in rows if r["group"] == "joint"})
groups = ["shape_true_pose", "pose_true_structure", "pose_reference", "joint"]
labels = ["Structure | true pose", "Pose | true structure", "Pose | reference", "Joint VAE + pose"]
colors = ["#777777", "#009E73", "#CC79A7", "#0072B2"]
styles = ["--", "-.", ":", "-"]
metrics = ["loss", "rmsd_A", "pose_deg"]
metric_labels = ["Image correlation loss", "Cα RMSD (Å)", "Full SO(3) error (°)"]
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7.2,
    "axes.labelsize": 7.2, "axes.titlesize": 8, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": .7,
    "svg.fonttype": "none", "pdf.fonttype": 42})
fig, axes = plt.subplots(3, len(angles), figsize=(183/25.4, 139/25.4), sharex=True, sharey="row")
fig.subplots_adjust(left=.10, right=.985, bottom=.09, top=.825, wspace=.27, hspace=.32)
handles = []
for col, angle in enumerate(angles):
    for row, metric in enumerate(metrics):
        ax = axes[row, col]
        for i, group in enumerate(groups):
            this_angle = 0 if group == "shape_true_pose" else angle
            data = sorted((r for r in rows if r["group"] == group and float(r["angle"]) == this_angle),
                          key=lambda r: int(r["epoch"]))
            expected = protocol["arguments"]["epochs"]+1
            if len(data) != expected:
                raise ValueError(f"Missing epochs for {group} {this_angle}")
            line, = ax.plot([int(r["epoch"]) for r in data], [float(r[metric]) for r in data],
                           color=colors[i], linestyle=styles[i], lw=1.1, label=labels[i])
            if col == row == 0:
                handles.append(line)
        if row == 0:
            ax.set_title(f"Input perturbation {angle:g}°", pad=6)
        if col == 0:
            ax.set_ylabel(metric_labels[row])
        if row == 2:
            ax.set_xlabel("Training epoch")
        ax.set_xlim(0, protocol["arguments"]["epochs"])
        ax.set_xticks([0,10,20] if protocol["arguments"]["epochs"]==20 else np.linspace(0,protocol["arguments"]["epochs"],3))
        ax.grid(axis="y", color="#e6ebef", lw=.6)
        ax.text(-.10, 1.08, chr(ord("a")+row*len(angles)+col), transform=ax.transAxes,
                fontweight="bold", fontsize=8, va="bottom")
fig.suptitle("1AKE pose / structure controls | 128 development particles", fontsize=9, y=.995)
fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.53,.95), ncol=2,
           frameon=False, handlelength=2.5, columnspacing=2.0)
fig.text(.10,.008,"Means from one seed; shared true-pose structure control; oracle diagnostics.",fontsize=7)
fig.canvas.draw()
rects = [[float(v) for v in ax.get_position().bounds] for ax in axes.flat]
assert np.ptp([r[2] for r in rects]) < 1e-6 and np.ptp([r[3] for r in rects]) < 1e-6
if args.audit_tools:
    sys.path.insert(0,args.audit_tools)
    from audit_panel_alignment import require_matplotlib_panel_alignment
    require_matplotlib_panel_alignment(fig,json_out=out/"learning_curves.alignment.json",
        overlay_svg=out/"learning_curves.alignment.svg",strict=True,tolerance_pt=1.5,gutter_tolerance_pt=1.5)
else:
    (out/"learning_curves.geometry.json").write_text(json.dumps({"axes":rects,"full_audit":"not_requested"},indent=2))
for ext in ("svg", "pdf", "png"):
    fig.savefig(out/f"learning_curves.{ext}",dpi=600)
plt.close(fig)
shutil.copy2(run/"curves.csv",out/"source_data.csv")
source = {"archetype":"quantitative grid", "run":str(run.resolve()), "seed":protocol["arguments"]["seed"],
    "particles":protocol["particles"],"oracle_scope":protocol["scope"],
    "caption":"Four-way 1AKE diagnostic. Rows show image loss, per-particle Kabsch CA RMSD and raw full SO(3). "
        "Columns show supplied perturbations. The single true-pose/learned-structure control is shared across columns. "
        "True-structure pose controls explicitly use oracle coordinates. All 13 conditions use the same 128 particles, "
        "seed=1, 128 reference warmup steps and 20 training epochs (80 updates). Means are single-seed descriptive results. "
        "This diagnostic does not establish pose-free recovery or full-data scientific acceptance.",
    "input_sha256":hashlib.sha256((run/"curves.csv").read_bytes()).hexdigest(),
    "plot_source_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    "uncertainty":"single-seed means; repeated-seed uncertainty pending"}
(out/"source_record.json").write_text(json.dumps(source,indent=2)+"\n")
(out/"caption.md").write_text(source["caption"]+"\n")
