"""Assemble measured 80S search/refinement and 1AKE method evidence in one figure."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from cryodyna.optpose.plotting import save_figure

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stages', type=Path, default=ROOT/'docs/optpose_evidence/80s_pose_stages_64/source_data.csv')
    p.add_argument('--abinit', type=Path, default=ROOT/'docs/optpose_evidence/abinit_pilot/abinit_metrics.csv')
    p.add_argument('--output', type=Path, default=ROOT/'docs/optpose_evidence/release_summary')
    args = p.parse_args()
    stages, abinit = pd.read_csv(args.stages), pd.read_csv(args.abinit)
    assert stages.n.nunique() == 1
    assert np.isfinite(stages[['stage', 'view_mean_deg', 'so3_mean_deg']]).all().all()
    assert np.isfinite(abinit[['epoch', 'pose_deg']]).all().all()
    grid = stages[stages.phase == 'Grid search'].sort_values('stage')
    adam = stages[stages.phase == 'Adam'].sort_values('stage')
    bridge = grid.iloc[-1:].copy(); bridge['stage'] = 0
    adam = pd.concat([bridge, adam])
    plt.rcParams.update({'font.size': 8, 'axes.labelsize': 8, 'axes.titlesize': 9,
                         'legend.fontsize': 8, 'xtick.labelsize': 7, 'ytick.labelsize': 7})
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.5))
    fig.subplots_adjust(left=.10, right=.975, bottom=.145, top=.835, wspace=.31, hspace=.96)
    for key, label, color, style in [('view_mean_deg', 'Viewing direction', '#0072B2', '-'),
                                     ('so3_mean_deg', 'Full SO(3)', '#009E73', '--')]:
        for ax, data in zip(axes[0], [grid, adam]):
            ax.plot(data.stage, data[key], marker='o', markersize=2.4, color=color, linestyle=style, label=label)
    axes[0, 0].set(xlabel='Search stage', ylabel='Mean angular error (°)', ylim=(0, 8.5),
                  xticks=grid.stage, xticklabels=['Coarse']+[f'Refine {i}' for i in grid.stage.iloc[1:]])
    axes[0, 1].set(xlabel='Adam pass', ylabel='Mean angular error (°)', ylim=(0, .56))
    axes[0, 1].annotate(f"{adam.view_mean_deg.iloc[-1]:.3f}° view\n{adam.so3_mean_deg.iloc[-1]:.3f}° SO(3)",
                         xy=(adam.stage.iloc[-1], adam.view_mean_deg.iloc[-1]), xytext=(16, .36),
                         fontsize=8, arrowprops={'arrowstyle':'-', 'color':'#555555', 'lw':.6})
    names = ['CryoDyna-optpose', 'Random SO(3)', 'DRGN-AI']
    colors = ['#0072B2', '#D55E00', '#009E73']
    for i, (name, color) in enumerate(zip(names, colors)):
        data = abinit[abinit.method == name].sort_values('epoch')
        if data.empty: raise ValueError(f'Measured values required for {name}')
        if len(data) == 1:
            axes[1, 0].axhline(data.pose_deg.iloc[0], color=color, linestyle='--', label=name)
        else:
            axes[1, 0].plot(data.epoch, data.pose_deg, color=color, label=name)
        endpoint = data.pose_deg.iloc[-1]
        axes[1, 1].bar(i, endpoint, width=.65, color=color)
        axes[1, 1].text(i, endpoint+3, f'{endpoint:.2f}°', ha='center', fontsize=7)
    axes[1, 0].set(xlabel='Training pass', ylabel='Aligned full SO(3) error (°)', ylim=(0, 150))
    axes[1, 1].set(ylabel='Final full SO(3) error (°)', ylim=(0, 150),
                  xticks=range(3), xticklabels=['Optpose', 'Random SO(3)', 'DRGN-AI'])
    for i, (ax, title) in enumerate(zip(axes.flat, ['Grid search', 'Adam refinement', 'Learning curves', 'Final endpoints'])):
        ax.set_title(title, loc='center', pad=7)
        ax.text(-.18, 1.07, chr(97+i), transform=ax.transAxes, weight='bold', fontsize=10)
        ax.grid(axis='y', color='#E5E7EB', lw=.5); ax.set_axisbelow(True)
    axes[0, 1].grid(False)
    fig.text(.10, .968, f'80S | fixed external density | {int(stages.n.iloc[0])} particles', fontsize=10, weight='bold')
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc='upper center', ncol=2,
               bbox_to_anchor=(.55, .936))
    fig.text(.10, .496, '1AKE | pose recovery remains inaccurate | 103 evaluation particles', fontsize=9, weight='bold')
    fig.legend(*axes[1, 0].get_legend_handles_labels(), loc='upper center', ncol=3,
               bbox_to_anchor=(.54, .477))
    fig.text(.10, .025, '80S: per-stage global alignment; panel a/b use different error ranges.\n'
             '1AKE: 25 gauge-fit particles; actual method budgets retained. 80S DRGN-AI evaluation pending.', fontsize=6.6)
    out = args.output; out.mkdir(parents=True, exist_ok=True)
    save_figure(fig, out/'80s_1ake_evidence')
    stages.to_csv(out/'80s_source.csv', index=False)
    abinit.to_csv(out/'1ake_source.csv', index=False)
    caption = '''# 80S pose-search control and the 1AKE failure case

(a,b) Fixed external 80S density, first 64 prepared synthetic particles, seed 0.
All particles are retained through coarse Hopf search, four refinements and 30
Adam passes. Adam pass 0 repeats the last search estimate. Axes have different
ranges. The evaluator chooses one global frame correction and whole-dataset
hand per stage using min(100,N) candidate corrections and mean Frobenius error;
fit and evaluation use the same 64 particles. Viewing-direction and complete
SO(3) angles are separate metrics. The reference volume is an input, so this
control measures pose recovery conditional on a known density.

(c,d) 1AKE reference-structure Optpose (128 training particles, 5 passes), learned
volume DRGN-AI (50,000 training particles, 31 passes) and Haar random SO(3).
The same 25 calibration particles fit each global frame/hand; the other 103
particles supply all plotted means. Actual training durations and priors differ.
All three endpoints remain far above the mean ≤5° accuracy target. Each method
has one measured trajectory; uncertainty over training repetitions remains pending.

The 80S DRGN-AI replication was in its first HPS epoch at this release snapshot.
Its measured curve/endpoints are pending. The published Fig. 1c uses viewing-direction
error (six runs), DOI: https://doi.org/10.1038/s41592-025-02720-4.
Sub-degree accuracy of the fixed-density control establishes a conditional
pose-fitting result; strict neural ab-initio replication remains an open gate.

Rebuild: `python scripts/figure_optpose_release.py` from the repository root.
The source tables and SHA256 records accompany this figure.
'''
    (out/'caption.md').write_text(caption)
    (out/'provenance.json').write_text(json.dumps({'sources': [
        {'path':str(path), 'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
        for path in (args.stages, args.abinit)], 'drgnai_80s':'first HPS epoch; metrics pending',
        'release_claim':'development evidence; scientific acceptance partial'}, indent=2)+'\n')


if __name__ == '__main__': main()
