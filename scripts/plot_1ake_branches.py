"""Measured branch-pruning evidence for the post-hoc 18-particle tail."""
import argparse
import json
from pathlib import Path
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
import numpy as np
import pandas as pd

sys.path.insert(0, '/home/nyh/.codex/skills/nature-figure/scripts')
from audit_panel_alignment import require_matplotlib_panel_alignment

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--input', type=Path, required=True)
p = parser.parse_args().input
data = pd.read_csv(p/'stages.csv')
summary = json.loads((p/'summary.json').read_text())
ids = summary['protocol']['particle_ids']
values = data.pivot(index='particle', columns='stage', values='beam_min_angle_deg').loc[ids].to_numpy()
pool = [r['near_in_pool'] for r in summary['stages']]
kept = [r['near_in_beam'] for r in summary['stages']]
contract = dict(question='At which stage are near-truth hypotheses pruned in the large-angle tail?',
    conclusion='At radius22, ten of thirteen eligible particle pools lose near-truth candidates at top8; radius27 leaves zero.',
    panels={'a':'Per-particle retained-beam accuracy', 'b':'Candidate-pool versus retained-beam counts'},
    selection='All 18 final >90deg particles selected post hoc from locked128',
    statistics='Exact descriptive counts; near means full SO3 angle <=20deg; no inferential intervals',
    dimensions_inches=[7.2,4.65], archetype='quantitative grid', backend='Python/matplotlib',
    scope='true-class density oracle, original images, zero translations; development diagnostic')
(p/'figure_contract.json').write_text(json.dumps(contract, indent=2)+'\n')
plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['DejaVu Sans'],
    'font.size':7,'axes.spines.top':False,'axes.spines.right':False,
    'axes.linewidth':.7,'pdf.fonttype':42,'svg.fonttype':'none'})
fig, axs = plt.subplots(1,2,figsize=(7.2,4.65))
colors=['#b7e4d1','#68b79f','#dbe8ae','#f1d899','#e5a074','#bb6366']
cmap=ListedColormap(colors)
norm=BoundaryNorm([0,5,10,20,45,90,181],len(colors))
mesh=axs[0].imshow(values, cmap=cmap,norm=norm,aspect='auto')
for i in range(18):
    for j in range(5):
        axs[0].text(j,i,f'{values[i,j]:.0f}',ha='center',va='center',fontsize=6.4)
axs[0].set_yticks(np.arange(18),[str(i) for i in ids],fontsize=6.4)
axs[0].set_xticks(range(5),['12','17','22','27','32'])
axs[0].set_xlabel('Search frequency radius (pixels)')
axs[0].set_ylabel('Particle ID')
axs[0].set_title('Closest retained candidate · SO(3) error (°)',fontsize=7.5,pad=12)
axs[0].set_xticks(np.arange(-.5,5,1),minor=True)
axs[0].set_yticks(np.arange(-.5,18,1),minor=True)
axs[0].grid(which='minor',color='white',linewidth=.5)
axs[0].tick_params(which='minor',bottom=False,left=False)
axs[0].spines[['left','bottom']].set_visible(False)
x=np.arange(5)
axs[1].bar(x-.17,pool,width=.32,color='#8bb6d3',label='Candidate pool')
axs[1].bar(x+.17,kept,width=.32,color='#316b95',label='Retained top 8')
for xx,a,b in zip(x,pool,kept):
    axs[1].text(xx-.17,a+.25,str(a),ha='center',fontsize=7)
    axs[1].text(xx+.17,b+.25,str(b),ha='center',fontsize=7)
axs[1].set_xticks(x,['12','17','22','27','32'])
axs[1].set_yticks([0,3,6,9,12,15,18])
axs[1].set_ylim(0,23)
axs[1].set_xlabel('Search frequency radius (pixels)')
axs[1].set_ylabel('Particles with a candidate within 20°')
axs[1].set_title('Where near-truth candidates are pruned',fontsize=7.5,pad=12)
axs[1].legend(loc='upper right',frameon=False,fontsize=6.5)
for i,ax in enumerate(axs):
    ax.text(-.2,1.06,chr(97+i),transform=ax.transAxes,weight='bold',fontsize=10)
fig.suptitle('1AKE: tracing the 18 large-angle failures',y=.98,fontsize=10)
fig.text(.5,.055,'All 18 particles ended above 90°; selected from the same 128-particle diagnostic.\n'
    'True-class density; zero translation; original search reproduced exactly. Truth labels are used after search.',
    ha='center',fontsize=6.4)
fig.subplots_adjust(left=.12,right=.985,top=.84,bottom=.18,wspace=.48)
require_matplotlib_panel_alignment(fig,json_out=str(p/'branch_trace.alignment.json'),strict=True)
fig.savefig(p/'branch_trace.pdf')
fig.savefig(p/'branch_trace.svg')
fig.savefig(p/'branch_trace.png',dpi=600)
fig.savefig(p/'branch_trace.tiff',dpi=600)
