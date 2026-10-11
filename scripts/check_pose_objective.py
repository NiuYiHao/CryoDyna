#!/usr/bin/env python
"""Perfect-model objective gradients and reference-frame oracle diagnostics."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from cryodyna.optpose.benchmark import ROOT, resolve_config, make_dataset, make_task, move, read_star, seed_all
from cryodyna.optpose.geometry import angular_error
from cryodyna.optpose.objectives import scaled_masked_cosine_loss
from cryodyna.utils.losses import calc_cor_loss
from cryodyna.utils.polymer import Polymer
from cryodyna.utils.rotation_conversion import axis_angle_to_matrix

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--config",default=str(ROOT/'projects/atom_configs/1ake_optpose.py'))
parser.add_argument("--output",required=True)
parser.add_argument("--particle-id",type=int,default=90)
args=parser.parse_args()
seed_all(1)
cfg=resolve_config(args.config);cfg.model.use_pose_head=False;cfg.use_pose_table=False;cfg.n_imgs_pose_search=0
dataset=make_dataset(cfg,'given');task=make_task(cfg,dataset,'opt','cpu')
batch=move(torch.utils.data.default_collate([dataset[args.particle_id]]),'cpu')
_,frame=read_star(cfg.dataset_attr.starfile_path);class_id=int(frame.iloc[args.particle_id].rlnClassNumber)
pdb_dir=Path(cfg.dataset_attr.ref_pdb_path).parent
coordinates=torch.tensor(Polymer.from_pdb(pdb_dir/f'1akeA_{class_id}.pdb').coord)[None]
result={'scope':'oracle diagnostics; no parameter updates','particle_id':args.particle_id,'class_id':class_id,
        'target':'exact same GMM+CTF projection at truth pose','gradients':{},
        'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
for name,fn in [('legacy',calc_cor_loss),('scaled_cosine',scaled_masked_cosine_loss)]:
    omega=torch.zeros(1,3,requires_grad=True)
    rotations=axis_angle_to_matrix(omega)@batch['rotmat']
    predicted=task._apply_ctf(batch,task._shared_projection(coordinates,rotations),task.lp_mask2d)
    loss=fn(predicted,predicted.detach().clone(),task.mask)
    gradient=torch.autograd.grad(loss,omega)[0]
    result['gradients'][name]={'loss':float(loss.detach()),'gradient_rad':gradient.detach().tolist(),
                             'gradient_norm':float(gradient.norm())}
reference=task.gmm_centers.detach().numpy().astype(float);rotations=[];offsets=[]
for i in range(1,51):
    p=Polymer.from_pdb(pdb_dir/f'1akeA_{i}.pdb').coord.astype(float)
    u,_,vh=np.linalg.svd((p-p.mean(0)).T@(reference-reference.mean(0)))
    correction=np.eye(3);correction[-1,-1]=np.linalg.det(u@vh)
    rotations.append(u@correction@vh);offsets.append(float(np.linalg.norm(p.mean(0)-reference.mean(0))))
angle=angular_error(np.stack(rotations),np.repeat(np.eye(3)[None],50,axis=0))
result['reference_CA_frame']={'mean_rotation_deg':float(angle.mean()),'max_rotation_deg':float(angle.max()),
    'mean_centroid_displacement_A':float(np.mean(offsets)),'max_centroid_displacement_A':float(np.max(offsets)),
    'scope':'ground-truth structure Kabsch transforms to supplied reference; separate diagnostic from raw pose accuracy'}
out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
