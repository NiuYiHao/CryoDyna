"""Sequential six-seed continuation using the original global batch budgets."""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import torch
import yaml

root = Path(__file__).resolve().parents[1]
lock = (root / 'queue.lock').open('w')
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)


def record(**entry):
    entry.update(time=time.strftime('%Y-%m-%dT%H:%M:%S%z'),
                 job_id=os.environ.get('SLURM_JOB_ID'), torch=torch.__version__)
    path = root / 'status.json'
    path.with_suffix('.tmp').write_text(json.dumps(entry, indent=2) + '\n')
    path.with_suffix('.tmp').replace(path)
    print(json.dumps(entry), flush=True)


seeds = ([int(os.environ['SLURM_ARRAY_TASK_ID'])]
         if 'SLURM_ARRAY_TASK_ID' in os.environ else range(6))
for seed in seeds:
    assert 0 <= seed < 6
    run = root / f'seed{seed}'
    config_path = run / 'configs.yaml'
    cfg = yaml.safe_load(config_path.read_text())
    assert cfg['pose'] is None
    for key in ('use_gt_poses', 'use_gt_trans', 'pretrain_with_gt_poses', 'refine_gt_poses'):
        assert cfg.get(key, False) is False, key
    backup = run / 'configs.before_cluster.yaml'
    if not backup.exists():
        backup.write_text(config_path.read_text())
    checkpoints = sorted((run / 'out').glob('weights.*.pkl'),
                         key=lambda p: int(p.stem.split('.')[-1]))
    start = -1
    cfg.pop('load', None)
    if checkpoints:
        latest = checkpoints[-1]
        checkpoint = torch.load(latest, map_location='cpu')
        epoch = int(checkpoint['epoch'])
        assert int(latest.stem.split('.')[-1]) == epoch
        assert 'optimizers_state_dict' in checkpoint
        if epoch >= 0:
            assert (run / f'out/pose.{epoch}.pkl').is_file()
        if epoch >= 29:
            continue
        start = epoch + 1
        cfg['load'] = str(latest)
    cfg.update(particles=str(root / 'inputs/particles.mrcs'),
               ctf=str(root / 'inputs/ctf.pkl'), epochs_sgd=30-max(start, 5))
    config_path.write_text(yaml.safe_dump(cfg))
    record(state='training', seed=seed, next_epoch=start, target_epochs=30,
           hps_epochs_remaining=max(5-max(start, 0), 0),
           sgd_epochs_remaining=cfg['epochs_sgd'], checkpoint=cfg.get('load'),
           provenance='deposited fixed-CTF 100k; original input and adapter retained',
           resume_boundary='upstream checkpoint omits RNG; CUDA/torch environment changed')
    with (root / f'logs/seed{seed}.log').open('a') as handle:
        rc = subprocess.call([sys.executable, '-u', str(root/'code/train80s.py'),
                              'train', str(run), '--no-analysis'],
                             stdout=handle, stderr=subprocess.STDOUT)
    if rc:
        record(state='failed', seed=seed, returncode=rc)
        sys.exit(rc)
completed = [i for i in range(6) if (root/f'seed{i}/out/weights.29.pkl').is_file()
             and (root/f'seed{i}/out/pose.29.pkl').is_file()]
record(state='complete' if len(completed) == 6 else 'seed_complete',
       completed_seeds=completed, completed_seed_count=len(completed))
