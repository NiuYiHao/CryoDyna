"""Isolated compatibility check with the production checkpoint and batch sizes."""
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import pickle
import shutil
import socket
import time

import numpy as np
import torch
import yaml
import train80s  # applies the existing memory adapter
from src.reconstruct import ModelTrainer

root = Path(__file__).resolve().parents[1]
job = os.environ.get('SLURM_JOB_ID', 'manual')
work = root / 'preflight' / job
work.mkdir(parents=True, exist_ok=False)
result = dict(job_id=job, host=socket.gethostname(), torch=torch.__version__,
              cuda=torch.version.cuda, check='isolated HPS and SGD train steps',
              truth_access='evaluation truth remains on local machine')
try:
    for name in ('torch', 'numpy', 'cryodrgn', 'healpy', 'scipy', 'pandas'):
        result.setdefault('versions', {})[name] = importlib.metadata.version(name)
    assert torch.cuda.is_available()
    result['gpu'] = torch.cuda.get_device_name()
    cfg = yaml.safe_load((root / 'seed0/configs.yaml').read_text())
    assert cfg['pose'] is None
    for key in ('use_gt_poses', 'use_gt_trans', 'pretrain_with_gt_poses', 'refine_gt_poses'):
        assert cfg.get(key, False) is False, key
    (work / 'out').mkdir()
    original = root / 'seed0/out/weights.1.pkl'
    before = hashlib.sha256(original.read_bytes()).hexdigest()
    copied = work / 'out/weights.1.pkl'
    shutil.copy2(original, copied)
    checkpoint = torch.load(copied, map_location='cpu')
    assert checkpoint['epoch'] == 1
    cfg.update(particles=str(root / 'inputs/particles.mrcs'),
               ctf=str(root / 'inputs/ctf.pkl'), load=str(copied), epochs_sgd=25)
    trainer = ModelTrainer(str(work), cfg)
    assert trainer.start_epoch == 2 and trainer.num_epochs == 30
    result['start_epoch'] = trainer.start_epoch
    result['target_epochs'] = trainer.num_epochs
    trainer.total_batch_count = trainer.total_particles_count = 0
    trainer.current_epoch_particles_count = 0
    trainer.batch_idx = 0
    trainer.epoch = 2
    trainer.cur_loss = 0.
    trainer.pose_only = True
    trainer.pretraining = False
    trainer.log_latents = False
    trainer.is_in_pose_search_step = True
    trainer.use_point_estimates = False
    trainer.optimized_modules = ['hypervolume']
    for mode in ('hps', 'sgd'):
        if mode == 'sgd':
            # These are learned poses from HPS, kept separate from evaluation truth.
            with (root / 'seed0/out/pose.1.pkl').open('rb') as handle:
                rotations, translations = pickle.load(handle)
            trainer.model.pose_table.initialize(rotations, translations)
            trainer.model.to(trainer.device)
            trainer.is_in_pose_search_step = False
            trainer.use_point_estimates = True
            trainer.optimized_modules = ['hypervolume', 'pose_table']
            trainer.epoch = 5
            trainer.output_mask.update_radius(trainer.resolution // 2)
        generator = (trainer.data_generator_pose_search if mode == 'hps'
                     else trainer.data_generator_latent_optimization)
        batch = next(iter(generator))
        n = len(batch['y'])
        loss_before = trainer.cur_loss
        torch.cuda.reset_peak_memory_stats()
        started = time.monotonic()
        trainer.train_step(batch, end_time=time.time())
        torch.cuda.synchronize()
        loss = (trainer.cur_loss - loss_before) / n
        assert np.isfinite(loss)
        result[mode] = dict(batch_size=n, loss=loss,
                           seconds=time.monotonic() - started,
                           peak_bytes=torch.cuda.max_memory_allocated())
        print(json.dumps({mode: result[mode]}), flush=True)
    assert hashlib.sha256(original.read_bytes()).hexdigest() == before
    result['checkpoint_sha256'] = before
    result['status'] = 'pass'
except Exception as exc:
    result.update(status='failed', error=repr(exc))
    raise
finally:
    (work / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2), flush=True)
