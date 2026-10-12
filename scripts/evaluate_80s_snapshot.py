"""Evaluate a verified completed-run snapshot with a fixed seed cohort per curve."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import pickle
import shlex
import subprocess
import sys

os.environ.setdefault('OPENBLAS_NUM_THREADS', '4')
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation
import yaml
from cryodyna.optpose import volume_metrics


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--snapshot', type=Path, required=True)
    p.add_argument('--volume-control', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    cache = args.output/'metrics'
    cache.mkdir(exist_ok=True)
    snapshot = json.loads((args.snapshot/'snapshot.json').read_text())
    seeds = snapshot['completed_seeds']
    assert seeds and len(set(seeds)) == len(seeds)
    for seed in seeds:
        assert sorted(r['epoch'] for r in snapshot['checkpoints'] if r['seed'] == seed) == list(range(30))
    truth_path = args.snapshot/'evaluation_truth/rotations.pkl'
    ids_path = args.snapshot/'evaluation_truth/indices.npy'
    truth = np.asarray(pickle.load(truth_path.open('rb')), dtype=np.float64)
    ids = np.load(ids_path)
    assert truth.shape == (100000, 3, 3) and ids.shape == (100000,)
    assert np.array_equal(ids, np.arange(len(truth))), 'Prepared subset must retain stored particle order'
    manifest = json.loads((args.volume_control/'manifest.json').read_text())
    assert Path(manifest['particles']).resolve() == (args.snapshot/'inputs/particles.mrcs').resolve()
    input_hashes = {}
    for name in ('particles.mrcs', 'ctf.pkl'):
        remote_path = snapshot['remote_root']+'/inputs/'+name
        remote_sha = subprocess.check_output([
            'ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=8', '-o', 'ClearAllForwardings=yes',
            snapshot['source_host'], 'sha256sum -- '+shlex.quote(remote_path)], text=True).split()[0]
        local_sha = digest(args.snapshot/'inputs'/name)
        assert remote_sha == local_sha, f'Training/evaluation input mismatch: {name}'
        input_hashes[name] = local_sha
    config_audit = {}
    for seed in seeds:
        for name in ('configs.yaml', 'out/drgnai-configs.yaml'):
            rel = f'seed{seed}/{name}'
            file = args.snapshot/rel
            assert digest(file) == snapshot['files'][rel]['sha256']
            config = yaml.safe_load(file.read_text())
            config = config.get('training', config)
            assert config['pose'] is None and config['no_trans'] is False
            for key in ('use_gt_poses', 'use_gt_trans', 'pretrain_with_gt_poses', 'refine_gt_poses'):
                assert config[key] is False, (rel, key)
            config_audit[rel] = dict(sha256=digest(file), pose=None, gt_pose_flags=False, learn_trans=True)
    common = dict(truth_sha256=digest(truth_path), ids_sha256=digest(ids_path), inputs=input_hashes,
                  evaluator_sha256=digest(Path(volume_metrics.__file__)),
                  analysis_sha256=digest(Path(__file__)))
    volume_metrics.validate()
    rows = []

    def measure(label, seed, epoch, rotations, source, translations=None):
        provenance = {**common, **source, 'method': label, 'seed': seed, 'epoch': epoch}
        key = hashlib.sha256(json.dumps(provenance, sort_keys=True).encode()).hexdigest()
        filename = cache/f'{key}.json'
        if filename.exists():
            result = json.loads(filename.read_text())
            assert result['provenance'] == provenance
            assert (result['method'], result['seed'], result['epoch'], result['n_particles']) == (label, seed, epoch, len(truth))
        else:
            rotations = np.asarray(rotations, dtype=np.float64)
            assert rotations.shape == truth.shape and np.isfinite(rotations).all()
            assert np.max(np.abs(np.linalg.det(rotations)-1)) < .002
            assert np.max(np.abs(rotations@rotations.transpose(0, 2, 1)-np.eye(3))) < .002
            result = dict(method=label, seed=seed, epoch=epoch, n_particles=len(truth),
                          provenance=provenance, alignments=volume_metrics.fit(rotations, truth))
            raw_view, raw_so3 = volume_metrics.angles(rotations, truth)
            result.update(raw_view_mean_deg=float(raw_view.mean()), raw_so3_mean_deg=float(raw_so3.mean()))
            if translations is not None:
                translations = np.asarray(translations, dtype=float)
                assert translations.shape == (len(truth), 2) and np.isfinite(translations).all()
                result['raw_shift_norm_mean_pixels'] = float(np.linalg.norm(translations, axis=1).mean())
                result['shift_scope'] = 'raw learned shift norm; global volume origin remains uncalibrated'
            filename.with_suffix('.tmp').write_text(json.dumps(result, indent=2)+'\n')
            filename.with_suffix('.tmp').replace(filename)
        metric = result['alignments']['paper_mean_frobenius']
        rows.append(dict(method=label, seed=seed, epoch=epoch, n_particles=len(truth),
                         **{k:v for k,v in metric.items() if k != 'right_alignment'},
                         cache_file=filename.name, source_sha256=source.get('sha256', 'seeded Haar draws')))
        print(label, seed, epoch, f"view={metric['view_mean_deg']:.6f} so3={metric['so3_mean_deg']:.6f}", flush=True)

    # All 30 observations from every fully completed run are included.
    for record in snapshot['checkpoints']:
        file = args.snapshot/record['pose']
        assert digest(file) == record['pose_sha256']
        rotations, translations = pickle.load(file.open('rb'))
        measure('DRGN-AI', record['seed'], record['epoch']+1, rotations,
                dict(path=str(file), sha256=record['pose_sha256'], weights_sha256=record['weights_sha256']),
                translations)
    for epoch in range(1, 31):
        file = args.volume_control/f'evaluation/epoch{epoch:03d}.npz'
        with np.load(file) as data:
            assert np.array_equal(data['particle_ids'], ids)
            measure('Optpose (density prior)', 0, epoch, data['rotations'],
                    dict(path=str(file), sha256=digest(file)))
    for seed in range(6):
        rotations = Rotation.random(len(truth), random_state=np.random.default_rng(seed)).as_matrix()
        measure('Random SO(3)', seed, 0, rotations, dict(distribution='Haar SO(3)', seed=seed))
    frame = pd.DataFrame(rows)
    frame.to_csv(args.output/'per_seed_metrics.csv', index=False)
    finals = frame[(frame.method == 'DRGN-AI') & (frame.epoch == 30)]
    status = dict(required_seeds=list(range(6)), completed_seeds=seeds,
                  completed_run_count=len(seeds), complete_30pass_seeds=len(seeds) == 6,
                  n_particles=100000, fixed_cohort_per_epoch=True,
                  completed_view_mean_deg=finals.view_mean_deg.tolist(),
                  completed_so3_mean_deg=finals.so3_mean_deg.tolist(),
                  completed_runs_below5=bool((finals.view_mean_deg < 5).all()),
                  paper_protocol_status='pending fixed-CTF versus lognormal defocus reconciliation',
                  snapshot_sha256=digest(args.snapshot/'snapshot.json'), **common)
    status['training_config_audit'] = config_audit
    (args.output/'status.json').write_text(json.dumps(status, indent=2)+'\n')
    caption = (f'All 100,000 particles; DRGN-AI completed seeds {seeds}, {len(seeds)}/6 planned runs; '
               'every displayed epoch uses this same completed cohort. Mean and sample SD across runs; '
               'fixed-density Optpose is one deterministic trajectory; Random SO(3) uses six Haar seeds. '
               'Global frame and hand selected from the first 100 particle-derived candidates by mean Frobenius norm. '
               'Fixed 15000-A defocus deposited dataset; lognormal-defocus paper fidelity remains pending. '
               'DRGN-AI learns density and poses from images/CTF; Optpose uses a fixed external density. '
               'Seed0 resumes after HPS2 with a new Torch/CUDA environment; other completed seeds start from pretraining.')
    renderer = Path(__file__).with_name('plot_curve_endpoint.py')
    for col, label, name in [('view_mean_deg', 'Viewing-direction error (°)', 'viewing_direction'),
                              ('so3_mean_deg', 'Aligned full SO(3) error (°)', 'full_so3')]:
        aggregate = frame.groupby(['method', 'epoch'])[col].agg(value='mean', sd='std', n='count').reset_index()
        aggregate['sd'] = aggregate.sd.fillna(0)
        aggregate.to_csv(args.output/f'{name}.csv', index=False)
        for suffix, methods in [('', ['Optpose (density prior)', 'Random SO(3)', 'DRGN-AI']),
                                ('_drgnai', ['DRGN-AI'])]:
            subprocess.run([sys.executable, str(renderer), '--csv', str(args.output/f'{name}.csv'),
                            '--dataset', 'Synthetic 80S', '--methods', *methods, '--metric-label', label,
                            '--output', str(args.output/(name+suffix)), '--caption', caption], check=True)
    print(json.dumps(status, indent=2), flush=True)


if __name__ == '__main__':
    main()
