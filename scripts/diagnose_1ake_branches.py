"""Post-hoc search-path audit of the 18 previously measured >90 degree tails."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import time

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO/'scripts')]
import mrcfile
import numpy as np
import torch
from cryodyna.optpose.benchmark import resolve_config, make_dataset, read_star
from cryodyna.optpose.volume_prior import FourierVolume
from cryodyna.utils import so3_grid, lie_tools
from diagnose_1ake_refinement import search, score, angles, write_csv

OLD = REPO/'output/pose_refinement_20261010/validation128'
RADII = (12, 17, 22, 27, 32)
NEAR = 20.0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    OUT = args.output
    OUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    rows = list(csv.DictReader((OLD/'stages.csv').open()))
    tails = [int(r['particle']) for r in rows if r['condition'] == 'observed'
             and r['variant'] == 'growing_k8' and int(r['stage']) == 4
             and float(r['angle_deg']) > 90]
    assert len(tails) == 18 and len(set(tails)) == 18
    protocol = dict(particle_ids=tails, selection='post-hoc final angle >90 deg in locked128',
                    scope='development diagnostic; true-class density oracle; zero translations',
                    radii=RADII, beam=8, coarse_size=4608, children_per_round=64,
                    near_threshold_deg=NEAR,
                    truth_access='angles and ranks computed after complete unchanged search',
                    source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    assert not (OUT/'protocol.json').exists(), 'Use a fresh directory to preserve evidence'
    (OUT/'protocol.json').write_text(json.dumps(protocol, indent=2)+'\n')
    cfg = resolve_config(str(REPO/'projects/atom_configs/1ake_optpose.py'))
    dataset = make_dataset(cfg, 'given')
    _, frame = read_star(cfg.dataset_attr.starfile_path)
    saved = np.load(OLD/'beams.npz')
    coarse_rot = lie_tools.quaternions_to_SO3(torch.from_numpy(so3_grid.grid_SO3(2)))
    stage_rows, candidates_rows, parent_rows, arrays = [], [], [], {}
    replay_max = 0.
    started = time.monotonic()
    with torch.no_grad():
        for i, idx in enumerate(tails):
            meta = frame.iloc[idx]
            cp = [128, 1, float(meta.rlnDefocusU), float(meta.rlnDefocusV),
                  float(meta.rlnDefocusAngle), 300, 2.7, .1, 0]
            volume = REPO/'tutorial_data_1ake/mrcs'/f'1akeA_{int(meta.rlnClassNumber)}.mrc'
            models = {r: FourierVolume(volume, r, cp) for r in RADII}
            for model in models.values():
                model.ctf.neg_()
            k, name = str(meta.rlnImageName).split('@')
            with mrcfile.mmap(str(Path(cfg.dataset_attr.dataset_dir)/name), permissive=True) as f:
                image = torch.from_numpy(f.data[int(k)-1].copy())[None]
            cache = {}
            beams, records = search(models, image, RADII, 8, cache)
            actual = torch.stack(beams).numpy()
            difference = float(np.max(np.abs(actual-saved[f'{idx}_observed_growing_k8'])))
            replay_max = max(replay_max, difference)
            assert difference == 0., (idx, difference)
            # Evaluation starts only after every search selection has completed.
            truth = dataset[idx]['rotmat']
            previous_near = None
            for stage, radius in enumerate(RADII):
                rotations = coarse_rot if stage == 0 else torch.from_numpy(records[stage-1]['children'])
                values = cache[radius] if stage == 0 else torch.from_numpy(records[stage-1]['child_scores'])
                keep = values.topk(8).indices.numpy()
                values = values.numpy()
                error = angles(rotations, truth)
                order = np.argsort(-values, kind='stable')
                rank = np.empty(len(order), dtype=int)
                rank[order] = np.arange(1, len(order)+1)
                near = error <= NEAR
                best_near_rank = int(rank[near].min()) if near.any() else None
                nearest = int(error.argmin())
                truth_score = float(score(models[radius], truth[None], image)[0])
                kept_near = bool(near[keep].any())
                stage_rows.append(dict(particle=idx, stage=stage, radius=radius,
                    pool_size=len(error), pool_min_angle_deg=float(error.min()),
                    beam_min_angle_deg=float(error[keep].min()),
                    winner_angle_deg=float(error[keep[0]]),
                    near_in_pool=int(near.any()), near_in_beam=int(kept_near),
                    near_pool_count=int(near.sum()), near_beam_count=int(near[keep].sum()),
                    best_near_score_rank=best_near_rank, nearest_candidate_score_rank=int(rank[nearest]),
                    winner_cosine=float(values[keep[0]]),
                    nearest_candidate_cosine=float(values[nearest]), truth_cosine=truth_score,
                    diagnostic_truth_rank=1+int((values > truth_score).sum()),
                    near_pruned=int(near.any() and not kept_near),
                    near_lost_at_expansion=int(previous_near is True and not near.any())))
                for j in range(len(error)):
                    candidates_rows.append(dict(particle=idx, stage=stage, candidate=j,
                        parent_beam=-1 if stage == 0 else j//8,
                        score_rank=int(rank[j]), retained=int(j in keep), cosine=float(values[j]),
                        diagnostic_angle_deg=float(error[j])))
                if stage:
                    rec = records[stage-1]
                    parent_errors = angles(rec['parent'], truth)
                    for p in range(8):
                        child = slice(p*8, (p+1)*8)
                        best = p*8+int(values[child].argmax())
                        parent_rows.append(dict(particle=idx, stage=stage, radius=radius, parent=p,
                            parent_angle_deg=float(parent_errors[p]),
                            parent_cosine=float(rec['parent_scores'][p]),
                            best_child_angle_deg=float(error[best]), best_child_cosine=float(values[best]),
                            child_min_angle_deg=float(error[child].min()),
                            kept_children=int(np.sum((keep >= p*8)&(keep < (p+1)*8)))))
                arrays[f'{idx}_{stage}_rotations'] = rotations.numpy()
                arrays[f'{idx}_{stage}_scores'] = values
                arrays[f'{idx}_{stage}_keep'] = keep
                previous_near = kept_near
            print(f'{i+1}/18 particle={idx} replay={difference} seconds={time.monotonic()-started:.1f}', flush=True)
    write_csv(OUT/'stages.csv', stage_rows)
    write_csv(OUT/'candidates.csv', candidates_rows)
    write_csv(OUT/'parents.csv', parent_rows)
    np.savez_compressed(OUT/'candidates.npz', **arrays)
    summary = dict(protocol=protocol, replay_max_abs=replay_max,
                   runtime_seconds=time.monotonic()-started, stages=[])
    for stage in range(5):
        subset = [r for r in stage_rows if r['stage'] == stage]
        summary['stages'].append(dict(stage=stage, radius=RADII[stage],
             near_in_pool=sum(r['near_in_pool'] for r in subset),
             near_in_beam=sum(r['near_in_beam'] for r in subset),
             near_pruned=sum(r['near_pruned'] for r in subset),
             near_lost_at_expansion=sum(r['near_lost_at_expansion'] for r in subset)))
    summary['final_near_beam_ids'] = [r['particle'] for r in stage_rows if r['stage'] == 4 and r['near_in_beam']]
    (OUT/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
