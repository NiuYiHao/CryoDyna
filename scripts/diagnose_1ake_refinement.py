"""Paired bandwidth/beam controls for the fixed true-class density diagnostic.

Search sees images and density/CTF only. True rotations serve synthetic closure
and evaluation. All variants use the same 32-particle selection as the prior run.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import mrcfile
import numpy as np
import torch
from cryodyna.optpose.benchmark import resolve_config, make_dataset, read_star
from cryodyna.optpose.pose import subdivide
from cryodyna.optpose.volume_prior import FourierVolume, normalized_features
from cryodyna.utils import so3_grid, lie_tools

VARIANTS = {
    "growing_k8": ((12, 17, 22, 27, 32), 8),
    "freeze12_k8": ((12, 12, 12, 12, 12), 8),
    "freeze17_k8": ((12, 17, 17, 17, 17), 8),
    "freeze22_k8": ((12, 17, 22, 22, 22), 8),
    "fixed32_k8": ((32, 32, 32, 32, 32), 8),
    "growing_k32": ((12, 17, 22, 27, 32), 32),
}


def angles(rotations, truth):
    return np.rad2deg(np.arccos(np.clip(
        (np.sum(np.asarray(rotations, dtype=np.float64) *
                np.asarray(truth, dtype=np.float64), axis=(-2, -1)) - 1) / 2,
        -1, 1)))


def score(model, rotations, image):
    y = normalized_features(model.observations(image))
    return torch.cat([normalized_features(model(r)) @ y.T
                      for r in rotations.split(128)]).flatten()


def search(models, image, schedule, beam, coarse_cache):
    """Return beams and scored candidate archives; truth is absent here."""
    quats = torch.from_numpy(so3_grid.grid_SO3(2))
    if schedule[0] not in coarse_cache:
        coarse_cache[schedule[0]] = score(
            models[schedule[0]], lie_tools.quaternions_to_SO3(quats), image)
    top = coarse_cache[schedule[0]].topk(beam).indices
    q = quats[top]
    ind = so3_grid.get_base_ind(top.numpy(), 2)
    beams = [lie_tools.quaternions_to_SO3(q)]
    records = []
    for level, radius in enumerate(schedule[1:]):
        parent = lie_tools.quaternions_to_SO3(q)
        children, grids, rr = subdivide(q, ind, 2 + level, torch.device("cpu"))
        values = score(models[radius], rr, image)
        keep = values.topk(beam).indices
        q = children.reshape(-1, 4)[keep]
        ind = grids.reshape(-1, 2)[keep.numpy()]
        records.append(dict(parent=parent.numpy(), children=rr.numpy(),
                            parent_scores=score(models[radius], parent, image).numpy(),
                            child_scores=values.numpy()))
        beams.append(lie_tools.quaternions_to_SO3(q))
    return beams, records


def write_csv(path, rows):
    with path.open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--previous", type=Path,
                   default=Path("develop/artifacts/parallel_progress_20261010"))
    p.add_argument("--particle-ids", type=Path, help="JSON list fixed before evaluation")
    p.add_argument("--variants", nargs="+", choices=list(VARIANTS), default=list(VARIANTS))
    p.add_argument("--conditions", nargs="+", choices=["observed", "clean"],
                   default=["observed", "clean"])
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "summary.json").exists():
        raise FileExistsError("Choose a fresh output directory to retain evidence")
    torch.set_num_threads(4)
    cfg = resolve_config("projects/atom_configs/1ake_optpose.py")
    dataset = make_dataset(cfg, "given")
    _, frame = read_star(cfg.dataset_attr.starfile_path)
    ids = (json.loads(args.particle_ids.read_text()) if args.particle_ids else
           json.loads((args.previous / "known_pose_summary.json").read_text())["particle_ids"])
    variants = {key: VARIANTS[key] for key in args.variants}
    previous = np.load(args.previous / "full_search_predictions.npz")
    previous_ids = {int(idx): i for i, idx in enumerate(previous["particle_ids"])}
    rows, cross, parents, spectral = [], [], [], []
    arrays = {"particle_ids": np.asarray(ids)}
    replay_differences = []
    start = time.time()
    with torch.no_grad():
        for position, idx in enumerate(ids):
            row = frame.iloc[idx]
            truth = dataset[idx]["rotmat"]
            cp = [128, 1, float(row.rlnDefocusU), float(row.rlnDefocusV),
                  float(row.rlnDefocusAngle), 300, 2.7, .1, 0]
            path = Path("tutorial_data_1ake/mrcs") / f"1akeA_{int(row.rlnClassNumber)}.mrc"
            models = {r: FourierVolume(path, r, cp) for r in (12, 17, 22, 27, 32)}
            for model in models.values():
                model.ctf.neg_()
            k, name = str(row.rlnImageName).split("@")
            with mrcfile.mmap(str(Path(cfg.dataset_attr.dataset_dir) / name), permissive=True) as f:
                observed = torch.from_numpy(f.data[int(k) - 1].copy())[None]
            spectrum = torch.zeros((1, 128, 128), dtype=torch.complex64)
            spectrum[:, models[32].mask] = models[32](truth[None])
            clean = torch.fft.fftshift(torch.fft.ifft2(torch.fft.ifftshift(
                spectrum, dim=(-2, -1))), dim=(-2, -1)).real
            clean /= clean.std(unbiased=False)
            for condition, image in (("observed", observed), ("clean", clean)):
                if condition not in args.conditions:
                    continue
                cache = {}
                for variant, (schedule, beam) in variants.items():
                    beams, records = search(models, image, schedule, beam, cache)
                    winners = torch.stack([b[0] for b in beams])
                    arrays[f"{idx}_{condition}_{variant}"] = torch.stack(beams).numpy()
                    for stage, radius in enumerate(schedule):
                        archive = torch.cat(beams[:stage + 1])
                        archive_scores = score(models[radius], archive, image)
                        archive_winner = archive[archive_scores.argmax()]
                        rows.append(dict(particle=idx, condition=condition, variant=variant,
                                         stage=stage, radius=radius, beam=beam,
                                         angle_deg=float(angles(winners[stage], truth)),
                                         cosine=float(score(models[radius], winners[stage:stage+1], image)[0]),
                                         truth_cosine=float(score(models[radius], truth[None], image)[0]),
                                         archive_angle_deg=float(angles(archive_winner, truth))))
                        if stage:
                            rec = records[stage-1]
                            parent_best = int(rec["parent_scores"].argmax())
                            child_best = int(rec["child_scores"].argmax())
                            parents.append(dict(particle=idx, condition=condition, variant=variant,
                                                stage=stage, radius=radius,
                                                parent_best_cosine=float(rec["parent_scores"][parent_best]),
                                                child_best_cosine=float(rec["child_scores"][child_best]),
                                                parent_best_angle=float(angles(rec["parent"][parent_best], truth)),
                                                child_best_angle=float(angles(rec["children"][child_best], truth)),
                                                n_scored_children=len(rec["children"])))
                    if variant == "growing_k8":
                        if idx in previous_ids:
                            old = previous["rotations"][previous_ids[idx], :, 4 if condition == "observed" else 0]
                            replay_differences.append(float(np.max(np.abs(old - winners.numpy()))))
                        for radius in models:
                            values = score(models[radius], winners, image)
                            true_value = float(score(models[radius], truth[None], image)[0])
                            for stage, value in enumerate(values):
                                cross.append(dict(particle=idx, condition=condition,
                                                  evaluated_radius=radius, candidate_stage=stage,
                                                  cosine=float(value), truth_cosine=true_value,
                                                  angle_deg=float(angles(winners[stage], truth))))
                # Matched true-pose spectral correlation diagnoses information by annulus.
                model = models[32]
                pred = model(truth[None])[0]
                y = model.observations(image)[0]
                radial = model.coordinates[:, :2].norm(dim=1)
                for lo, hi in ((0, 12), (12, 17), (17, 22), (22, 27), (27, 32)):
                    mask = (radial > lo) & (radial <= hi)
                    dot = (pred[mask].conj()*y[mask]).real.sum()
                    cosine = dot / (pred[mask].norm()*y[mask].norm()).clamp_min(1e-20)
                    spectral.append(dict(particle=idx, condition=condition, low=lo, high=hi,
                                         cosine=float(cosine), pixels=int(mask.sum()),
                                         signal_power=float(pred[mask].abs().square().sum()),
                                         observed_power=float(y[mask].abs().square().sum())))
            print(json.dumps(dict(completed=position+1, total=len(ids), elapsed_s=time.time()-start)), flush=True)
    for name, content in (("stages", rows), ("cross_scores", cross),
                          ("parent_child", parents), ("annuli", spectral)):
        write_csv(args.output / f"{name}.csv", content)
    np.savez_compressed(args.output / "beams.npz", **arrays)
    summary = {}
    for condition in args.conditions:
        summary[condition] = {}
        for variant in variants:
            chosen = [r for r in rows if r["condition"] == condition and
                      r["variant"] == variant and r["stage"] == 4]
            a = np.array([r["angle_deg"] for r in chosen])
            summary[condition][variant] = dict(n=len(a), mean_deg=float(a.mean()),
                median_deg=float(np.median(a)), fraction_below_5=float((a < 5).mean()),
                archive_mean_deg=float(np.mean([r["archive_angle_deg"] for r in chosen])))
    summary["replay_max_rotation_difference"] = max(replay_differences) if replay_differences else None
    summary["protocol"] = dict(particles=ids, scope="fixed true-class density oracle diagnostic",
        metric="un-aligned full SO(3) geodesic degrees; float64 trace evaluation",
        translation="fixed zero, matching centered STAR metadata", variants=variants,
        candidate_children="8 children per parent after quaternion-distance pruning",
        archive="rescore retained beams from all visited stages; output selector only; search path unchanged",
        original_snr="source formula unresolved", elapsed_s=time.time()-start,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
