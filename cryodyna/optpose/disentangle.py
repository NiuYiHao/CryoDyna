"""Four-way development controls for pose/structure optimization.

This deliberately uses oracle pose or structures in named diagnostic controls.
It is neither a pose-free training benchmark nor a scientific acceptance test.
The learned structure branch is the actual CryoDyna VAE with its original losses.
"""
import argparse
import copy
import csv
import hashlib
import json
from pathlib import Path
import random
import time

import numpy as np
import torch

from .benchmark import (ROOT, resolve_config, read_star, prepare_inputs, make_dataset,
    make_task, seed_all, move, loader, pretrain, task_state, load_task_state,
    truth_data, loss_terms, set_mode, sha256, write_json, save_checkpoint)
from .geometry import angular_error, kabsch_rmsd, perturb_rotations
from .objectives import scaled_masked_cosine_loss
from cryodyna.utils.losses import calc_cor_loss

GROUPS = ("shape_true_pose", "pose_true_structure", "pose_reference", "joint")


def structure_hash(task):
    state = task_state(task)
    digest = hashlib.sha256()
    for key, value in sorted(state["task"].items()):
        if key.startswith("model."):
            digest.update(key.encode())
            digest.update(value.numpy().tobytes())
    for name, layers in sorted(state["encoder_unregistered"].items()):
        for i, layer in enumerate(layers):
            for key, value in sorted(layer.items()):
                digest.update(f"{name}.{i}.{key}".encode())
                digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def predict(task, batch, group, true_rotations, true_coordinates, initial_rotations):
    ix = batch["idx"].long()
    if group in ("shape_true_pose", "joint"):
        deformation, z = task._shared_forward(batch["proj"])
        structure = task.deformer.transform(deformation, task.gmm_centers)
    else:
        structure = true_coordinates[ix] if group == "pose_true_structure" else task.gmm_centers[None].expand(len(ix), -1, -1)
        z = structure.new_zeros(len(ix), 0)
    if group == "shape_true_pose":
        rotations = true_rotations[ix]
    else:
        rotations, _ = task.pose_table(ix)
    batch = dict(batch, rotmat=initial_rotations[ix])
    image = task._apply_ctf(batch, task._shared_projection(structure, rotations), task.lp_mask2d)
    return image, structure, rotations, z, batch


@torch.no_grad()
def evaluate(task, dataset, args, group, truth_rotations, truth_coordinates,
             initial_rotations, particle_ids, epoch, out):
    rng = (random.getstate(), np.random.get_state(), torch.get_rng_state())
    set_mode(task, False)
    errors, rmsds, rotations, structures, losses, objectives = [], [], [], [], [], []
    for batch in loader(dataset, args.batch_size, 0, workers=0):
        batch = move(batch, args.device)
        image, structure, rot, _, _ = predict(task, batch, group, truth_rotations, truth_coordinates, initial_rotations)
        ix = batch["idx"].cpu().numpy()
        errors.extend(angular_error(rot.cpu().numpy(), truth_rotations[ix].cpu().numpy()))
        rmsds.extend(kabsch_rmsd(structure.cpu().numpy(), truth_coordinates[ix].cpu().numpy()))
        losses.append((len(ix), float(calc_cor_loss(image, task.low_pass_images(batch["proj"]), task.mask))))
        objective = scaled_masked_cosine_loss(image, task.low_pass_images(batch["proj"]), task.mask) if args.objective == "cosine" else calc_cor_loss(image, task.low_pass_images(batch["proj"]), task.mask)
        objectives.append((len(ix), float(objective)))
        rotations.append(rot.cpu().numpy()); structures.append(structure.cpu().numpy())
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / f"epoch{epoch:03}.npz", rotations=np.concatenate(rotations),
        coordinates=np.concatenate(structures), raw_pose_deg=errors, rmsd_A=rmsds, particle_ids=particle_ids)
    result = {"epoch": epoch, "loss": sum(n*v for n,v in losses)/len(dataset),
        "objective_loss": sum(n*v for n,v in objectives)/len(dataset),
        "rmsd_A": float(np.mean(rmsds)), "pose_deg": float(np.mean(errors)),
        "pose_p90_deg": float(np.quantile(errors, .9))}
    write_json(out / f"epoch{epoch:03}.json", result)
    random.setstate(rng[0]); np.random.set_state(rng[1]); torch.set_rng_state(rng[2])
    return result


def run(args):
    out = Path(args.output).resolve()
    if (out / "protocol.json").exists():
        raise FileExistsError("Choose a fresh output for this development matrix")
    out.mkdir(parents=True, exist_ok=True)
    seed_all(args.seed)
    cfg = resolve_config(args.config)
    truth_star = cfg.dataset_attr.starfile_path
    _, frame = read_star(truth_star)
    ids = np.asarray(json.loads(Path(args.particle_ids).read_text()), dtype=int)
    if ids.ndim != 1 or len(set(ids)) != len(ids) or min(ids) < 0 or max(ids) >= len(frame):
        raise ValueError("Expected unique dataset particle row IDs")
    ids = np.sort(ids)
    names = prepare_inputs(cfg, out / "inputs", "hps", 0, args.seed, ids)
    cfg.dataset_attr.starfile_path = str(out / "inputs/particles.star")
    cfg.pose_init = "given"
    cfg.n_imgs_pose_search = 0
    cfg.use_pose_table = True
    cfg.model.use_pose_head = False
    cfg.optimize_translations = False
    cfg.loss.pose_reg_weight = args.pose_reg
    cfg.optimizer.pose_lr = args.pose_lr
    dataset = make_dataset(cfg, "hps")
    task = make_task(cfg, dataset, "opt", args.device)
    # These legacy layers were absent from the original optimizer. Preserve
    # their weights and modes, while avoiding unused parameter-gradient buffers.
    for name in ("attn_layers", "post_norm"):
        for layer in getattr(task.model.encoder, name, []):
            layer.requires_grad_(False)
    manifest = {"truth_star": truth_star, "truth_sha256": sha256(truth_star),
        "particle_ids": ids.tolist(), "image_names": names,
        "ref_pdb": cfg.dataset_attr.ref_pdb_path, "pdb_dir": str(Path(cfg.dataset_attr.ref_pdb_path).parent)}
    truth, classes, coords = truth_data(manifest)
    origins = frame.iloc[ids][["rlnOriginXAngst", "rlnOriginYAngst"]].to_numpy()
    if np.abs(origins).max() > 1e-6:
        raise ValueError("Initial four-way experiment is restricted to centered particles")
    truth_rotations = torch.tensor(truth, dtype=torch.float32, device=args.device)
    truth_coordinates = torch.tensor(np.stack([coords[c] for c in classes]), dtype=torch.float32, device=args.device)
    cfg.dump(str(out / "config.py"))
    pretrain(task, dataset, args)
    # CPU state_dict tensors alias model storage; deepcopy freezes the shared
    # initialization before any control changes parameters.
    base_state = copy.deepcopy(task_state(task))
    base_hash = structure_hash(task)
    save_checkpoint(out / "shared_initialization.pt", base_state)
    protocol = {"status": "running", "scope": "oracle development training controls",
        "metric": "raw full SO(3), per-particle proper Kabsch CA RMSD, existing image correlation loss",
        "arguments": vars(args), "particles": len(ids), "class_count": len(set(classes)),
        "manifest": manifest, "structure_initialization_sha256": base_hash,
        "source_sha256": sha256(__file__), "ctf_sigma": "v2; 2A residue GMM, original residue electron amplitudes",
        "truth_roles": {"shape_true_pose": "truth rotations fixed in forward diagnostic",
            "pose_true_structure": "truth coordinates fixed in forward diagnostic",
            "pose_reference": "reference coordinates fixed",
            "joint": "reference-initialized VAE plus indexed pose table"},
        "initial_pose_source": "controlled perturbation of simulation truth; supplied-pose development test",
        "training_star_sha256": sha256(out / "inputs/particles.star"),
        "optimizer": "AdamW; existing structure/pose parameter groups; pose translation fixed",
        "initialization_shared": True, "model_budget": "one shared reference warmup; identical epoch/order budget", "completed_runs": []}
    write_json(out / "protocol.json", protocol)
    final = []
    for group in args.groups:
        for angle in ([0] if group == "shape_true_pose" else args.angles):
            seed_all(args.seed)
            load_task_state(task, base_state)
            assert structure_hash(task) == base_hash
            learn_structure = group in ("shape_true_pose", "joint")
            learn_pose = group != "shape_true_pose"
            task.model.requires_grad_(learn_structure)
            task.pose_table.rotation.requires_grad_(learn_pose)
            task.pose_table.translation.requires_grad_(False)
            initial = torch.tensor(perturb_rotations(truth, angle, args.seed+77), dtype=torch.float32, device=args.device)
            task.pose_table.initialize(torch.arange(len(ids), device=args.device), initial, initial.new_zeros(len(ids),2))
            optimizer = task.configure_optimizers()
            run_out = out / f"{group}_{angle:g}deg"
            curves = [evaluate(task, dataset, args, group, truth_rotations, truth_coordinates,
                               initial, ids, 0, run_out / "evaluation")]
            started = time.time()
            for epoch in range(args.epochs):
                task.benchmark_epoch = epoch
                set_mode(task, True)
                if not learn_structure:
                    set_mode(task, False)
                for batch in loader(dataset, args.batch_size, args.seed+100+epoch, True, workers=0):
                    batch = move(batch, args.device)
                    optimizer.zero_grad(set_to_none=True)
                    image, structure, _, _, regularization_batch = predict(task, batch, group,
                        truth_rotations, truth_coordinates, initial)
                    image_loss, structural = loss_terms(task, batch["proj"], image, structure)
                    if args.objective == "cosine":
                        image_loss = task.cfg.loss.gmm_cryoem_weight * scaled_masked_cosine_loss(image, task.low_pass_images(batch["proj"]), task.mask)
                    loss = image_loss + structural
                    if learn_pose:
                        loss = loss + task.pose_table_regularization(regularization_batch)
                    if not torch.isfinite(loss):
                        raise RuntimeError("Nonfinite training objective")
                    loss.backward()
                    optimizer.step()
                curves.append(evaluate(task, dataset, args, group, truth_rotations, truth_coordinates,
                                       initial, ids, epoch+1, run_out / "evaluation"))
                print(group, angle, curves[-1], flush=True)
            checkpoint = {"pose": task.pose_table.state_dict(), "optimizer": optimizer.state_dict(),
                "task": task_state(task) if learn_structure else None,
                "shared_initialization_sha256": base_hash, "seed": args.seed,
                "rng": {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state()}}
            save_checkpoint(run_out / "endpoint.pt", checkpoint)
            write_json(run_out / "curves.json", curves)
            result = {"group": group, "angle": angle, "seconds": time.time()-started,
                "initial": curves[0], "final": curves[-1]}
            final.append(result)
            protocol["completed_runs"].append(f"{group}_{angle:g}deg")
            write_json(out / "protocol.json", protocol)
    protocol["status"] = "completed"
    write_json(out / "protocol.json", protocol)
    write_json(out / "summary.json", {"scope": protocol["scope"], "runs": final})
    with (out / "curves.csv").open("w", newline="") as f:
        fields = ["group", "angle", "epoch", "loss", "objective_loss", "rmsd_A", "pose_deg", "pose_p90_deg"]
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for run in final:
            values = json.loads((out / f"{run['group']}_{run['angle']:g}deg/curves.json").read_text())
            for value in values:
                writer.writerow({"group": run["group"], "angle": run["angle"], **value})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "projects/atom_configs/1ake_optpose.py"))
    parser.add_argument("--particle-ids", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--init-steps", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--angles", type=float, nargs="+", default=[0,5,10,20])
    parser.add_argument("--groups", choices=GROUPS, nargs="+", default=list(GROUPS))
    parser.add_argument("--pose-lr", type=float, default=.003)
    parser.add_argument("--pose-reg", type=float, default=.5)
    parser.add_argument("--objective", choices=("legacy", "cosine"), default="legacy")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 2 or args.init_steps < 1:
        parser.error("Use positive epochs/warmup and batch size of at least two")
    if len(set(args.angles)) != len(args.angles) or any(not 0<=a<=180 for a in args.angles):
        parser.error("Use distinct angles between 0 and 180 degrees")
    run(args)
