"""Read historical CryoDyna Output dirs without altering the input artifacts."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from .benchmark import (resolve_config, read_star, make_dataset, make_task, seed_all, set_mode,
    write_json, sha256, truth_data, move, loader)
from .geometry import angular_error, star_rotations, kabsch_rmsd, fit_gauge, validate_rotations


def import_legacy(outdir, output, method, truth_star, pdb_dir, device, snapshot_epochs=None):
    source, output = Path(outdir).resolve(), Path(output).resolve()
    if output == source: raise ValueError("Choose a separate evaluation output directory")
    snapshots = sorted(source.glob("[0-9]*_[0-9]*/ckpt.pt"), key=lambda p: tuple(map(int, p.parent.name.split("_"))))
    if snapshot_epochs is not None:
        def saved_epoch(path):
            epoch, step = map(int, path.parent.name.split("_"))
            return 0 if step == 0 else epoch+1
        snapshots = [p for p in snapshots if saved_epoch(p) in snapshot_epochs]
        if set(map(saved_epoch, snapshots)) != set(snapshot_epochs): raise ValueError("Requested historical epochs are missing")
    if not snapshots: raise ValueError("Expected config.py and epoch_step/ckpt.pt in the Output dir")
    missing = [str(p.parent) for p in snapshots if not (p.parent/"z.npy").exists()]
    if missing: raise ValueError(f"Saved latent codes are required: {missing}")
    seed_all(1)
    first = torch.load(snapshots[0], map_location="cpu", weights_only=False)
    method = method or ("opt" if "pose_head" in first or "hps_pose" in first else "original")
    cfg = resolve_config(source/"config.py")
    search_images = cfg.get("n_imgs_pose_search", 0)
    mode = "hps" if cfg.get("pose_init") == "hps" or search_images > 0 else "given"
    cfg.extra_input_data_attr.ckpt_path = None
    cfg.n_imgs_pose_search = 0
    cfg.use_pose_table = False
    dataset = make_dataset(cfg, mode)
    _, actual = read_star(cfg.dataset_attr.starfile_path)
    _, truth_frame = read_star(truth_star)
    lookup = {v: i for i, v in enumerate(truth_frame.rlnImageName)}
    if len(lookup) != len(truth_frame): raise ValueError("Ground truth has duplicate particle identities")
    ids = np.array([lookup[v] for v in actual.rlnImageName])
    if len(ids) < 2: raise ValueError("At least two particles are required")
    true_rotations = star_rotations(truth_frame.iloc[ids])
    base = star_rotations(actual) if mode == "given" else np.tile(np.eye(3), (len(ids), 1, 1))
    initial_error = angular_error(base, true_rotations)
    angle = float(np.median(initial_error)) if mode == "given" else 0.
    if mode == "given" and np.max(np.abs(initial_error-angle)) > .05:
        raise ValueError("Initial perturbations differ across particles; import as an explicitly specified experiment")
    angle = round(angle, 4)
    calibration = np.random.default_rng(102).permutation(len(ids))[:max(1, len(ids)//5)]
    manifest = {"schema": 1, "kind": "run", "method": method, "pose_init": mode, "angle": angle,
        "seed": cfg.get("seed", 1), "particle_ids": ids.tolist(), "image_names": actual.rlnImageName.tolist(),
        "alignment_ids": calibration.tolist(), "evaluation_ids": np.setdiff1d(np.arange(len(ids)), calibration).tolist(),
        "truth_star": str(Path(truth_star).resolve()), "truth_sha256": sha256(truth_star),
        "pdb_dir": str(Path(pdb_dir).resolve()), "ref_pdb": cfg.dataset_attr.ref_pdb_path,
        "source_outdir": str(source), "status": "historical_import", "epochs": cfg.trainer.max_epochs,
        "row_order_contract": "Saved z rows follow sorted unique dataset indices, as in CryoDyna validation hook"}
    if method == "opt":
        # Historical snapshots without learned pose state require rerunning training.
        for path in snapshots:
            state = torch.load(path, map_location="cpu", weights_only=False)
            if "pose_head" not in state and "hps_pose" not in state and "pose_table" not in state:
                raise ValueError(f"Missing learned pose state at {path}; rerun this experiment with complete checkpoints")
    task = make_task(cfg, dataset, method, device)
    truth, classes, structures = truth_data(manifest)
    output.mkdir(parents=True, exist_ok=True)
    evaluation = output/"evaluation"; evaluation.mkdir(exist_ok=True)
    eval_ids = np.asarray(manifest["evaluation_ids"])
    rows = []
    from cryodyna.utils.rotation_conversion import axis_angle_to_matrix
    from cryodyna.utils.losses import calc_cor_loss
    from cryodyna.utils.fft_utils import primal_to_fourier_2d, fourier_to_primal_2d
    with torch.no_grad():
        for path in snapshots:
            state = torch.load(path, map_location=device, weights_only=False)
            task.model.load_state_dict(state["model"])
            for key in ("gmm_sigmas", "gmm_amps"):
                getattr(task, key).copy_(state[key])
            if method == "opt" and task.use_pose_head:
                if "pose_head" not in state: raise ValueError(f"Missing pose_head at {path}")
                task.pose_head.load_state_dict(state["pose_head"])
            set_mode(task, False)
            z = np.load(path.parent/"z.npy")
            if len(z) != len(ids): raise ValueError("z rows and particle identities differ")
            pose = state.get("hps_pose") if mode == "hps" else None
            if pose is not None:
                valid = pose.get("valid")
                if valid is not None and not bool(valid.all()): raise ValueError(f"HPS cache incomplete at {path}")
                rotations = pose["rotations"].cpu().numpy()
                translations = pose["translations_yx"].cpu().numpy()
                if rotations.shape != (len(ids), 3, 3) or translations.shape != (len(ids), 2):
                    raise ValueError("HPS pose rows must match the saved latent-code particle order")
            else:
                if mode == "hps": raise ValueError(f"Missing HPS rotations at {path}")
                rotations = base.copy()
                if "rlnOriginXAngst" in actual:
                    translations = actual[["rlnOriginYAngst", "rlnOriginXAngst"]].to_numpy() / task.apix
                else:
                    translations = actual[["rlnOriginY", "rlnOriginX"]].to_numpy() * dataset.apix / task.apix
            table = state.get("pose_table")
            if table is not None:
                from .pose import PoseTable
                pt = PoseTable(len(ids)).to(device); pt.load_state_dict(table)
                rotations, translations = [v.cpu().numpy() for v in pt(torch.arange(len(ids), device=device))]
            values, losses = [], []
            corrected = []
            snapshot_epoch = int(path.parent.name.split("_")[0])
            search_epochs = max(2, search_images // len(ids) + 1) if search_images > 0 else 0
            for batch in loader(dataset, 128, 0):
                batch = move(batch, device); ix = batch["idx"].cpu().numpy()
                latent = torch.as_tensor(z[ix], device=device)
                pred = task.deformer.transform(task.model.eval_z(latent), task.gmm_centers)
                rot = torch.as_tensor(rotations[ix], device=device, dtype=torch.float32)
                if method == "opt" and task.use_pose_head and snapshot_epoch >= search_epochs:
                    rot = axis_angle_to_matrix(task.pose_head(latent)) @ rot
                corrected.append(rot.cpu().numpy())
                values.append(kabsch_rmsd(pred.cpu().numpy(), np.stack([structures[c] for c in classes[ix]])))
                image = task._apply_ctf(batch, task._shared_projection(pred, rot), task.lp_mask2d)
                shifts = torch.as_tensor(translations[ix], device=device, dtype=torch.float32)
                f = torch.fft.fftshift(torch.fft.fftfreq(image.shape[-1], device=device))
                phase = -2*torch.pi*(shifts[:, 0, None, None]*f[None, :, None]+shifts[:, 1, None, None]*f[None, None, :])
                observed = fourier_to_primal_2d(primal_to_fourier_2d(batch["proj"])*torch.exp(1j*phase[:, None])).real
                losses.append((len(ix), float(calc_cor_loss(image, task.low_pass_images(observed), task.mask))))
            rmsd = np.concatenate(values); corrected = np.concatenate(corrected)
            raw = angular_error(corrected, truth)
            aligned, gauge = fit_gauge(corrected, truth, calibration)
            aligned_errors = angular_error(aligned, truth)
            errors = aligned_errors if mode == "hps" else raw
            epoch_number, global_step = map(int, path.parent.name.split("_"))
            epoch = 0 if global_step == 0 else epoch_number+1
            row = dict(epoch=epoch, angle=angle, seed=manifest["seed"], particles=len(ids),
                       evaluation_particles=len(eval_ids), loss=sum(n*v for n,v in losses)/len(ids),
                       rmsd_A=float(rmsd[eval_ids].mean()), pose_deg=float(errors[eval_ids].mean()),
                       raw_pose_deg=float(raw[eval_ids].mean()), aligned_pose_deg=float(aligned_errors[eval_ids].mean()),
                       pose_p90_deg=float(np.quantile(errors[eval_ids], .9)))
            rows.append(row)
            np.savez_compressed(evaluation/f"epoch{epoch:03d}.npz", rotations=corrected, rmsd_A=rmsd,
                                raw_pose_deg=raw, aligned_pose_deg=aligned_errors, particle_ids=ids)
            print(json.dumps(row), flush=True)
    pd.DataFrame(rows).to_csv(output/"metrics.csv", index=False)
    write_json(output/"manifest.json", manifest)
    return output
