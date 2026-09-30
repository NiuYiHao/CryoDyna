"""End-to-end benchmark. Training receives a stripped/perturbed STAR only.

Evaluation has separate access to ground truth and is never used by optimizers.
Run directories are self-describing; all paths are resolved from the config.
"""
import argparse
import hashlib
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import pickle
import random
import shutil
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import starfile
import torch
from scipy.spatial.transform import Rotation
from mmengine import Config

from .geometry import angular_error, fit_gauge, kabsch_rmsd, perturb_rotations, star_rotations

ROOT = Path(__file__).resolve().parents[2]
SCHEMA = 1


class AngleArrayAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        try:
            angles = [float(item) for value in values for item in value.strip("[]").split(",") if item.strip()]
        except ValueError:
            raise argparse.ArgumentError(self, "Use numbers or a quoted JSON-style array of angles")
        if not angles: raise argparse.ArgumentError(self, "At least one angle is required")
        setattr(namespace, self.dest, angles)


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.set_num_threads(4)


def read_star(path):
    data = starfile.read(path)
    return data, data["particles"] if isinstance(data, dict) else data


def resolve_config(path):
    cfg = Config.fromfile(str(Path(path).resolve()))
    for key in ("starfile_path", "dataset_dir", "ref_pdb_path"):
        p = Path(cfg.dataset_attr[key]).expanduser()
        cfg.dataset_attr[key] = str(p.resolve() if p.is_absolute() else (ROOT / p).resolve())
    return cfg


def selected_ids(count, requested, seed):
    if requested <= 0 or requested > count:
        raise ValueError(f"--particles must be in [1,{count}]")
    return np.sort(np.random.default_rng(seed).choice(count, requested, replace=False))


def prepare_inputs(cfg, folder, mode, angle, seed, ids):
    """Save just image/CTF metadata, plus experimental initial poses if given."""
    content, frame = read_star(cfg.dataset_attr.starfile_path)
    frame = frame.iloc[ids].copy().reset_index(drop=True)
    columns = [c for c in frame if c.startswith("rln") and any(
        c.startswith(prefix) for prefix in ("rlnImageName", "rlnOpticsGroup", "rlnDefocus", "rlnVoltage",
                                           "rlnSphericalAberration", "rlnAmplitudeContrast", "rlnPhaseShift",
                                           "rlnImagePixelSize", "rlnImageSize", "rlnMagnification", "rlnDetectorPixelSize"))]
    clean = frame[columns].copy()
    if mode == "given":
        rotations = perturb_rotations(star_rotations(frame), angle, seed)
        clean[["rlnAngleRot", "rlnAngleTilt", "rlnAnglePsi"]] = -Rotation.from_matrix(rotations).as_euler("ZYZ", degrees=True)[:, ::-1]
        for axis in "XY":
            key = f"rlnOrigin{axis}Angst"
            clean[key] = frame[key].to_numpy() if key in frame else frame[f"rlnOrigin{axis}"].to_numpy() * cfg.dataset_attr.apix
    new = {"optics": content["optics"], "particles": clean} if isinstance(content, dict) and "optics" in content else clean
    folder.mkdir(parents=True, exist_ok=True)
    starfile.write(new, folder / "particles.star", overwrite=True)
    np.save(folder / "particle_ids.npy", ids)
    # Fail early instead of accepting zero-filled missing images from legacy dataio.
    import mrcfile
    for name, group in frame.assign(stack=frame.rlnImageName.str.split("@").str[-1]).groupby("stack"):
        path = Path(cfg.dataset_attr.dataset_dir) / name
        with mrcfile.mmap(path, permissive=True) as mrc:
            required = group.rlnImageName.str.split("@").str[0].astype(int).max()
            available = mrc.data.shape[0] if mrc.data.ndim == 3 else 1
            if required > available:
                raise ValueError(f"Particle index exceeds stack size: {path}")
    return frame.rlnImageName.tolist()


def make_dataset(cfg, mode):
    from cryodyna.utils.dataio import StarfileDataSet, StarfileDatasetConfig
    dataset = StarfileDataSet(StarfileDatasetConfig(
        dataset_dir=cfg.dataset_attr.dataset_dir, starfile_path=cfg.dataset_attr.starfile_path,
        apix=cfg.dataset_attr.apix, side_shape=cfg.dataset_attr.side_shape,
        down_side_shape=cfg.data_process.down_side_shape, mask_rad=cfg.data_process.mask_rad,
        ignore_rots=mode == "hps", ignore_trans=mode == "hps"))
    cfg.data_process.down_side_shape = dataset.down_side_shape
    cfg.data_process.down_apix = dataset.apix * dataset.side_shape / dataset.down_side_shape
    return dataset


class BenchMixin:
    @property
    def current_epoch(self):
        return self.benchmark_epoch

    def all_gather(self, data, *args, **kwargs):
        return data


def make_task(cfg, dataset, method, device):
    # Console entry points from the active environment include mkdssp.
    env_bin = str(Path(sys.executable).resolve().parent)
    os.environ["PATH"] = env_bin + os.pathsep + os.environ.get("PATH", "")
    if not (shutil.which("mkdssp") or shutil.which("dssp")):
        raise RuntimeError("Install DSSP in the active environment (conda install -c salilab dssp)")
    sys.path.insert(0, str(ROOT / "projects")) if str(ROOT / "projects") not in sys.path else None
    module = importlib.import_module("train_atom" if method == "original" else "train_atom_with_pose")
    cls = type("BenchmarkTask", (BenchMixin, module.CryoEMTask), {})
    task = cls(cfg, dataset)
    task.benchmark_epoch = 0
    task.to(device)
    task.grid = task.grid.to(device)
    # Legacy lists remain outside the optimizer, matching the original method.
    for name in ("attn_layers", "post_norm"):
        for layer in getattr(task.model.encoder, name, []):
            layer.to(device)
    return task


def set_mode(task, training):
    task.train(training)
    for name in ("attn_layers", "post_norm"):
        for layer in getattr(task.model.encoder, name, []):
            layer.train(training)


def move(batch, device):
    return {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}


def loader(dataset, batch_size, seed, shuffle=False, workers=0):
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
        generator=torch.Generator().manual_seed(seed), num_workers=workers, drop_last=False)


def loss_terms(task, gt, pred_image, structure):
    from cryodyna.utils.losses import calc_cor_loss
    from train_atom_with_pose import CryoEMTask
    image = task.cfg.loss.gmm_cryoem_weight * calc_cor_loss(pred_image, task.low_pass_images(gt), task.mask)
    structural = sum(getattr(CryoEMTask, "_calculate_"+key+"_loss")(task, structure)
                     for key in ("connect", "sse", "dist", "clash"))
    return image, structural


def infer(task, batch, method):
    deformation, z = task._shared_forward(batch["proj"])
    structure = task.deformer.transform(deformation, task.gmm_centers)
    if method == "original":
        rotations, shifts = task.get_batch_pose(batch)
    else:
        rotations, shifts = task.predict_pose(batch, structure, z)
    image = task._apply_ctf(batch, task._shared_projection(structure, rotations), task.lp_mask2d)
    from cryodyna.utils.fft_utils import primal_to_fourier_2d, fourier_to_primal_2d
    # Same differentiable shift operator for both methods; input shifts are pixels.
    n = batch["proj"].shape[-1]
    f = torch.fft.fftshift(torch.fft.fftfreq(n, device=task.device))
    phase = -2 * torch.pi * (shifts[:, 0, None, None] * f[None, :, None] + shifts[:, 1, None, None] * f[None, None, :])
    observed = fourier_to_primal_2d(primal_to_fourier_2d(batch["proj"]) * torch.exp(1j * phase[:, None])).real
    return observed, image, structure, z, rotations, shifts


def task_state(task):
    return {"task": {k: v.detach().cpu() for k, v in task.state_dict().items() if not k.startswith("grid.")},
            "encoder_unregistered": {name: [{k: v.detach().cpu() for k, v in layer.state_dict().items()}
              for layer in getattr(task.model.encoder, name, [])] for name in ("attn_layers", "post_norm")}}


def load_task_state(task, state):
    result = task.load_state_dict(state["task"], strict=False)
    missing = [k for k in result.missing_keys if not k.startswith("grid.")]
    if missing or result.unexpected_keys:
        raise ValueError(f"Checkpoint incompatibility: missing={missing}, unexpected={result.unexpected_keys}")
    for name, values in state["encoder_unregistered"].items():
        for layer, value in zip(getattr(task.model.encoder, name, []), values):
            layer.load_state_dict(value)


def save_checkpoint(path, state):
    tmp = Path(str(path) + ".tmp")
    torch.save(state, tmp)
    tmp.replace(path)


def environment_record():
    result = {"python": sys.version, "torch": torch.__version__, "cuda": torch.version.cuda,
              "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
    git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, capture_output=True)
    result["git_head"] = git.stdout.strip() if git.returncode == 0 else "source-snapshot"
    result["packages"] = {name: importlib.metadata.version(name) for name in
                          ("numpy", "scipy", "torch", "lightning", "mmengine", "starfile", "mrcfile", "torch-geometric", "healpy")}
    files = list((ROOT / "cryodyna").rglob("*.py")) + list((ROOT / "projects").glob("*.py"))
    result["source_sha256"] = {str(p.relative_to(ROOT)): sha256(p) for p in sorted(files)}
    return result


def initial_model_fingerprint(task):
    h = hashlib.sha256()
    tensors = dict(task.model.state_dict())
    for name in ("attn_layers", "post_norm"):
        for i, layer in enumerate(getattr(task.model.encoder, name, [])):
            tensors.update({f"{name}.{i}.{key}": value for key, value in layer.state_dict().items()})
    for key, value in sorted(tensors.items()):
        h.update(key.encode()); h.update(value.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def pretrain(task, dataset, args):
    """Reference-only zero-deformation warmup, with a fixed step budget."""
    seed_all(args.seed + 41)
    set_mode(task, True)
    optimizer = torch.optim.AdamW(task.model.parameters(), lr=task.cfg.optimizer.lr)
    step = 0
    while step < args.init_steps:
        for batch in loader(dataset, args.batch_size, args.seed + 101 + step, True, args.workers):
            batch = move(batch, task.device)
            optimizer.zero_grad()
            deformation, _ = task._shared_forward(batch["proj"])
            loss = deformation.square().mean()
            loss.backward()
            optimizer.step()
            step += 1
            if step >= args.init_steps:
                break


def truth_data(manifest):
    from cryodyna.utils.polymer import Polymer
    path = manifest["truth_star"]
    if sha256(path) != manifest["truth_sha256"]:
        raise ValueError("Ground-truth STAR changed since run creation")
    _, all_frame = read_star(path)
    ids = np.asarray(manifest["particle_ids"])
    frame = all_frame.iloc[ids]
    if frame.rlnImageName.tolist() != manifest["image_names"]:
        raise ValueError("Particle identity/order mismatch")
    rotations = star_rotations(frame)
    classes = frame.rlnClassNumber.to_numpy(int)
    reference = Polymer.from_pdb(manifest["ref_pdb"])
    coordinates = {}
    atom_keys = list(zip(reference.chain_id, reference.res_id, reference.atom_name))
    for i in np.unique(classes):
        pdb = Polymer.from_pdb(str(Path(manifest["pdb_dir"]) / f"1akeA_{i}.pdb"))
        if list(zip(pdb.chain_id, pdb.res_id, pdb.atom_name)) != atom_keys or not np.all(pdb.atom_name == "CA"):
            raise ValueError("C-alpha atom identities/order differ from the reference")
        coordinates[i] = pdb.coord
    return rotations, classes, coordinates


@torch.no_grad()
def evaluate_task(task, dataset, args, manifest, epoch, output):
    # Save/restore RNG so evaluation frequency cannot alter training randomness.
    rng = (random.getstate(), np.random.get_state(), torch.get_rng_state(), torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])
    seed_all(args.seed + 5000)
    set_mode(task, False)
    truth, classes, structures = truth_data(manifest)
    rotations, shifts, latents, rmsd, image_losses = [], [], [], [], []
    for batch in loader(dataset, args.eval_batch_size, 0, workers=args.workers):
        batch = move(batch, task.device)
        gt, image, pred, z, r, t = infer(task, batch, manifest["method"])
        from cryodyna.utils.losses import calc_cor_loss
        image_loss = calc_cor_loss(image, task.low_pass_images(gt), task.mask)
        image_losses.append((len(z), float(image_loss)))
        ix = batch["idx"].cpu().numpy()
        target = np.stack([structures[c] for c in classes[ix]])
        rmsd.append(kabsch_rmsd(pred.cpu().numpy(), target))
        rotations.append(r.cpu().numpy()); shifts.append(t.cpu().numpy()); latents.append(z.cpu().numpy())
    rotations, shifts, latents, rmsd = map(np.concatenate, (rotations, shifts, latents, rmsd))
    raw = angular_error(rotations, truth)
    fit_ids = np.asarray(manifest["alignment_ids"])
    aligned, gauge = fit_gauge(rotations, truth, fit_ids)
    aligned_error = angular_error(aligned, truth)
    eval_ids = np.asarray(manifest["evaluation_ids"])
    chosen = aligned_error if manifest["pose_init"] == "hps" else raw
    values = {"epoch": epoch, "angle": manifest["angle"], "seed": args.seed,
              "particles": len(rmsd), "evaluation_particles": len(eval_ids),
              "loss": sum(n*v for n, v in image_losses)/len(rmsd),
              "rmsd_A": float(rmsd[eval_ids].mean()), "pose_deg": float(chosen[eval_ids].mean()),
              "pose_p90_deg": float(np.quantile(chosen[eval_ids], .9)),
              "raw_pose_deg": float(raw[eval_ids].mean()),
              "aligned_pose_deg": float(aligned_error[eval_ids].mean())}
    if not all(np.isfinite(v) for v in values.values() if isinstance(v, (float, int))):
        raise RuntimeError("Nonfinite evaluation metric")
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / f"epoch{epoch:03d}.npz", rotations=rotations, translations_yx=shifts,
                        z=latents, rmsd_A=rmsd, raw_pose_deg=raw, aligned_pose_deg=aligned_error,
                        particle_ids=np.asarray(manifest["particle_ids"]))
    write_json(output / f"epoch{epoch:03d}.json", {**values, "gauge": gauge})
    random.setstate(rng[0]); np.random.set_state(rng[1]); torch.set_rng_state(rng[2])
    if rng[3]: torch.cuda.set_rng_state_all(rng[3])
    return values


def run_one(args, method, angle, directory, mode="given"):
    seed_all(args.seed)
    directory = Path(directory).resolve()
    manifest_path = directory / "manifest.json"
    if manifest_path.exists() and not args.resume:
        raise FileExistsError(f"Run already exists: {directory}; use --resume or a fresh directory")
    directory.mkdir(parents=True, exist_ok=True)
    cfg = resolve_config(args.config)
    if cfg.extra_input_data_attr.get("ckpt_path"):
        raise ValueError("Benchmark initialization must be shared; clear extra_input_data_attr.ckpt_path")
    truth_path = cfg.dataset_attr.starfile_path
    _, frame = read_star(truth_path)
    ids = selected_ids(len(frame), args.particles, args.sample_seed)
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text())
        for key, value in (("method", method), ("angle", angle), ("pose_init", mode), ("seed", args.seed), ("particle_ids", ids.tolist())):
            if old[key] != value: raise ValueError(f"Resume contract mismatch: {key}")
        old_args = {k: v for k, v in old["arguments"].items() if k != "resume"}
        new_args = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items() if k != "resume"}
        if old_args != new_args: raise ValueError("Resume requires identical arguments (except --resume)")
        if old["environment"]["source_sha256"] != environment_record()["source_sha256"]:
            raise ValueError("Source changed; choose a fresh output directory")
        if sha256(truth_path) != old["truth_sha256"]:
            raise ValueError("Truth/input STAR changed since this run")
        if old.get("training_star_sha256") != sha256(directory/"inputs/particles.star"):
            raise ValueError("Saved training input changed since this run")
        if old["status"] == "complete": return directory
        if not (directory/"resume.pt").exists():
            raise FileNotFoundError("Incomplete run has no resume.pt; choose a fresh output directory")
        names = old["image_names"]
    else:
        names = prepare_inputs(cfg, directory / "inputs", mode, angle, args.seed, ids)
    cfg.dataset_attr.starfile_path = str(directory / "inputs/particles.star")
    cfg.pose_init = mode
    cfg.n_imgs_pose_search = 0
    cfg.use_pose_table = method == "opt" and args.pose_model in ("table", "hybrid")
    cfg.model.use_pose_head = method == "opt" and args.pose_model in ("head", "hybrid")
    cfg.model.pose_head_type = args.head_type
    cfg.loss.pose_reg_weight = args.pose_reg
    cfg.optimizer.pose_lr = args.pose_lr
    cfg.optimize_translations = not args.rotation_only
    cfg.hps_score = args.hps_score
    cfg.t_extent = 0. if args.rotation_only else args.translation_extent
    cfg.t_n_grid = 1 if args.rotation_only else 3
    cfg.hps_particle_chunk_size = 4
    cfg.n_kept_poses = args.hps_kept
    cfg.l_start, cfg.l_end = args.hps_l_start, args.hps_l_end
    if mode == "hps": cfg.n_imgs_pose_search = 1  # override schedule explicitly below
    dataset = make_dataset(cfg, mode)
    if manifest_path.exists():
        if old.get("config_sha256") != sha256(directory/"config.py"):
            raise ValueError("Saved configuration changed since this run")
    else:
        cfg.dump(str(directory / "config.py"))
    task = make_task(cfg, dataset, method, args.device)
    if mode == "hps": task.epochs_pose_search = args.hps_epochs
    fit = np.random.default_rng(args.sample_seed + 1).permutation(len(ids))[:max(1, len(ids)//5)]
    evaluate = np.setdiff1d(np.arange(len(ids)), fit)
    manifest = {"schema": SCHEMA, "kind": "run", "method": method, "pose_init": mode, "angle": angle,
        "seed": args.seed, "particle_ids": ids.tolist(), "image_names": names,
        "alignment_ids": fit.tolist(), "evaluation_ids": evaluate.tolist(),
        "truth_star": truth_path, "truth_sha256": sha256(truth_path), "pdb_dir": str(Path(args.pdb_dir).resolve()),
        "training_star_sha256": sha256(directory/"inputs/particles.star"), "config_sha256": sha256(directory/"config.py"),
        "ref_pdb": cfg.dataset_attr.ref_pdb_path, "config": "config.py", "epochs": args.epochs,
        "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "environment": environment_record(), "status": "running", "metric": "mean full SO(3) geodesic; mean per-particle Kabsch C-alpha RMSD"}
    optimizer = task.configure_optimizers()
    first_epoch = 0
    curves = []
    if args.resume and (directory / "resume.pt").exists():
        old = json.loads(manifest_path.read_text())
        manifest["initial_model_sha256"] = old.get("initial_model_sha256")
        for key in ("method", "pose_init", "angle", "seed", "particle_ids", "truth_sha256", "epochs"):
            if old[key] != manifest[key]: raise ValueError(f"Resume contract mismatch: {key}")
        if old["environment"]["source_sha256"] != manifest["environment"]["source_sha256"]:
            raise ValueError("Source changed; start a fresh run to preserve provenance")
        state = torch.load(directory / "resume.pt", map_location=args.device, weights_only=False)
        load_task_state(task, state)
        optimizer.load_state_dict(state["optimizer"])
        first_epoch, curves = state["epoch"], state["curves"]
        torch.set_rng_state(state["torch_rng"].cpu())
        np.random.set_state(state["numpy_rng"]); random.setstate(state["python_rng"])
        if torch.cuda.is_available(): torch.cuda.set_rng_state_all([v.cpu() for v in state["cuda_rng"]])
    else:
        write_json(manifest_path, manifest)
        pretrain(task, dataset, args)
        manifest["initial_model_sha256"] = initial_model_fingerprint(task)
        write_json(manifest_path, manifest)
        # Reset sampling and dropout RNG after the common reference warmup.
        seed_all(args.seed + 1000)
        initial = evaluate_task(task, dataset, args, manifest, 0, directory / "evaluation")
        initial["train_loss"] = None
        curves.append(initial)
        save_checkpoint(directory / "epoch000.pt", task_state(task))
        pd.DataFrame(curves).to_csv(directory / "metrics.csv", index=False)
    try:
        for epoch in range(first_epoch, args.epochs):
            started = time.time()
            task.benchmark_epoch = epoch
            set_mode(task, True)
            total, count = 0., 0
            for batch in loader(dataset, args.batch_size, args.sample_seed + epoch, True, args.workers):
                batch = move(batch, args.device)
                optimizer.zero_grad()
                gt, pred_image, structure, z, _, _ = infer(task, batch, method)
                image_loss, regularizer = loss_terms(task, gt, pred_image, structure)
                loss = image_loss + regularizer
                if method == "opt" and task.use_pose_head and not task.is_in_pose_search_step:
                    loss = loss + args.pose_reg * task.pose_head(z).square().sum(-1).mean()
                if method == "opt" and task.use_pose_table and not task.is_in_pose_search_step:
                    loss = loss + task.pose_table_regularization(batch)
                if not torch.isfinite(loss): raise RuntimeError("Nonfinite training loss")
                loss.backward()
                optimizer.step()
                total += float(loss.detach()) * len(z); count += len(z)
            result = evaluate_task(task, dataset, args, manifest, epoch+1, directory / "evaluation")
            result.update(train_loss=total/count, seconds=time.time()-started)
            curves.append(result)
            state = task_state(task)
            if (epoch+1) % args.checkpoint_every == 0 or epoch+1 == args.epochs:
                save_checkpoint(directory / f"epoch{epoch+1:03d}.pt", state)
            # Every epoch retains decoder + saved z/poses for exact metric replay.
            save_checkpoint(directory / "evaluation" / f"epoch{epoch+1:03d}.decoder.pt",
                            {k: v.detach().cpu() for k, v in task.model.decoder.state_dict().items()})
            save_checkpoint(directory / "resume.pt", {**state, "optimizer": optimizer.state_dict(),
                "epoch": epoch+1, "curves": curves, "torch_rng": torch.get_rng_state(),
                "numpy_rng": np.random.get_state(), "python_rng": random.getstate(),
                "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []})
            pd.DataFrame(curves).to_csv(directory / "metrics.csv", index=False)
            print(json.dumps({"run": directory.name, **result}), flush=True)
            if shutil.disk_usage(directory).free < 1.5 * 1024**3:
                raise RuntimeError("Free space below 1.5 GiB; resume after choosing sufficient storage")
        manifest["status"] = "complete"
        write_json(manifest_path, manifest)
    except BaseException as error:
        manifest.update(status="interrupted", error=str(error))
        write_json(manifest_path, manifest)
        raise
    return directory


def run_matrix(args, mode="given"):
    out = Path(args.outdir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    methods = ["opt"] if mode == "hps" else args.methods
    angles = [0.] if mode == "hps" else args.angles
    jobs = [{"method": m, "angle": a, "path": f"{m}/{a:g}deg_seed{args.seed}"} for m in methods for a in angles]
    if (out / "manifest.json").exists() and not args.resume:
        raise FileExistsError("Output matrix exists; choose a new path or --resume")
    matrix = {"schema": SCHEMA, "kind": "matrix", "pose_init": mode, "jobs": jobs}
    if (out / "manifest.json").exists():
        if json.loads((out / "manifest.json").read_text()) != matrix:
            raise ValueError("Resume requires the same matrix methods, angles and seed")
    else:
        write_json(out / "manifest.json", matrix)
    for method in methods:
        write_json(out / method / "manifest.json", {"schema": SCHEMA, "kind": "matrix", "pose_init": mode,
            "jobs": [{**j, "path": Path(j["path"]).name} for j in jobs if j["method"] == method]})
    for job in jobs:
        run_one(args, job["method"], job["angle"], out / job["path"], mode)
    initial_hashes = {json.loads((out/j["path"]/"manifest.json").read_text()).get("initial_model_sha256") for j in jobs}
    if len(initial_hashes) != 1:
        raise ValueError("Methods/angles started from different structural models; inspect initial_model_sha256")
    from .plotting import plot_learning
    plot_learning(out, out / "figures")
    if mode == "hps": compare_abinit(out / jobs[0]["path"], args.drgnai_outdir, out / "figures", args.seed)


def collect_runs(directory):
    directory = Path(directory).resolve()
    path = directory / "manifest.json"
    if path.exists():
        manifest = json.loads(path.read_text())
        if manifest.get("kind") == "matrix":
            if not manifest.get("jobs"): raise ValueError(f"Empty run matrix: {directory}")
            result = []
            for job in manifest["jobs"]:
                result.extend(collect_runs(directory / job["path"]))
            return result
        if manifest.get("kind") == "run":
            if not (directory/"metrics.csv").exists(): raise FileNotFoundError(f"Metrics pending: {directory}")
            return [(directory, manifest)]
    raise ValueError(f"Expected a benchmark run/matrix: {directory}. Use import-legacy for historical Output dirs.")


def compare_abinit(opt_dir, drgnai_dir, output, seed):
    """Evaluate saved DRGN-AI rotations with the identical identities and metric."""
    opt_dir, output = Path(opt_dir), Path(output)
    manifest = json.loads((opt_dir / "manifest.json").read_text())
    truth, _, _ = truth_data(manifest)
    ids = np.asarray(manifest["particle_ids"])
    fit, evaluate = np.asarray(manifest["alignment_ids"]), np.asarray(manifest["evaluation_ids"])
    rows = []
    def append(pred, name, epoch):
        aligned, _ = fit_gauge(pred, truth, fit)
        errors = angular_error(aligned, truth)[evaluate]
        rows.append(dict(method=name, epoch=epoch, pose_deg=float(errors.mean()), pose_p90_deg=float(np.quantile(errors, .9))))
    if manifest["pose_init"] != "hps":
        raise ValueError("Ab-initio comparison requires a run without supplied poses")
    for path in sorted((opt_dir / "evaluation").glob("epoch*.npz")):
        with np.load(path, allow_pickle=False) as data:
            if not np.array_equal(data["particle_ids"], ids):
                raise ValueError(f"Particle identity mismatch: {path}")
            append(data["rotations"], "CryoDyna-optpose", int(path.stem[5:]))
    append(Rotation.random(len(ids), random_state=seed).as_matrix(), "Random SO(3)", 0)
    if drgnai_dir:
        source = Path(drgnai_dir).resolve()
        # Inputs must be accompanied by particle-order evidence.
        possible = [source / "particle_ids.npy", source.parent.parent / "inputs/particle_ids.npy"]
        mapping = next((p for p in possible if p.exists()), None)
        if mapping:
            source_ids = np.load(mapping)
            if len(np.unique(source_ids)) != len(source_ids):
                raise ValueError("Duplicate DRGN-AI particle IDs")
            lookup = {int(v): i for i, v in enumerate(source_ids)}
            take = [lookup[int(v)] for v in ids]
        else:
            candidate = source.parent.parent / "inputs/particles.star"
            if not candidate.exists(): raise ValueError("DRGN-AI needs particle_ids.npy or adjacent inputs/particles.star")
            _, source_frame = read_star(candidate)
            lookup = {v: i for i, v in enumerate(source_frame.rlnImageName)}
            if len(lookup) != len(source_frame): raise ValueError("Duplicate DRGN-AI particle names")
            take = [lookup[v] for v in manifest["image_names"]]
        paths = sorted(source.glob("pose.*.pkl"), key=lambda p: int(p.stem.split(".")[-1]))
        if not paths: raise ValueError("No DRGN-AI pose checkpoints found")
        for path in paths:
            epoch = int(path.stem.split(".")[-1])
            if epoch < 0: continue
            with path.open("rb") as stream: pred, _ = pickle.load(stream)
            append(np.asarray(pred)[take], "DRGN-AI", epoch+1)
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output / "abinit_metrics.csv", index=False)
    from .plotting import plot_abinit
    plot_abinit(pd.DataFrame(rows), output)
    final = max((r for r in rows if r["method"] == "CryoDyna-optpose"), key=lambda r: r["epoch"])
    controls = {name: max((r for r in rows if r["method"] == name), key=lambda r: r["epoch"])
                for name in ("Random SO(3)", "DRGN-AI") if any(r["method"] == name for r in rows)}
    write_json(output / "abinit_acceptance.json", {"mean_limit_deg": 5, "p90_limit_deg": 10,
        "status": manifest["status"], "prediction_replay": True,
        "release_pass": manifest["status"] == "complete" and len(ids) == 50000 and len(controls) == 2
            and final["pose_deg"] <= 5 and final["pose_p90_deg"] <= 10
            and all(final["pose_deg"] < c["pose_deg"] for c in controls.values()),
        "accuracy_pass": final["pose_deg"] <= 5 and final["pose_p90_deg"] <= 10,
        "controls_complete": len(controls) == 2,
        "better_than_controls": len(controls) == 2 and all(final["pose_deg"] < c["pose_deg"] for c in controls.values()),
        "scope": "full" if len(ids) == 50000 else "pilot", "final": final, "controls": controls})


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("diagnose")
    p.add_argument("--config", default=str(ROOT / "projects/atom_configs/1ake_optpose.py"))
    p.add_argument("--output", required=True)
    p.add_argument("--particles", type=int, default=16)
    p.add_argument("--device", default="cuda")
    p.add_argument("--hps-l-start", type=int, default=12)
    p.add_argument("--hps-l-end", type=int, default=32)
    p.add_argument("--hps-kept", type=int, default=8)
    p.add_argument("--atomic-template", action="store_true")
    p = commands.add_parser("prepare-drgnai")
    p.add_argument("--config", default=str(ROOT/"projects/atom_configs/1ake_optpose.py"))
    p.add_argument("--outdir", required=True)
    p.add_argument("--particles", type=int, default=50000)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--sample-seed", type=int, default=101)
    for name in ("run", "abinit"):
        p = commands.add_parser(name)
        p.add_argument("--config", default=str(ROOT / "projects/atom_configs/1ake_optpose.py"))
        p.add_argument("--pdb-dir", default=str(ROOT / "tutorial_data_1ake/pdbs"))
        p.add_argument("--outdir", required=True)
        p.add_argument("--methods", nargs="+", choices=["original", "opt"], default=["original", "opt"])
        p.add_argument("--angles", action=AngleArrayAction, nargs="+", default=[0, 5, 10, 15, 20])
        p.add_argument("--seed", type=int, default=1)
        p.add_argument("--sample-seed", type=int, default=101)
        p.add_argument("--particles", type=int, default=50000)
        p.add_argument("--epochs", type=int, default=20)
        p.add_argument("--init-steps", type=int, default=128)
        p.add_argument("--checkpoint-every", type=int, default=5)
        p.add_argument("--batch-size", type=int, default=64)
        p.add_argument("--eval-batch-size", type=int, default=128)
        p.add_argument("--workers", type=int, default=0)
        p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
        p.add_argument("--pose-model", choices=["table", "head", "hybrid"], default="hybrid" if name == "abinit" else "head")
        p.add_argument("--head-type", choices=["mlp", "small_mlp"], default="mlp")
        p.add_argument("--pose-lr", type=float, default=.003)
        p.add_argument("--pose-reg", type=float, default=.5)
        p.add_argument("--rotation-only", action="store_true")
        p.add_argument("--translation-extent", type=float, default=4.)
        p.add_argument("--hps-epochs", type=int, default=1)
        p.add_argument("--hps-score", choices=["l2", "correlation"], default="correlation")
        p.add_argument("--hps-kept", type=int, default=8)
        p.add_argument("--hps-l-start", type=int, default=12)
        p.add_argument("--hps-l-end", type=int, default=32)
        p.add_argument("--drgnai-outdir")
        p.add_argument("--resume", action="store_true")
    p = commands.add_parser("evaluate")
    p.add_argument("--outdir", required=True)
    p.add_argument("--output")
    p.add_argument("--method", choices=["original", "opt"])
    p.add_argument("--snapshot-epochs", type=int, nargs="+", help="Optional historical checkpoint epoch selection")
    p.add_argument("--truth-star", default=str(ROOT/"tutorial_data_1ake/uniform_snr0-0001_ctf/simulation.star"))
    p.add_argument("--pdb-dir", default=str(ROOT/"tutorial_data_1ake/pdbs"))
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p = commands.add_parser("compare")
    p.add_argument("--runs", nargs="+", required=True, help="Repeated LABEL=OUTDIR values")
    p.add_argument("--output", required=True)
    p = commands.add_parser("compare-abinit")
    p.add_argument("--outdir", required=True)
    p.add_argument("--drgnai-outdir")
    p.add_argument("--output", required=True)
    p.add_argument("--seed", type=int, default=1)
    p = commands.add_parser("acceptance")
    p.add_argument("--endpoint-only", action="store_true", help="Replay final saved predictions; record endpoint-only verification scope")
    p.add_argument("--outdir", required=True)
    p.add_argument("--output", required=True)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.command == "diagnose":
        from .diagnostics import diagnose
        diagnose(args.config, args.output, args.particles, args.device, args.hps_l_start, args.hps_l_end, args.hps_kept, args.atomic_template)
    elif args.command == "prepare-drgnai":
        from .drgnai import prepare_drgnai
        print(prepare_drgnai(args.config, args.outdir, args.particles, args.seed, args.sample_seed))
    elif args.command in ("run", "abinit"):
        if len(set(args.angles)) != len(args.angles) or any(not np.isfinite(a) or a < 0 or a > 180 for a in args.angles):
            raise ValueError("Use distinct finite angles in [0,180]")
        if min(args.epochs, args.particles, args.batch_size, args.eval_batch_size, args.checkpoint_every) < 1 or args.particles < 2:
            raise ValueError("Positive budgets and at least two particles are required")
        run_matrix(args, "hps" if args.command == "abinit" else "given")
    elif args.command == "evaluate":
        from .plotting import plot_learning
        output = Path(args.output) if args.output else Path(args.outdir)/"figures"
        if not (Path(args.outdir)/"manifest.json").exists():
            from .legacy import import_legacy
            directory = import_legacy(args.outdir, output/"imported", args.method, args.truth_star, args.pdb_dir, args.device, args.snapshot_epochs)
        else:
            directory = Path(args.outdir)
            from .audit import verify_predictions
            for run, manifest in collect_runs(directory):
                verify_predictions(run, manifest, device=args.device)
        plot_learning(directory, output)
    elif args.command == "compare":
        from .plotting import plot_comparison
        plot_comparison(args.runs, Path(args.output))
    elif args.command == "compare-abinit":
        compare_abinit(args.outdir, args.drgnai_outdir, args.output, args.seed)
    elif args.command == "acceptance":
        from .audit import acceptance
        print(json.dumps(acceptance(args.outdir, args.output, endpoint_only=args.endpoint_only), indent=2))
