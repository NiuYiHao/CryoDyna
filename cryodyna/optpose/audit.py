"""Verify saved particle identities and recompute metrics from saved predictions."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from .geometry import angular_error, fit_gauge, kabsch_rmsd


def verify_initial_models(directories, atol=1e-6, rtol=1e-6):
    """Check structural tensors, including legacy unregistered attention layers.

    Repeated GPU scatter reductions can differ by float32 rounding. Retain
    exact fingerprints for provenance and audit actual tensors at this tolerance.
    Pose-head/table parameters are method-specific and outside this comparison.
    """
    def tensors(directory):
        state = torch.load(Path(directory)/"epoch000.pt", map_location="cpu", weights_only=False)
        result = {k: v for k, v in state["task"].items() if k.startswith("model.")}
        for name, layers in state["encoder_unregistered"].items():
            for i, layer in enumerate(layers):
                result.update({f"{name}.{i}.{k}": v for k, v in layer.items()})
        if not result:
            raise ValueError("Initial checkpoint requires structural tensors")
        return result
    directories = list(map(Path, directories))
    if not directories:
        raise ValueError("At least one initial checkpoint is required")
    reference = tensors(directories[0])
    rows = []
    for directory in directories:
        candidate = tensors(directory)
        if candidate.keys() != reference.keys():
            raise ValueError(f"Initial structural tensor names differ: {directory}")
        max_difference = 0.
        for key, target in reference.items():
            value = candidate[key]
            if value.shape != target.shape or value.dtype != target.dtype:
                raise ValueError(f"Initial tensor schema differs: {directory}: {key}")
            if target.is_floating_point():
                torch.testing.assert_close(value, target, atol=atol, rtol=rtol,
                                           msg=lambda msg: f"Initial structure mismatch: {directory}: {key}: {msg}")
                if value.numel():
                    max_difference = max(max_difference, float((value-target).abs().max()))
            elif not torch.equal(value, target):
                raise ValueError(f"Initial structure metadata differs: {directory}: {key}")
        rows.append(dict(run=str(directory), max_absolute_difference=max_difference))
    return dict(pass_initialization=True, reference=str(directories[0]), atol=atol, rtol=rtol,
                tensor_count=len(reference), comparisons=rows)


def verify_predictions(directory, manifest, replay_structures=True, device="cpu", endpoint_only=False):
    from .benchmark import (truth_data, resolve_config, make_dataset, make_task, load_task_state,
                            collect_runs, write_json, seed_all)
    directory = Path(directory)
    frame = pd.read_csv(directory/"metrics.csv")
    required = ["epoch", "loss", "rmsd_A", "pose_deg"]
    if frame.empty or not set(required).issubset(frame) or frame.epoch.duplicated().any():
        raise ValueError(f"Empty, incomplete or duplicate metric rows: {directory}")
    if not np.isfinite(frame[required].to_numpy()).all():
        raise ValueError(f"Nonfinite metric values: {directory}")
    if endpoint_only:
        frame = frame.sort_values("epoch").tail(1)
    truth, classes, structures = truth_data(manifest)
    eval_ids = np.asarray(manifest["evaluation_ids"])
    task = None
    # Historical imports have already decoded source checkpoints during import.
    if replay_structures and "source_outdir" not in manifest:
        seed_all(manifest["seed"])
        cfg = resolve_config(directory/"config.py")
        dataset = make_dataset(cfg, manifest["pose_init"])
        task = make_task(cfg, dataset, manifest["method"], device)
        initial = directory/"epoch000.pt"
        if not initial.exists(): raise FileNotFoundError(f"Initial full checkpoint required: {initial}")
        load_task_state(task, torch.load(initial, map_location=device, weights_only=False))
        task.eval()
    rows = []
    with torch.no_grad():
        for _, row in frame.iterrows():
            epoch = int(row.epoch)
            path = directory/"evaluation"/f"epoch{epoch:03d}.npz"
            if not path.exists(): raise FileNotFoundError(f"Missing per-particle predictions: {path}")
            with np.load(path, allow_pickle=False) as data:
                if not np.array_equal(data["particle_ids"], manifest["particle_ids"]):
                    raise ValueError(f"Particle order mismatch: {path}")
                raw = angular_error(data["rotations"], truth)
                np.testing.assert_allclose(raw, data["raw_pose_deg"], atol=1e-5)
                aligned, _ = fit_gauge(data["rotations"], truth, manifest["alignment_ids"])
                aligned_error = angular_error(aligned, truth)
                chosen = aligned_error if manifest["pose_init"] == "hps" else raw
                rmsd = data["rmsd_A"]
                if task is not None:
                    decoder = directory/"evaluation"/f"epoch{epoch:03d}.decoder.pt"
                    if epoch and decoder.exists():
                        task.model.decoder.load_state_dict(torch.load(decoder, map_location=device, weights_only=True))
                    elif epoch:
                        checkpoint = directory/f"epoch{epoch:03d}.pt"
                        if not checkpoint.exists(): raise FileNotFoundError(f"Missing decoder checkpoint for epoch {epoch}")
                        load_task_state(task, torch.load(checkpoint, map_location=device, weights_only=False))
                    z = data["z"]
                    replay = []
                    for start in range(0, len(z), 256):
                        latent = torch.as_tensor(z[start:start+256], device=device)
                        pred = task.deformer.transform(task.model.eval_z(latent), task.gmm_centers).cpu().numpy()
                        target = np.stack([structures[c] for c in classes[start:start+256]])
                        replay.append(kabsch_rmsd(pred, target))
                    np.testing.assert_allclose(np.concatenate(replay), rmsd, atol=2e-4, rtol=2e-4)
                np.testing.assert_allclose(float(rmsd[eval_ids].mean()), row.rmsd_A, atol=1e-6)
                np.testing.assert_allclose(float(chosen[eval_ids].mean()), row.pose_deg, atol=1e-6)
                rows.append(dict(epoch=epoch, particle_count=len(raw), metric_replay="passed"))
    report = {"source": str(directory), "structure_decoder_replay": task is not None,
              "scope": "endpoint" if endpoint_only else "all_saved_epochs", "epochs": rows}
    write_json(directory/("verification_endpoint.json" if endpoint_only else "verification.json"), report)
    return report


def acceptance(outdir, output, endpoint_only=False):
    from .benchmark import collect_runs, write_json
    runs = collect_runs(outdir)
    initial = verify_initial_models([directory for directory, _ in runs])
    rows = []
    contracts = []
    for directory, manifest in runs:
        verify_predictions(directory, manifest, device="cuda" if torch.cuda.is_available() else "cpu", endpoint_only=endpoint_only)
        frame = pd.read_csv(directory/"metrics.csv")
        last = frame.sort_values("epoch").iloc[-1]
        rows.append({"method": manifest["method"], "angle": manifest["angle"], "seed": manifest["seed"],
                     "rmsd_A": float(last.rmsd_A), "pose_deg": float(last.pose_deg), "epoch": int(last.epoch),
                     "particles": len(manifest["particle_ids"]), "status": manifest["status"]})
        contracts.append((manifest["particle_ids"], manifest["evaluation_ids"], manifest["truth_sha256"]))
    if not contracts or any(c != contracts[0] for c in contracts[1:]):
        raise ValueError("Acceptance requires identical particle identities and evaluation contract")
    frame = pd.DataFrame(rows)
    checks = []
    for (angle, seed), group in frame.groupby(["angle", "seed"]):
        if set(group.method) != {"original", "opt"} or len(group) != 2:
            checks.append({"angle": float(angle), "seed": int(seed), "pass": False, "reason": "Both methods required"})
            continue
        original = group[group.method == "original"].iloc[0]
        opt = group[group.method == "opt"].iloc[0]
        rmsd_pass = opt.rmsd_A < original.rmsd_A
        pose_pass = opt.pose_deg < original.pose_deg if angle > 0 else opt.pose_deg <= .05
        checks.append({"angle": float(angle), "seed": int(seed), "rmsd_pass": bool(rmsd_pass),
                       "pose_pass": bool(pose_pass), "pass": bool(rmsd_pass and pose_pass and opt.epoch == original.epoch),
                       "rmsd_difference_A": float(opt.rmsd_A-original.rmsd_A),
                       "pose_difference_deg": float(opt.pose_deg-original.pose_deg)})
    complete = set(frame.angle) >= {0, 5, 10, 15, 20} and set(frame.status) == {"complete"}
    full = bool((frame.particles == 50000).all())
    report = {"full_particle_count": full, "default_angles_complete": complete, "prediction_replay": True, "prediction_replay_scope": "endpoint" if endpoint_only else "all_saved_epochs",
              "initialization_verification": initial,
              "perturbation_pass": full and complete and all(c["pass"] for c in checks), "checks": checks, "results": rows}
    write_json(Path(output), report)
    return report
