"""Export a pose-free DRGN-AI benchmark configuration from the same input STAR."""
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
import starfile
import yaml
from .benchmark import read_star, resolve_config, selected_ids, write_json, sha256


def prepare_drgnai(config, outdir, count=50000, seed=1, sample_seed=101):
    cfg = resolve_config(config)
    source, frame = read_star(cfg.dataset_attr.starfile_path)
    ids = selected_ids(len(frame), count, sample_seed)
    frame = frame.iloc[ids].reset_index(drop=True)
    outdir = Path(outdir).resolve()
    if (outdir/"run/configs.yaml").exists(): raise FileExistsError("DRGN-AI output already configured")
    inputs = outdir/"inputs"; inputs.mkdir(parents=True, exist_ok=True)
    starfile.write(pd.DataFrame({"rlnImageName": frame.rlnImageName}), inputs/"particles.star", overwrite=True)
    np.save(inputs/"particle_ids.npy", ids)
    optics = source["optics"].set_index("rlnOpticsGroup").loc[frame.rlnOpticsGroup] if isinstance(source, dict) and "optics" in source else frame
    def column(name, fallback=None):
        if name in frame: return frame[name].to_numpy()
        if name in optics: return optics[name].to_numpy()
        if fallback is not None: return np.full(len(frame), fallback)
        raise ValueError(f"Missing CTF field {name}")
    ctf = np.column_stack([column("rlnImageSize", cfg.dataset_attr.side_shape),
        column("rlnImagePixelSize", cfg.dataset_attr.apix), column("rlnDefocusU"), column("rlnDefocusV"),
        column("rlnDefocusAngle"), column("rlnVoltage"), column("rlnSphericalAberration"),
        column("rlnAmplitudeContrast"), column("rlnPhaseShift", 0.)]).astype(np.float32)
    if not np.isfinite(ctf).all(): raise ValueError("Nonfinite CTF input")
    with (inputs/"ctf.pkl").open("wb") as stream: pickle.dump(ctf, stream)
    run = outdir/"run"; run.mkdir(exist_ok=True)
    settings = dict(particles=str(inputs/"particles.star"), ctf=str(inputs/"ctf.pkl"),
        datadir=cfg.dataset_attr.dataset_dir, pose=None, seed=seed,
        quick_config=dict(capture_setup="spa", reconstruction_type="het", pose_estimation="abinit", conf_estimation="autodecoder"),
        use_gt_poses=False, use_gt_trans=False, refine_gt_poses=False, pretrain_with_gt_poses=False,
        no_trans=False, no_trans_search_at_pose_search=False, n_imgs_pretrain=10000,
        n_imgs_pose_search=500000, epochs_sgd=20, lazy=True, max_threads=4, num_workers=2,
        batch_size_hps=1, batch_size_sgd=8, batch_size_known_poses=8, log_heavy_interval=1, log_interval=2000)
    (run/"configs.yaml").write_text(yaml.safe_dump(settings, sort_keys=False))
    write_json(outdir/"input_manifest.json", dict(truth_sha256=sha256(cfg.dataset_attr.starfile_path),
        particle_ids=ids.tolist(), image_names=frame.rlnImageName.tolist(), seed=seed,
        training_pose_access="none", ctf_sha256=sha256(inputs/"ctf.pkl")))
    return run
