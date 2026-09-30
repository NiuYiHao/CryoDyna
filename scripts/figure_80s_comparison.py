"""Build the requested figure from completed, particle-matched 80S runs.

DRGN-AI inputs are read from the existing reproduction task. Optpose inputs use
the benchmark's epoch*.npz format: rotations and particle_ids. Training truth is
used exclusively here for evaluation. Missing observations produce status.json.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys

os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_evaluator(path):
    spec = importlib.util.spec_from_file_location("existing_80s_evaluator", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.validate()
    return mod


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--drgnai-root", type=Path, required=True)
    parser.add_argument("--optpose-root", type=Path, required=True,
                        help="seedN/evaluation/epochNNN.npz (epoch 0 is initialization)")
    parser.add_argument("--evaluator", type=Path, default=Path(__file__).resolve().parents[1]/"cryodyna/optpose/volume_metrics.py")
    parser.add_argument("--renderer", type=Path, default=Path(__file__).with_name("plot_curve_endpoint.py"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(6)))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--accuracy-limit", type=float, default=5.)
    parser.add_argument("--volume-control", action="store_true",
                        help="Read one deterministic fixed-volume control directly from optpose-root")
    parser.add_argument("--progress", action="store_true",
                        help="Render clearly labelled interim observations after each method has a checkpoint")
    args = parser.parse_args()
    if len(set(args.seeds)) != len(args.seeds) or args.epochs < 1:
        raise ValueError("Unique seeds and a positive epoch count are required")
    args.output.mkdir(parents=True, exist_ok=True)
    status = {"dataset": "synthetic 80S", "required_seeds": args.seeds,
              "required_epochs": args.epochs, "missing": [],
              "figure": "pending measured results", "paper_fidelity": "see reproduction protocol differences"}
    jobs = []
    opt_label = "Optpose (density prior)" if args.volume_control else "CryoDyna-optpose"
    for seed in args.seeds:
        for epoch in range(1, args.epochs + 1):
            dr = args.drgnai_root / f"seed{seed}/out"
            dp = dr / f"pose.{epoch-1}.pkl"
            entries = [("DRGN-AI", dp, dr / f"weights.{epoch-1}.pkl")]
            op = args.optpose_root / f"seed{seed}/evaluation/epoch{epoch:03d}.npz"
            if args.volume_control:
                op = args.optpose_root / f"evaluation/epoch{epoch:03d}.npz"
            if not args.volume_control or seed == args.seeds[0]:
                entries.append((opt_label, op, op))
            for method, path, marker in entries:
                if path.is_file() and marker.is_file():
                    jobs.append((method, seed, epoch, path))
                else:
                    status["missing"].append(dict(method=method, seed=seed, epoch=epoch, path=str(path)))
    status["available_checkpoints"] = len(jobs)
    status_path = args.output / "status.json"
    status_path.write_text(json.dumps(status, indent=2) + "\n")
    present = {job[0] for job in jobs}
    if status["missing"] and (not args.progress or present != {"DRGN-AI", opt_label}):
        print(json.dumps(dict(available=len(jobs), pending=len(status["missing"]), status=str(status_path))))
        return 2
    if args.volume_control:
        manifest = json.loads((args.optpose_root/"manifest.json").read_text())
        if Path(manifest["particles"]).resolve() != (args.drgnai_root/"inputs/particles.mrcs").resolve():
            raise ValueError("The fixed-volume control must consume the exact shared image file")
        status["optpose_variant"] = manifest
    evaluator = load_evaluator(args.evaluator)
    truth_path = args.drgnai_root / "evaluation_truth/rotations.pkl"
    ids_path = args.drgnai_root / "evaluation_truth/indices.npy"
    truth = np.asarray(pickle.load(truth_path.open("rb")), dtype=float)
    ids = np.load(ids_path)
    if truth.shape != (100000, 3, 3) or ids.shape != (100000,):
        raise ValueError("Expected the defined 100,000-particle 80S condition")
    if len(np.unique(ids)) != len(ids):
        raise ValueError("Unique particle identities are required")
    records, sources = [], []
    for method, seed, epoch, path in jobs:
        if method == "DRGN-AI":
            rotations = pickle.load(path.open("rb"))[0]
        else:
            with np.load(path) as data:
                if not np.array_equal(data["particle_ids"], ids):
                    raise ValueError(f"Particle correspondence mismatch: {path}")
                rotations = data["rotations"]
        rotations = np.asarray(rotations, dtype=float)
        if rotations.shape != truth.shape or not np.isfinite(rotations).all():
            raise ValueError(f"Invalid rotation array: {path}")
        if (np.max(np.abs(rotations @ rotations.transpose(0, 2, 1)-np.eye(3))) > .002
                or np.max(np.abs(np.linalg.det(rotations)-1)) > .002):
            raise ValueError(f"SO(3) validation failed: {path}")
        metric = evaluator.fit(rotations, truth)["paper_mean_frobenius"]
        records.append(dict(method=method, seed=seed, epoch=epoch, **metric))
        sources.append(dict(path=str(path.resolve()), sha256=digest(path)))
    for seed in args.seeds:
        rotations = Rotation.random(len(truth), random_state=np.random.default_rng(seed)).as_matrix()
        metric = evaluator.fit(rotations, truth)["paper_mean_frobenius"]
        records.append(dict(method="Random SO(3)", seed=seed, epoch=0, **metric))
    frame = pd.DataFrame(records)
    frame.to_csv(args.output / "per_seed_metrics.csv", index=False)
    drgnai_rows = frame[frame.method == "DRGN-AI"]
    final = drgnai_rows.sort_values("epoch").groupby("seed", as_index=False).tail(1)
    status["comparison_complete"] = not bool(status["missing"])
    status["drgnai_final_view_deg"] = final.view_mean_deg.tolist()
    status["accuracy_limit_deg"] = args.accuracy_limit
    status["accuracy_pass"] = bool((final.view_mean_deg < args.accuracy_limit).all())
    status.update(truth_sha256=digest(truth_path), particle_ids_sha256=digest(ids_path),
                  evaluator_sha256=digest(args.evaluator), checkpoints=sources)
    status_path.write_text(json.dumps(status, indent=2) + "\n")
    # Render all measured outcomes, preserving failures as evidence.
    if not status["accuracy_pass"]:
        print("Accuracy target remains pending; rendering measured results with recorded gate status")
    for column, label, folder in [("view_mean_deg", "Viewing-direction error (°)", "viewing_direction"),
                                  ("so3_mean_deg", "Aligned SO(3) error (°)", "full_so3")]:
        aggregate = frame.groupby(["method", "epoch"])[column].agg(value="mean", sd="std").reset_index()
        aggregate["sd"] = aggregate.sd.fillna(0.)
        if args.progress and status["missing"]:
            aggregate = aggregate.drop(columns="sd")
        csv = args.output / f"{folder}.csv"
        aggregate.to_csv(csv, index=False)
        caption = (f"100,000 identical particles; requested DRGN-AI seeds {args.seeds}. "
                   "First 100 candidates select one global alignment by mean Frobenius norm, with global hand selection. "
                   "Actual recorded passes and completed seeds are listed in per_seed_metrics.csv. Random SO(3) uses fixed Haar draws. "
                   "Observed accuracy gate is distinct from paper protocol fidelity; consult status.json and protocol records.")
        if args.volume_control:
            caption += " Optpose is a deterministic fixed external density control using HPS and PoseTable; the atomic deformation model is outside this control. Its prior differs from DRGN-AI."
        if status["missing"]:
            caption = "INTERIM MEASUREMENTS: training remains in progress; uncertainty bands await completed repetitions. " + caption
        else:
            caption += " Bands and bars show sample SD across the six DRGN-AI runs and six random draws; the deterministic control has a single trajectory."
        subprocess.run([sys.executable, str(args.renderer), "--csv", str(csv), "--dataset", "Synthetic 80S",
                        "--methods", opt_label, "Random SO(3)", "DRGN-AI", "--metric-label", label,
                        "--output", str(args.output / folder), "--caption", caption], check=True)
    status["figure"] = "rendered; figure QA pending"
    status_path.write_text(json.dumps(status, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
