"""Development-only known-structure/pose forward calibration.

Truth is used explicitly for oracle diagnostics, never for training or method
selection on a final test set. This probe performs no parameter updates.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import time

import mrcfile
import numpy as np
import starfile
import torch
from torch.nn import functional as F
from scipy.spatial.transform import Rotation

from cryodyna.gmm.gmm import EMAN2Grid, Gaussian, batch_projection
from cryodyna.utils.ctf_utils import CTFCryoDRGN
from cryodyna.utils.dataio import Mask, StarfileDataSet, StarfileDatasetConfig
from cryodyna.utils.fft_utils import primal_to_fourier_2d, fourier_to_primal_2d
from cryodyna.utils.polymer import Polymer


def checksum(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


class DensitySlice:
    """Centered Cartesian Fourier projection on an even image lattice.

    MRC arrays are ZYX, while the sampled coordinates are XYZ. Coordinates
    transform as k @ R, matching real-space atomic centers transformed by R.
    """
    def __init__(self, density, apix, device="cpu"):
        density = torch.as_tensor(density, dtype=torch.float32, device=device)
        n = density.shape[-1]
        if density.shape != (n, n, n) or n % 2:
            raise ValueError("Expected an even cubic density lattice")
        self.n, self.apix = n, apix
        ft = torch.fft.fftshift(torch.fft.fftn(torch.fft.ifftshift(density)))
        # The positive Nyquist endpoint repeats the negative endpoint. Padding
        # all axes preserves periodic interpolation at quarter-turn boundaries.
        for dim in range(3):
            ft = torch.cat((ft, ft.narrow(dim, 0, 1)), dim=dim)
        self.source = torch.view_as_real(ft).permute(3, 0, 1, 2)[None].contiguous()
        axis = torch.arange(n, device=device, dtype=torch.float32) - n // 2
        yy, xx = torch.meshgrid(axis, axis, indexing="ij")
        self.coords = torch.stack((xx, yy, torch.zeros_like(xx)), -1).reshape(-1, 3)

    def spectra(self, rotations):
        xyz = self.coords[None] @ rotations
        grid = (2 * (xyz + self.n / 2).remainder(self.n) / self.n - 1)
        grid = grid.reshape(1, 1, -1, self.n ** 2, 3)
        sampled = F.grid_sample(self.source, grid, align_corners=True, mode="bilinear")
        sampled = sampled[0, :, 0].permute(1, 2, 0).contiguous()
        return torch.view_as_complex(sampled).reshape(-1, 1, self.n, self.n)


def comparison(predictions, target, mask):
    pred = mask(predictions).flatten(1)
    obs = mask(target).flatten(1).expand_as(pred)
    dot = (pred * obs).sum(1)
    cosine = dot / (pred.norm(dim=1) * obs.norm(dim=1)).clamp_min(1e-12)
    # Same reduction and epsilon as utils.losses.calc_cor_loss, per image.
    train_loss = -dot / (pred.std(1) + 1e-5) / (obs.std(1) + 1e-5) / mask.num_masked
    return cosine.cpu().numpy(), train_loss.cpu().numpy()


def balanced_ids(frame, count, seed):
    if not 1 <= count <= len(frame):
        raise ValueError("Particle count must lie within the dataset")
    rng = np.random.default_rng(seed)
    groups = [rng.permutation(ids) for _, ids in frame.groupby("rlnClassNumber").groups.items()]
    rng.shuffle(groups)
    ids = [int(group[i]) for i in range(max(map(len, groups)))
           for group in groups if i < len(group)]
    return np.sort(ids[:count])


def run(args):
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    root = Path(args.dataset_root).resolve()
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    star = root / "uniform_snr0-0001_ctf/simulation.star"
    content = starfile.read(star)
    frame, optics = content["particles"], content["optics"].iloc[0]
    n, apix = int(optics.rlnImageSize), float(optics.rlnImagePixelSize)
    dataset = StarfileDataSet(StarfileDatasetConfig(
        dataset_dir=str(star.parent), starfile_path=str(star), apix=apix,
        side_shape=n, mask_rad=1.0))
    ids = balanced_ids(frame, args.particles, args.seed)
    np.save(out / "particle_ids.npy", ids)
    grid = EMAN2Grid(n, apix).to(args.device)
    ctf = CTFCryoDRGN(n, apix, kV=float(optics.rlnVoltage),
        cs=float(optics.rlnSphericalAberration),
        amplitudeContrast=float(optics.rlnAmplitudeContrast)).to(args.device)
    loss_mask = Mask(n, 0.9375).to(args.device)
    frequencies = torch.fft.fftshift(torch.fft.fftfreq(n, apix)).to(args.device)
    radial = torch.sqrt(frequencies[:, None] ** 2 + frequencies[None, :] ** 2)
    reference_id = args.reference_class
    records, input_files, densities = [], {star}, {}
    reference = Polymer.from_pdb(root / f"pdbs/1akeA_{reference_id}.pdb")
    angles = np.asarray(args.angles, dtype=float)
    if len(set(angles)) != len(angles) or 0 not in angles:
        raise ValueError("Probe angles must be distinct and include zero")
    identity_idx = int(np.flatnonzero(angles == 0)[0])
    started = time.time()

    def get_density(class_id):
        if class_id not in densities:
            path = root / f"mrcs/1akeA_{class_id}.mrc"
            with mrcfile.open(path) as mrc:
                if abs(float(mrc.voxel_size.x) - apix) > 1e-5:
                    raise ValueError("Density and image pixel sizes differ")
                density = mrc.data.copy()
            densities[class_id] = DensitySlice(density, apix, args.device)
            input_files.add(path)
        return densities[class_id]

    with (out / "probe_metrics.csv").open("w", newline="") as stream, torch.no_grad():
        fields = ["particle_id", "class_id", "model", "target", "bandwidth_A", "axis",
                  "angle_deg", "cosine", "training_correlation_loss"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for number, idx in enumerate(ids):
            raw = dataset[int(idx)]
            batch = {k: v[None].to(args.device) for k, v in raw.items() if torch.is_tensor(v)}
            class_id = int(frame.iloc[idx].rlnClassNumber)
            if max(abs(float(raw["shiftX"])), abs(float(raw["shiftY"]))) > 1e-6:
                raise ValueError("This initial centered-particle diagnostic requires zero translations")
            oracle_path = root / f"pdbs/1akeA_{class_id}.pdb"
            input_files.update([oracle_path, root / f"pdbs/1akeA_{reference_id}.pdb"])
            input_files.add(star.parent / raw["imgname_raw"].split("@")[-1])
            oracle = Polymer.from_pdb(oracle_path)
            for axis_name, axis in zip("xyz", np.eye(3)):
                rotations = torch.tensor(Rotation.from_rotvec(
                    np.radians(angles)[:, None] * axis).as_matrix(),
                    dtype=torch.float32, device=args.device) @ batch["rotmat"]
                for model in ("true_density", "reference_density", "true_residue_gmm", "reference_residue_gmm"):
                    if model.endswith("density"):
                        class_source = class_id if model == "true_density" else reference_id
                        spectrum = get_density(class_source).spectra(rotations)
                    else:
                        polymer = oracle if model == "true_residue_gmm" else reference
                        coords = torch.tensor(polymer.coord, device=args.device)
                        amplitudes = torch.tensor(polymer.num_electron, dtype=torch.float32, device=args.device)
                        projected = batch_projection(Gaussian(
                            coords[None].expand(len(angles), -1, -1),
                            torch.full_like(amplitudes[None], 2.0), amplitudes[None]),
                            rotations, grid.line())[:, None]
                        spectrum = primal_to_fourier_2d(projected)
                    params = {k: v.expand(len(angles), -1, -1) for k, v in batch.items()
                              if k in ("defocusU", "defocusV", "angleAstigmatism")}
                    spectrum = ctf(spectrum, ctf_params=params)
                    for bandwidth in args.bandwidths:
                        filtered = fourier_to_primal_2d(spectrum * (radial < 1 / bandwidth)).real
                        observed = fourier_to_primal_2d(primal_to_fourier_2d(batch["proj"])
                                                       * (radial < 1 / bandwidth)).real
                        for target_name, target in (("observed", observed),
                                                   ("matched_noiseless", filtered[identity_idx:identity_idx+1])):
                            cosines, losses = comparison(filtered, target, loss_mask)
                            for angle, cosine, loss in zip(angles, cosines, losses):
                                row = dict(zip(fields, [int(idx), class_id, model, target_name,
                                    bandwidth, axis_name, float(angle), float(cosine), float(loss)]))
                                writer.writerow(row)
                                records.append(row)
            stream.flush()
            print(f"particle {number+1}/{len(ids)} class={class_id}", flush=True)

    summary = []
    for model in sorted({r["model"] for r in records}):
        for bandwidth in args.bandwidths:
            at_truth = [r for r in records if r["model"] == model and r["bandwidth_A"] == bandwidth
                        and r["target"] == "observed" and r["angle_deg"] == 0 and r["axis"] == "x"]
            minima = []
            for idx in ids:
                for axis in "xyz":
                    group = [r for r in records if r["model"] == model and r["bandwidth_A"] == bandwidth
                             and r["target"] == "observed" and r["axis"] == axis and r["particle_id"] == idx]
                    minima.append(min(group, key=lambda r: r["training_correlation_loss"])["angle_deg"])
            summary.append({"model": model, "bandwidth_A": bandwidth,
                "known_pose_mean_cosine": float(np.mean([r["cosine"] for r in at_truth])),
                "known_pose_mean_training_loss": float(np.mean([r["training_correlation_loss"] for r in at_truth])),
                "axis_probes_minimum_at_truth_fraction": float(np.mean(np.asarray(minima) == 0)),
                "axis_probes_minimum_abs_angle_mean": float(np.mean(np.abs(minima)))})
    metadata = {"status": "completed", "scope": "development oracle diagnostics; no training or test-set claim",
        "particle_count": len(ids), "class_count": len(set(frame.iloc[ids].rlnClassNumber)),
        "arguments": vars(args), "particle_ids": ids.tolist(), "seconds": time.time()-started,
        "summary": summary, "inputs_sha256": {str(p): checksum(p) for p in sorted(input_files)},
        "source_sha256": checksum(__file__),
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=Path(__file__).parents[2], text=True).strip(),
        "method": "native Fourier density slice vs residue GMM sigma=2A/residue electron weights; common v2 CTF/LP/loss mask",
        "interpretation": "Independent axis probes are local loss samples; minima are not recovered poses or accuracy estimates.",
        "ctf_applications_per_prediction": 1}
    (out / "summary.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--particles", type=int, default=32)
    parser.add_argument("--reference-class", type=int, default=50)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--bandwidths", type=float, nargs="+", default=[8, 4])
    parser.add_argument("--angles", type=float, nargs="+", default=[-20, -10, -5, 0, 5, 10, 20])
    run(parser.parse_args())
