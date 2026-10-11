"""Locked 2x2 diagnostic: corner-spectrum weighting and Fourier interpolation.

True-class densities are oracle priors. Search receives density, CTF and images.
True rotations are used for synthetic closure and post-search evaluation only.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import mrcfile
import numpy as np
import torch
from torch.nn import functional as F
from cryodyna.optpose.benchmark import resolve_config, make_dataset, read_star
from cryodyna.optpose.volume_prior import normalized_features
from cryodyna.utils.ctf import compute_ctf
from diagnose_1ake_refinement import search, score, angles, write_csv

RADII = (12, 17, 22, 27, 32)


def fft2(image):
    return torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(image, dim=(-2, -1))), dim=(-2, -1))


def corner_profile(images):
    """Estimate radial spectral shape from four 32x32 Hann-windowed corners."""
    window = torch.hann_window(32, periodic=False)
    window = window[:, None] * window[None, :]
    radial = torch.stack(torch.meshgrid(torch.arange(128)-64, torch.arange(128)-64,
                                      indexing="ij")).float().square().sum(0).sqrt()
    bins = (radial / 4).long()
    profiles = []
    for image in images:
        image = image - image.mean()
        patches = torch.stack([image[y:y+32, x:x+32] for y in (0, 96) for x in (0, 96)])
        # Padding here samples the windowed-patch spectrum on the image frequency grid.
        psd = torch.fft.fftshift(torch.fft.fft2(patches * window, s=(128, 128)), dim=(-2, -1)).abs().square().mean(0)
        psd /= psd[radial > 0].mean()
        profiles.append(torch.stack([psd[bins == i].mean() for i in range(int(bins.max())+1)]))
    profile = torch.stack(profiles).mean(0)
    profile /= profile[:9].mean()
    return profile


def volume_fft(path, factor):
    with mrcfile.open(str(path)) as handle:
        density = torch.from_numpy(handle.data.copy()).float()
        assert density.shape == (128, 128, 128)
        assert all(abs(float(handle.voxel_size[a])-1) < 1e-6 for a in ("x", "y", "z"))
    if factor == 2:
        density = F.pad(density, (64, 64, 64, 64, 64, 64))
    spectrum = torch.fft.fftshift(torch.fft.fftn(torch.fft.ifftshift(density)))
    spectrum /= spectrum.abs().max()
    return torch.view_as_real(spectrum).permute(3, 0, 1, 2)[None].contiguous()


class Projection:
    def __init__(self, spectrum, factor, radius, cp, profile=None):
        self.volume, self.factor = spectrum, factor
        self.nfft = 128 * factor
        axis = torch.arange(128).float()-64
        yy, xx = torch.meshgrid(axis, axis, indexing="ij")
        self.mask = (xx.square()+yy.square() <= radius**2) & (xx.square()+yy.square() > 0)
        self.coordinates = torch.stack([xx[self.mask], yy[self.mask], torch.zeros_like(xx[self.mask])], -1)
        parameters = torch.as_tensor(cp, dtype=torch.float32)
        self.ctf = -compute_ctf(self.coordinates[:, :2]/128, *parameters[2:])
        self.weight = (torch.ones(len(self.ctf)) if profile is None else
                       profile[(self.coordinates[:, :2].norm(dim=1)/4).long()].clamp_min(.1).rsqrt())

    def __call__(self, rotations):
        xyz = (self.coordinates[None] @ rotations) * self.factor
        grid = (2*(xyz+self.nfft/2)/(self.nfft-1)-1).reshape(1, 1, -1, len(self.ctf), 3)
        values = F.grid_sample(self.volume, grid, align_corners=True, mode="bilinear")
        values = values[0, :, 0].permute(1, 2, 0).contiguous()
        return torch.view_as_complex(values) * self.ctf * self.weight

    def observations(self, images):
        return fft2(images)[:, self.mask] * self.weight


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--particle-ids", type=Path,
                        default=Path("output/pose_refinement_20261010/validation_ids.json"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "protocol.json").exists():
        raise FileExistsError("Use a fresh directory for a new experiment")
    torch.set_num_threads(4)
    cfg = resolve_config("projects/atom_configs/1ake_optpose.py")
    _, frame = read_star(cfg.dataset_attr.starfile_path)
    ids = json.loads(args.particle_ids.read_text())
    protocol = dict(particle_ids=ids, scope="fixed true-class density; repeated diagnostic on locked128",
        variants=["native_cosine", "native_weighted", "padded_cosine", "padded_weighted"],
        radii=RADII, beam=8, initial_grid=4608, scored_children_per_refinement=64,
        translation="fixed zero", metric="raw full SO3 geodesic, float64",
        truth_access="synthetic closure and evaluation only",
        weighting="inverse sqrt corner-estimated radial PSD; four32x32 Hann corners per image; bin width4; floor0.1",
        weighting_scope="same128 images provide an unsupervised background estimate; signal leakage into corners remains possible",
        padding="real-space centered zero padding128 to256 before3D FFT; frequency coordinates scaled2; image lattice and CTF unchanged",
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        search_source_sha256=hashlib.sha256(Path(__file__).with_name("diagnose_1ake_refinement.py").read_bytes()).hexdigest(),
        star_sha256=hashlib.sha256(Path(cfg.dataset_attr.starfile_path).read_bytes()).hexdigest(),
        original_snr="source formula unresolved", fixed_before_run=True)
    (args.output / "protocol.json").write_text(json.dumps(protocol, indent=2))
    images = {}
    for idx in ids:
        k, path = str(frame.iloc[idx].rlnImageName).split("@")
        with mrcfile.mmap(str(Path(cfg.dataset_attr.dataset_dir)/path), permissive=True) as f:
            images[idx] = torch.from_numpy(f.data[int(k)-1].copy())
    profile = corner_profile([images[i] for i in ids])
    write_csv(args.output / "corner_spectrum.csv", [dict(radius_low=4*i, radius_high=4*(i+1),
        relative_psd=float(value), feature_weight=float(value.clamp_min(.1).rsqrt())) for i, value in enumerate(profile)])
    dataset = make_dataset(cfg, "given")
    old = np.load("output/pose_refinement_20261010/validation128/beams.npz")
    rows, matches, array = [], [], {"particle_ids": np.array(ids)}
    replay, identities = [], []
    current_class, spectra = None, None
    start = time.time()
    with torch.no_grad():
        for position, idx in enumerate(sorted(ids, key=lambda i: int(frame.iloc[i].rlnClassNumber))):
            row = frame.iloc[idx]
            class_id = int(row.rlnClassNumber)
            if class_id != current_class:
                path = Path("tutorial_data_1ake/mrcs")/f"1akeA_{class_id}.mrc"
                spectra = {factor: volume_fft(path, factor) for factor in (1, 2)}
                current_class = class_id
            cp = [128, 1, float(row.rlnDefocusU), float(row.rlnDefocusV), float(row.rlnDefocusAngle), 300, 2.7, .1, 0]
            model_sets = {f"{name}_{weighting}": {r: Projection(spectra[factor], factor, r, cp,
                profile if weighting == "weighted" else None) for r in RADII}
                for name, factor in (("native", 1), ("padded", 2)) for weighting in ("cosine", "weighted")}
            truth = dataset[idx]["rotmat"]
            array[f"{idx}_evaluation_truth"] = truth.numpy()
            identity = torch.eye(3)[None]
            native = model_sets["native_cosine"][32]
            padded = model_sets["padded_cosine"][32]
            identities.append(float((native(identity)-padded(identity)).abs().norm()/native(identity).abs().norm()))
            clean_images = {}
            for name, model in (("native", native), ("padded", padded)):
                spectrum = torch.zeros((1, 128, 128), dtype=torch.complex64)
                spectrum[:, model.mask] = model(truth[None])
                clean = torch.fft.fftshift(torch.fft.ifft2(torch.fft.ifftshift(spectrum, dim=(-2, -1))), dim=(-2, -1)).real
                clean_images[f"clean_{name}"] = clean/clean.std(unbiased=False)
            conditions = {"observed": images[idx][None], **clean_images}
            for condition, image in conditions.items():
                for variant, models in model_sets.items():
                    if condition != "observed" and variant.endswith("weighted"):
                        continue
                    beams, _ = search(models, image, RADII, 8, {})
                    winners = torch.stack([b[0] for b in beams])
                    array[f"{idx}_{condition}_{variant}"] = winners.numpy()
                    if condition == "observed" and variant == "native_cosine":
                        replay.append(float(np.max(np.abs(winners.numpy()-old[f"{idx}_observed_growing_k8"][:, 0]))))
                    for stage, radius in enumerate(RADII):
                        rows.append(dict(particle=idx, class_id=class_id, condition=condition, variant=variant,
                            stage=stage, radius=radius, angle_deg=float(angles(winners[stage], truth)),
                            score=float(score(models[radius], winners[stage:stage+1], image)[0])))
            observed = images[idx][None]
            a, b = native(truth[None]), padded(truth[None])
            matches.append(dict(particle=idx, native_true_cosine=float(score(native, truth[None], observed)[0]),
                padded_true_cosine=float(score(padded, truth[None], observed)[0]),
                native_padded_cosine=float((normalized_features(a)*normalized_features(b)).sum())))
            print(json.dumps(dict(completed=position+1, total=len(ids), elapsed_s=time.time()-start)), flush=True)
    write_csv(args.output / "stages.csv", rows)
    write_csv(args.output / "true_pose_forward.csv", matches)
    np.savez_compressed(args.output / "predictions.npz", **array)
    summary = {}
    for condition in ("observed", "clean_native", "clean_padded"):
        summary[condition] = {}
        for variant in protocol["variants"]:
            values = np.array([r["angle_deg"] for r in rows if r["condition"] == condition and r["variant"] == variant and r["stage"] == 4])
            if len(values):
                summary[condition][variant] = dict(n=len(values), mean_deg=float(values.mean()),
                    median_deg=float(np.median(values)), fraction_below_5=float((values<5).mean()))
    summary["replay_max_rotation_abs_difference"] = max(replay)
    summary["identity_projection_relative_difference_max"] = max(identities)
    summary["forward_match"] = {k:float(np.mean([r[k] for r in matches])) for k in matches[0] if k != "particle"}
    summary["elapsed_s"] = time.time()-start
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
