"""Reuse the existing 80S image/CTF input in a pose-free CryoDyna STAR file."""
import argparse
import hashlib
import json
from pathlib import Path
import pickle

import mrcfile
import numpy as np
import pandas as pd
import starfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--drgnai-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    stack = (args.drgnai_root / "inputs/particles.mrcs").resolve()
    ctf_path = (args.drgnai_root / "inputs/ctf.pkl").resolve()
    ctf = np.asarray(pickle.load(ctf_path.open("rb")))
    with mrcfile.mmap(str(stack), mode="r") as images:
        n, height, width = images.data.shape
        if (n, height, width) != (100000, 128, 128):
            raise ValueError("Expected 100,000 128x128 particles")
    if ctf.shape != (n, 9) or not np.isfinite(ctf).all():
        raise ValueError("Expected finite per-image CTF parameters")
    if not np.allclose(ctf[:, 0], width) or not np.allclose(ctf[:, 1], 3.77):
        raise ValueError("Image size or pixel size differs from the 80S condition")
    if (ctf[:, 5:8] != ctf[0, 5:8]).any():
        raise ValueError("This adapter requires shared microscope parameters")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    star_path = output / "particles.star"
    if star_path.exists():
        raise FileExistsError(star_path)
    particles = pd.DataFrame({"rlnImageName": [f"{i+1}@{stack}" for i in range(n)],
                              "rlnOpticsGroup": np.ones(n, dtype=int),
                              "rlnDefocusU": ctf[:, 2], "rlnDefocusV": ctf[:, 3],
                              "rlnDefocusAngle": ctf[:, 4], "rlnPhaseShift": ctf[:, 8]})
    optics = pd.DataFrame({"rlnOpticsGroupName": ["80S"], "rlnOpticsGroup": [1],
                           "rlnImageSize": [width], "rlnImageDimensionality": [2],
                           "rlnImagePixelSize": [float(ctf[0, 1])],
                           "rlnVoltage": [float(ctf[0, 5])],
                           "rlnSphericalAberration": [float(ctf[0, 6])],
                           "rlnAmplitudeContrast": [float(ctf[0, 7])]})
    starfile.write(dict(optics=optics, particles=particles), star_path, overwrite=False)
    np.save(output / "particle_ids.npy", np.arange(n))
    check = starfile.read(star_path)
    assert len(check["particles"]) == n
    np.testing.assert_allclose(check["particles"].rlnDefocusU, ctf[:, 2])
    assert check["particles"].rlnImageName.iloc[-1] == f"{n}@{stack}"
    record = dict(dataset="synthetic 80S", n=n, box=width, apix=float(ctf[0, 1]),
                  particles=str(stack), particle_file_bytes=stack.stat().st_size,
                  ctf=str(ctf_path), ctf_sha256=hashlib.sha256(ctf_path.read_bytes()).hexdigest(),
                  star=str(star_path), star_sha256=hashlib.sha256(star_path.read_bytes()).hexdigest(),
                  pose_access="image and CTF only", particle_order="all 100000 prepared images in stored order",
                  reference_structure="pending 80S atomic reference", training="pending adapter and reference")
    (output / "manifest.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
