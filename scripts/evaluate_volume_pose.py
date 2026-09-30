"""Evaluate every saved fixed-density pose epoch against explicit truth rotations."""
import argparse
import hashlib
import json
from pathlib import Path
import pickle
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from cryodyna.optpose.volume_metrics import fit, angles, validate


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''): h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--truth', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args(); validate()
    args.output.mkdir(parents=True, exist_ok=True)
    truth = np.asarray(pickle.load(args.truth.open('rb')), dtype=float)
    rows, sources, ids_previous = [], [], None
    for path in sorted((args.run/'evaluation').glob('epoch*.npz')):
        with np.load(path) as saved:
            ids = saved['particle_ids']; pred = saved['rotations'].astype(float)
        if len(np.unique(ids)) != len(ids): raise ValueError('Unique particle IDs required')
        if ids_previous is not None: np.testing.assert_array_equal(ids_previous, ids)
        ids_previous = ids.copy()
        if pred.shape != (len(ids),3,3) or not np.isfinite(pred).all(): raise ValueError('Invalid rotations')
        np.testing.assert_allclose(pred @ pred.transpose(0,2,1), np.broadcast_to(np.eye(3), pred.shape), atol=.002)
        np.testing.assert_allclose(np.linalg.det(pred), 1., atol=.002)
        target = truth[ids]
        metric = fit(pred, target)['paper_mean_frobenius']; view, so3 = angles(pred, target)
        rows.append(dict(epoch=int(path.stem[5:]),n=len(ids),raw_view_mean_deg=float(view.mean()),
                         raw_so3_mean_deg=float(so3.mean()), checkpoint=str(path.resolve()), **metric))
        sources.append(dict(path=str(path.resolve()),sha256=digest(path)))
    if not rows: raise ValueError('Saved epoch predictions are required')
    frame = pd.DataFrame(rows)
    frame.to_csv(args.output/'metrics.csv', index=False)
    tidy = []
    for method, key in [('Viewing direction', 'view_mean_deg'), ('Full SO(3)', 'so3_mean_deg')]:
        tidy.extend(dict(method=method,epoch=int(row.epoch),value=getattr(row,key)) for row in frame.itertuples())
    pd.DataFrame(tidy).to_csv(args.output/'curve_endpoint.csv', index=False)
    manifest = json.loads((args.run/'manifest.json').read_text())
    inputs = [Path(manifest[k]) for k in ['particles','ctf','volume']]+[args.truth]
    (args.output/'provenance.json').write_text(json.dumps(dict(
        checkpoints=sources, inputs=[dict(path=str(path.resolve()),sha256=digest(path)) for path in inputs],
        manifest=manifest, evaluator_sha256=digest(Path(__file__).resolve().parents[1]/'cryodyna/optpose/volume_metrics.py'),
        metric='first min(100,N) candidates; global hand and frame; mean Frobenius selection; all-particle fit and evaluation'),indent=2)+'\n')
    print(frame[['epoch','n','view_mean_deg','so3_mean_deg']].to_string(index=False))


if __name__ == '__main__': main()
