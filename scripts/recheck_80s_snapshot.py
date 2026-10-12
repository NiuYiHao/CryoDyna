"""Independently recompute completed 80S endpoints from saved frame corrections."""
import argparse
import json
from pathlib import Path
import pickle

import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evaluation', type=Path, required=True)
    parser.add_argument('--snapshot', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    table = pd.read_csv(args.evaluation/'per_seed_metrics.csv')
    status = json.loads((args.evaluation/'status.json').read_text())
    drgn = table[table.method == 'DRGN-AI']
    assert len(drgn) == 30*len(status['completed_seeds'])
    assert drgn.groupby('epoch').seed.apply(set).map(lambda x: x == set(status['completed_seeds'])).all()
    truth = np.asarray(pickle.load((args.snapshot/'evaluation_truth/rotations.pkl').open('rb')), dtype=np.float64)
    checks = []
    for row in drgn[drgn.epoch == 30].itertuples():
        detail = json.loads((args.evaluation/'metrics'/row.cache_file).read_text())
        rotation, _ = pickle.load((args.snapshot/f'seed{row.seed}/out/pose.29.pkl').open('rb'))
        metric = detail['alignments']['paper_mean_frobenius']
        hand = np.diag([1., 1., -1. if metric['hand_flip'] else 1.])
        aligned = hand @ np.asarray(rotation, dtype=np.float64) @ np.array(metric['right_alignment'])
        v, t = aligned[:, 2, :], truth[:, 2, :]
        view = np.rad2deg(np.arctan2(np.linalg.norm(np.cross(v, t), axis=1), np.sum(v*t, axis=1)))
        relative = aligned @ truth.transpose(0, 2, 1)
        so3 = np.rad2deg(np.arccos(np.clip((np.trace(relative, axis1=1, axis2=2)-1)/2, -1, 1)))
        np.testing.assert_allclose([view.mean(), so3.mean()], [row.view_mean_deg, row.so3_mean_deg], atol=1e-8, rtol=0)
        assert np.mean(view < 5) == row.view_fraction_below5
        checks.append(dict(seed=int(row.seed), n=len(view), view_mean_deg=float(view.mean()),
                           so3_mean_deg=float(so3.mean()), view_max_deg=float(view.max()),
                           view_fraction_below5=float(np.mean(view < 5)),
                           max_mean_difference_deg=max(abs(view.mean()-row.view_mean_deg), abs(so3.mean()-row.so3_mean_deg))))
    for name, column in [('viewing_direction', 'view_mean_deg'), ('full_so3', 'so3_mean_deg')]:
        aggregate = pd.read_csv(args.evaluation/f'{name}.csv').query("method == 'DRGN-AI'").set_index('epoch')
        for epoch, group in drgn.groupby('epoch'):
            values = group[column].to_numpy()
            np.testing.assert_allclose([aggregate.loc[epoch, 'value'], aggregate.loc[epoch, 'sd']],
                                       [np.mean(values), np.std(values, ddof=1)], atol=1e-12, rtol=0)
    result = dict(status='pass', scope='independent angle and aggregate arithmetic using saved global frame/hand',
                  fixed_cohort=True, endpoint_checks=checks, aggregate_rows_checked=60)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/'independent_recheck.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
