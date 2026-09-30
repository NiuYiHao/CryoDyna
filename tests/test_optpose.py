import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from cryodyna.optpose.geometry import angular_error, perturb_rotations, fit_gauge, kabsch_rmsd
from cryodyna.optpose.pose import PoseTable, subdivide
from cryodyna.utils import so3_grid


class GeometryTests(unittest.TestCase):
    def test_angles_and_perturbation(self):
        truth = Rotation.random(50, random_state=3).as_matrix()
        for angle in [0, 5, 10, 15, 20, 180]:
            np.testing.assert_allclose(angular_error(perturb_rotations(truth, angle, 7), truth), angle, atol=3e-6)
        # In-plane error is part of the metric.
        np.testing.assert_allclose(angular_error(Rotation.from_euler("z", [30], degrees=True).as_matrix(), np.eye(3)[None]), 30)

    def test_global_gauge_and_hand(self):
        truth = Rotation.random(60, random_state=2).as_matrix()
        q = Rotation.random(random_state=8).as_matrix()
        s = np.diag([1., 1., -1.])
        for pred in [truth @ q, s @ truth @ q @ s]:
            aligned, _ = fit_gauge(pred, truth, np.arange(20))
            self.assertLess(angular_error(aligned, truth).max(), 3e-6)

    def test_atomic_rmsd(self):
        x = np.random.default_rng(9).normal(size=(5, 20, 3))
        q = Rotation.random(random_state=2).as_matrix()
        np.testing.assert_allclose(kabsch_rmsd(x @ q + 3, x), 0, atol=1e-12)
        self.assertGreater(kabsch_rmsd(x * 2, x).mean(), .5)

    def test_table_gradient_and_save_restore(self):
        table = PoseTable(4)
        ids = torch.tensor([1, 3])
        truth = torch.tensor(Rotation.random(2, random_state=1).as_matrix(), dtype=torch.float32)
        table.initialize(ids, truth, torch.zeros(2, 2))
        optimizer = torch.optim.Adam(table.parameters(), lr=.01)
        initial = float((table(ids)[0]-torch.eye(3)).square().sum().detach())
        for _ in range(20):
            optimizer.zero_grad()
            (table(ids)[0]-torch.eye(3)).square().sum().backward()
            optimizer.step()
        self.assertLess(float((table(ids)[0]-torch.eye(3)).square().sum().detach()), initial)
        r, _ = table(ids)
        torch.testing.assert_close(r @ r.transpose(-1, -2), torch.eye(3).expand(2, 3, 3))
        self.assertTrue((r.det() > .99999).all())
        saved = io.BytesIO()
        torch.save(table.state_dict(), saved)
        saved.seek(0)
        restored = PoseTable(4)
        restored.load_state_dict(torch.load(saved, weights_only=True))
        torch.testing.assert_close(restored(ids)[0], r)
        self.assertFalse(restored.valid[0])

    def test_subdivision_against_scalar_grid(self):
        q = torch.from_numpy(so3_grid.grid_SO3(1)[[0, 59, 120]])
        ids = so3_grid.get_base_ind(np.array([0, 59, 120]), 1)
        result, result_ids, _ = subdivide(q, ids, 1, "cpu")
        for i in range(3):
            children, child_ids = so3_grid.get_neighbor(q[i].numpy(), *ids[i], 1)
            # Ties may reorder equally distant children; compare sets.
            self.assertEqual(set(map(tuple, result_ids[i])), set(map(tuple, child_ids)))
            self.assertLess(np.min(np.linalg.norm(result[i].numpy()[:, None]-children, axis=-1), axis=1).max(), 1e-6)


class OutputContractTests(unittest.TestCase):
    def make_run(self, root, name, method="original", batch_size=64):
        import pandas as pd
        directory = Path(root)/name; directory.mkdir()
        manifest = dict(schema=1, kind="run", method=method, pose_init="given", angle=5, seed=1,
            particle_ids=[0, 1, 2], evaluation_ids=[1, 2], alignment_ids=[0], truth_sha256="test",
            arguments=dict(batch_size=batch_size, eval_batch_size=128, init_steps=128, sample_seed=101, epochs=1))
        (directory/"manifest.json").write_text(json.dumps(manifest))
        pd.DataFrame([dict(epoch=1, rmsd_A=1., pose_deg=5., loss=-.2)]).to_csv(directory/"metrics.csv", index=False)
        return directory

    def test_empty_matrix_rejected(self):
        from cryodyna.optpose.benchmark import collect_runs
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp)/"manifest.json").write_text(json.dumps(dict(kind="matrix", jobs=[])))
            with self.assertRaisesRegex(ValueError, "Empty"):
                collect_runs(tmp)

    def test_comparison_rejects_different_budgets(self):
        from cryodyna.optpose.plotting import plot_comparison
        with tempfile.TemporaryDirectory() as tmp:
            a = self.make_run(tmp, "a")
            b = self.make_run(tmp, "b", "opt", 32)
            with self.assertRaisesRegex(ValueError, "batching"):
                plot_comparison([f"Original={a}", f"Opt={b}"], Path(tmp)/"out")

    def test_comparison_rejects_particle_mismatch(self):
        from cryodyna.optpose.plotting import plot_comparison
        with tempfile.TemporaryDirectory() as tmp:
            a = self.make_run(tmp, "a")
            b = self.make_run(tmp, "b", "opt")
            manifest = json.loads((b/"manifest.json").read_text())
            manifest["particle_ids"] = [2, 1, 0]
            (b/"manifest.json").write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "identical particles"):
                plot_comparison([f"Original={a}", f"Opt={b}"], Path(tmp)/"out")

    def test_cli_array_and_labels(self):
        from cryodyna.optpose.benchmark import build_parser
        parser = build_parser()
        args = parser.parse_args(["run", "--outdir", "x", "--angles", "0", "2.5", "17"])
        self.assertEqual(args.angles, [0, 2.5, 17])
        args = parser.parse_args(["run", "--outdir", "x", "--angles", "[0, 2.5, 17]"])
        self.assertEqual(args.angles, [0, 2.5, 17])
        args = parser.parse_args(["compare", "--runs", "A=x", "B=y", "--output", "z"])
        self.assertEqual(args.runs, ["A=x", "B=y"])

    def test_resume_rejects_matrix_change_before_writing(self):
        from cryodyna.optpose.benchmark import build_parser, run_matrix
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"manifest.json"
            content = json.dumps(dict(schema=1, kind="matrix", pose_init="given", jobs=[]))
            path.write_text(content)
            args = build_parser().parse_args(["run", "--outdir", tmp, "--resume"])
            with self.assertRaisesRegex(ValueError, "same matrix"):
                run_matrix(args)
            self.assertEqual(path.read_text(), content)

    def test_learning_plot_shared_axis_contains_all_angles(self):
        from cryodyna.optpose.plotting import plot_learning
        import pandas as pd
        import matplotlib.pyplot as plt
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); jobs = []
            for angle in [0, 5, 10, 15, 20]:
                directory = self.make_run(tmp, f"a{angle}")
                manifest = json.loads((directory/"manifest.json").read_text())
                manifest["angle"] = angle
                (directory/"manifest.json").write_text(json.dumps(manifest))
                pd.DataFrame([dict(epoch=1, rmsd_A=1., pose_deg=angle, loss=-.2)]).to_csv(directory/"metrics.csv", index=False)
                jobs.append(dict(path=directory.name))
            (root/"manifest.json").write_text(json.dumps(dict(kind="matrix", jobs=jobs)))
            def check_limits(fig, stem):
                self.assertEqual(len(fig.axes), 15)
                self.assertTrue(all(axis.get_ylim()[1] > 20 for axis in fig.axes[10:]))
                plt.close(fig)
            with patch("cryodyna.optpose.plotting.save_figure", side_effect=check_limits):
                plot_learning(root, root/"figures")

    def test_pilot_cannot_pass_release_acceptance(self):
        from cryodyna.optpose.audit import acceptance
        import pandas as pd
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            jobs = []
            for angle in [0, 5, 10, 15, 20]:
                for method in ["original", "opt"]:
                    directory = self.make_run(tmp, f"{method}{angle}", method)
                    manifest = json.loads((directory/"manifest.json").read_text())
                    manifest.update(angle=angle, status="complete")
                    (directory/"manifest.json").write_text(json.dumps(manifest))
                    pd.DataFrame([dict(epoch=1, rmsd_A=1. if method == "original" else .9,
                        pose_deg=angle if method == "original" else angle*.9, loss=-.2)]).to_csv(directory/"metrics.csv", index=False)
                    jobs.append(dict(path=directory.name))
            (root/"manifest.json").write_text(json.dumps(dict(kind="matrix", jobs=jobs)))
            with patch("cryodyna.optpose.audit.verify_predictions") as replay, patch(
                    "cryodyna.optpose.audit.verify_initial_models", return_value={"pass_initialization": True}):
                report = acceptance(root, root/"acceptance.json")
            self.assertEqual(replay.call_count, 10)
            self.assertTrue(all(c["pass"] for c in report["checks"]))
            self.assertFalse(report["perturbation_pass"])

    def test_abinit_input_strips_truth(self):
        import mrcfile
        import pandas as pd
        import starfile
        from mmengine import Config
        from cryodyna.optpose.benchmark import prepare_inputs, read_star
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with mrcfile.new(root/"particles.mrcs") as stream:
                stream.set_data(np.zeros((3, 16, 16), dtype=np.float32))
            frame = pd.DataFrame(dict(rlnImageName=[f"{i+1}@particles.mrcs" for i in range(3)],
                rlnAngleRot=[5, 6, 7], rlnAngleTilt=[30, 40, 50], rlnAnglePsi=[8, 9, 10],
                rlnOriginXAngst=[1., 2., 3.], rlnOriginYAngst=[2., 3., 4.],
                rlnClassNumber=[1, 2, 3], rlnDefocusU=[10000.]*3, rlnDefocusV=[10000.]*3,
                rlnDefocusAngle=[0.]*3))
            starfile.write(frame, root/"input.star")
            cfg = Config(dict(dataset_attr=dict(starfile_path=str(root/"input.star"), dataset_dir=str(root), apix=1.)))
            prepare_inputs(cfg, root/"train", "hps", 0, 1, np.arange(3))
            _, actual = read_star(root/"train/particles.star")
            self.assertFalse(any(c.startswith(("rlnAngle", "rlnOrigin", "rlnClass")) for c in actual))
            self.assertEqual(actual.rlnImageName.tolist(), frame.rlnImageName.tolist())


if __name__ == "__main__":
    unittest.main()
