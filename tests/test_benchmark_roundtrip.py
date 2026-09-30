"""Optional real-data checkpoint/resume integration test (64 particles, CUDA)."""
import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import pandas as pd
import torch
from cryodyna.optpose import benchmark as benchmark


@unittest.skipUnless(torch.cuda.is_available() and
    (benchmark.ROOT/"tutorial_data_1ake/uniform_snr0-0001_ctf/simulation.star").exists(),
    "Optional integration test requires 1AKE inputs and CUDA")
class ResumeTests(unittest.TestCase):
    def test_resume_matches_continuous(self):
        with tempfile.TemporaryDirectory(prefix="optpose-roundtrip-") as tmp:
            root = Path(tmp)
            parser = benchmark.build_parser()
            args = parser.parse_args(["run", "--outdir", str(root/"continuous"), "--particles", "64",
                "--epochs", "2", "--init-steps", "4", "--angles", "5", "--rotation-only",
                "--methods", "opt", "--pose-model", "table", "--batch-size", "32", "--eval-batch-size", "32"])
            benchmark.run_one(args, "opt", 5., root/"continuous")
            interrupted = copy.deepcopy(args); interrupted.outdir = str(root/"resumed")
            save = benchmark.save_checkpoint
            def interrupt_after_saved_epoch(path, state):
                save(path, state)
                if Path(path).name == "resume.pt" and state["epoch"] == 1:
                    raise KeyboardInterrupt("Controlled interruption after durable epoch boundary")
            with patch.object(benchmark, "save_checkpoint", side_effect=interrupt_after_saved_epoch):
                with self.assertRaises(KeyboardInterrupt):
                    benchmark.run_one(interrupted, "opt", 5., root/"resumed")
            interrupted.resume = True
            benchmark.run_one(interrupted, "opt", 5., root/"resumed")
            a = pd.read_csv(root/"continuous/metrics.csv")
            b = pd.read_csv(root/"resumed/metrics.csv")
            for key in ("loss", "rmsd_A", "pose_deg", "train_loss"):
                torch.testing.assert_close(torch.tensor(a[key].to_numpy()), torch.tensor(b[key].to_numpy()),
                                           atol=1e-5, rtol=1e-5, equal_nan=True)
            from cryodyna.optpose.audit import verify_predictions
            import json
            manifest = json.loads((root/"resumed/manifest.json").read_text())
            verify_predictions(root/"resumed", manifest, device="cuda")


if __name__ == "__main__": unittest.main(verbosity=2)
