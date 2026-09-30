"""Regression gates for shared initialization and 80S metric conventions."""
import tempfile
from pathlib import Path
import unittest
import torch
from cryodyna.optpose.audit import verify_initial_models
from cryodyna.optpose.volume_metrics import validate


class ReleaseAuditTests(unittest.TestCase):
    def test_float32_roundoff_and_real_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            a,b=root/'a',root/'b'; a.mkdir(); b.mkdir()
            base = dict(task={'model.weight':torch.ones(2), 'pose_head.weight':torch.ones(2)},
                        encoder_unregistered={'attn_layers':[{'weight':torch.ones(2)}]})
            torch.save(base,a/'epoch000.pt')
            base['task']['model.weight'] += 5e-7
            base['task']['pose_head.weight'] += 10
            torch.save(base,b/'epoch000.pt')
            result=verify_initial_models([a,b])
            self.assertTrue(result['pass_initialization'])
            self.assertGreater(result['comparisons'][1]['max_absolute_difference'],0)
            base['encoder_unregistered']['attn_layers'][0]['weight'] += .01
            torch.save(base,b/'epoch000.pt')
            with self.assertRaises(AssertionError): verify_initial_models([a,b])

    def test_view_full_angle_and_global_hand(self):
        validate()


if __name__ == '__main__': unittest.main()
