import tempfile
from pathlib import Path
import unittest

import mrcfile
import numpy as np
import torch
from cryodyna.optpose.volume_prior import FourierVolume


class VolumePriorTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name)/"volume.mrc"
        self.vol = np.random.default_rng(3).normal(size=(16, 16, 16)).astype(np.float32)
        with mrcfile.new(str(self.path)) as m:
            m.set_data(self.vol); m.voxel_size = 3.77
        self.model = FourierVolume(self.path, 5, [16, 3.77, 15000, 15000, 0, 300, 2, .1, 0], image_side=16)

    def tearDown(self):
        self.tmp.cleanup()

    def test_identity_slice_is_centered_fft_plane(self):
        ft = torch.fft.fftshift(torch.fft.fftn(torch.fft.ifftshift(torch.tensor(self.vol))))
        expected = ft[8][self.model.mask]/ft.abs().max()*self.model.ctf
        actual = self.model(torch.eye(3)[None])[0]
        torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)

    def test_translation_order_is_yx_pixels(self):
        r = torch.eye(3)[None]
        base = self.model(r)
        shift = torch.tensor([[2., -1.]])
        xy = self.model.coordinates[:, :2]
        expected = base*torch.exp(-2j*torch.pi*(2*xy[:, 1]-xy[:, 0])/16)
        torch.testing.assert_close(self.model(r, shift), expected, atol=1e-6, rtol=1e-5)

    def test_rotation_gradient_matches_finite_difference(self):
        from scipy.spatial.transform import Rotation
        r = torch.tensor(Rotation.from_rotvec([.19, .31, -.11]).as_matrix(), dtype=torch.float32)[None].requires_grad_()
        loss = self.model(r).abs().square().sum()
        loss.backward()
        step = 1e-4
        plus, minus = r.detach().clone(), r.detach().clone()
        plus[0, 0, 1] += step; minus[0, 0, 1] -= step
        finite = (self.model(plus).abs().square().sum()-self.model(minus).abs().square().sum())/(2*step)
        torch.testing.assert_close(r.grad[0, 0, 1], finite, atol=.03, rtol=.02)


if __name__ == "__main__": unittest.main()
