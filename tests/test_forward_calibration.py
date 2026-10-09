"""Numerical checks for the diagnostic Fourier coordinate convention."""
import unittest
import numpy as np
import torch

from cryodyna.optpose.calibration import DensitySlice, comparison
from cryodyna.utils.dataio import Mask
from cryodyna.utils.fft_utils import primal_to_fourier_2d, fourier_to_primal_2d
from cryodyna.utils.losses import calc_cor_loss


class ForwardCalibrationTest(unittest.TestCase):
    def test_identity_projection_preserves_offcenter_density_and_phase(self):
        torch.manual_seed(4)
        density = torch.rand(16, 16, 16)
        projector = DensitySlice(density, 1)
        actual = fourier_to_primal_2d(projector.spectra(torch.eye(3)[None])).real
        expected = density.sum(0)[None, None]
        torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)

    def test_quarter_turn_uses_realspace_xyz_rotation_convention(self):
        n = 16
        density = torch.zeros(n, n, n)
        density[n//2, n//2, n//2+2] = 1
        rotation = torch.tensor([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
        actual = fourier_to_primal_2d(DensitySlice(density, 1).spectra(rotation[None])).real[0, 0]
        expected = torch.zeros(n, n)
        expected[n//2+2, n//2] = 1
        torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)

    def test_per_image_loss_matches_training_reduction(self):
        torch.manual_seed(5)
        predicted, observed = torch.rand(3, 1, 16, 16), torch.rand(1, 1, 16, 16)
        mask = Mask(16, .9375)
        _, losses = comparison(predicted, observed, mask)
        expected = calc_cor_loss(predicted, observed.expand_as(predicted), mask)
        self.assertAlmostEqual(float(np.mean(losses)), float(expected), places=6)


if __name__ == "__main__":
    unittest.main()
