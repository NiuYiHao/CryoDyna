"""Stationarity and scale conventions of the pose correlation candidate."""
import unittest
import torch
from cryodyna.optpose.objectives import scaled_masked_cosine_loss
from cryodyna.utils.losses import calc_cor_loss
from cryodyna.utils.dataio import Mask


class PoseObjectiveTests(unittest.TestCase):
    def test_nonzero_mean_perfect_match_is_stationary(self):
        target = torch.linspace(.2,2,256,dtype=torch.float64).reshape(1,1,16,16)
        prediction = target.clone().requires_grad_()
        old = calc_cor_loss(prediction,target)
        old_grad = torch.autograd.grad(old,prediction)[0]
        corrected = scaled_masked_cosine_loss(prediction,target)
        gradient = torch.autograd.grad(corrected,prediction)[0]
        self.assertGreater(float(old_grad.norm()), .1)
        self.assertLess(float(gradient.norm()), 1e-12)

    def test_positive_scale_match_with_mask_is_stationary(self):
        torch.manual_seed(2)
        target = torch.rand(2,1,16,16,dtype=torch.float64)+2
        prediction = (3*target).requires_grad_()
        mask = Mask(16,.9)
        loss = scaled_masked_cosine_loss(prediction,target,mask)
        grad = torch.autograd.grad(loss,prediction)[0]
        self.assertLess(float(grad.norm()),1e-12)
        self.assertAlmostEqual(float(loss),-255/float(mask.num_masked),places=12)

    def test_zero_images_are_finite_and_have_zero_gradient(self):
        prediction = torch.zeros(1,1,16,16,dtype=torch.float64,requires_grad=True)
        loss = scaled_masked_cosine_loss(prediction,torch.zeros_like(prediction))
        gradient = torch.autograd.grad(loss,prediction)[0]
        self.assertTrue(torch.isfinite(gradient).all())
        self.assertEqual(float(loss),0)
        self.assertEqual(float(gradient.norm()),0)


if __name__ == "__main__":
    unittest.main()
