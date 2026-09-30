"""Numerical HPS contract checks against the local drgn-ai implementation."""
import sys
from pathlib import Path
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'projects'), str(ROOT)]
from train_atom_with_pose import HierarchicalPoseSearch, CryoEMTask, PoseHeadSmallMLP
from cryodyna.gmm.gmm import EMAN2Grid
from cryodyna.utils.ctf_utils import CTFIdentity, CTFCryoDRGN
from cryodyna.utils.fft_utils import primal_to_fourier_2d, fourier_to_primal_2d
from cryodyna.utils.rotation_conversion import axis_angle_to_matrix
try:
    from develop.drgnai.src import pose_search as reference
    from develop.drgnai.src.lattice import Lattice
    from develop.drgnai.src.mask import get_circular_mask
    HAS_REFERENCE = True
except ImportError:
    HAS_REFERENCE = False

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32 = False


def make_search(cls=HierarchicalPoseSearch, **overrides):
    cfg = dict(base_healpy=1, n_iter=3, n_kept_poses=8, l_start=4, l_end=12,
               t_extent=3., t_n_grid=3, hps_chunk_size=64, hps_particle_chunk_size=4)
    cfg.update(overrides)
    return cls(cfg, torch.ones(16, device=DEVICE), torch.linspace(.5, 2., 16, device=DEVICE),
               EMAN2Grid(32, 1.).to(DEVICE), CTFIdentity(), None)


def fixture(search, count=3):
    torch.manual_seed(17)
    centers = torch.randn(1, 16, 3, device=DEVICE) * 4
    structures = centers.expand(count, -1, -1).contiguous()
    rotations = axis_angle_to_matrix(torch.randn(count, 3, device=DEVICE))
    shifts = torch.tensor([[1., -1.], [-1., 2.], [0., 0.]], device=DEVICE)[:count]
    projections = search._shared_projection(structures, rotations)
    axis = torch.fft.fftshift(torch.fft.fftfreq(32, device=DEVICE))
    phase = 2 * torch.pi * (shifts[:, 0, None, None] * axis[None, :, None]
                          + shifts[:, 1, None, None] * axis[None, None, :])
    images = fourier_to_primal_2d(primal_to_fourier_2d(projections) * torch.exp(1j * phase[:, None])).real
    return dict(proj=images, idx=torch.arange(count, device=DEVICE)), structures, rotations, shifts


class HartleyReference(HierarchicalPoseSearch):
    """Use upstream translation, interpolation and selection for all rounds."""
    def translate_images(self, images, shifts, mask):
        lattice = Lattice(images.shape[-1], device=images.device, ignore_dc=self.ignore_dc)
        hartley = images.real - images.imag
        translated = lattice.translate_ht(hartley[:, mask], shifts.flip(-1), mask=mask.flatten())
        return torch.complex((translated + translated.flip(-1))/2,
                             (translated.flip(-1) - translated)/2)

    def _rotate_base_spectra(self, spectra, angles, mask):
        lattice = Lattice(spectra.shape[-1], device=spectra.device, ignore_dc=self.ignore_dc)
        h = (spectra.real - spectra.imag)[..., mask][:, None]
        rotated = reference.rotate_images(h, angles, self._radius, lattice.coords[mask.flatten()], lattice)[:, 0]
        return torch.complex((rotated + rotated.flip(-1))/2,
                             (rotated.flip(-1) - rotated)/2)

    def get_frequency_mask(self, res, radius, device):
        self._radius = radius
        lattice = Lattice(res, device=device, ignore_dc=self.ignore_dc)
        return get_circular_mask(lattice, radius).reshape(res, res)

    def compute_err(self, images, projections):
        i, p = images.real - images.imag, projections.real - projections.imag
        return .5 * p.square().sum(-1)[:, None] - i @ p.transpose(-1, -2)

    def keep_matrix(self, loss, batch_size, max_poses):
        if not hasattr(self, 'trace'):
            self.trace = []
        result = reference.keep_matrix(loss, batch_size, max_poses)
        self.trace.append((loss.detach().cpu(), result.detach().cpu()))
        return tuple(result)


class TracedSearch(HierarchicalPoseSearch):
    def keep_matrix(self, loss, batch_size, max_poses):
        if not hasattr(self, 'trace'):
            self.trace = []
        result = super().keep_matrix(loss, batch_size, max_poses)
        self.trace.append((loss.detach().cpu(), torch.stack(result).cpu()))
        return result


class HPSTests(unittest.TestCase):
    @unittest.skipUnless(HAS_REFERENCE, "Optional local DRGN-AI parity reference")
    def test_phase_hartley_and_integer_shift(self):
        search = make_search()
        torch.manual_seed(5)
        images = torch.randn(2, 32, 32, device=DEVICE)
        ft = search._symmetric_spectrum(primal_to_fourier_2d(images))
        for radius in [4, 16]:
            mask = search.get_frequency_mask(33, radius, DEVICE)
            shifts = torch.tensor([[2., -3.], [.3, -.7]], device=DEVICE)
            actual = search.translate_images(ft, shifts, mask)
            lattice = Lattice(33, device=DEVICE)
            expected = lattice.translate_ht((ft.real-ft.imag)[:, mask], shifts.flip(-1), mask=mask.flatten())
            torch.testing.assert_close(actual.real-actual.imag, expected, atol=3e-4, rtol=2e-5)
            rolled = search._symmetric_spectrum(primal_to_fourier_2d(torch.roll(images, (2,-3), (-2,-1))))
            torch.testing.assert_close(actual[:,0], rolled[:,mask], atol=3e-4, rtol=2e-5)

    @unittest.skipUnless(HAS_REFERENCE, "Optional local DRGN-AI parity reference")
    def test_all_round_scores_and_candidate_ids(self):
        actual, expected = make_search(TracedSearch), make_search(HartleyReference)
        batch, structures, _, _ = fixture(actual)
        np.random.seed(7)
        result = actual.opt_theta_trans(batch, structures)
        np.random.seed(7)
        truth = expected.opt_theta_trans(batch, structures)
        self.assertEqual(len(actual.trace), 4)
        for (scores, ids), (reference_scores, reference_ids) in zip(actual.trace, expected.trace):
            torch.testing.assert_close(scores, reference_scores, atol=.08, rtol=2e-5)
            torch.testing.assert_close(ids, reference_ids, atol=0, rtol=0)
        for a,b in zip(result, truth):
            torch.testing.assert_close(a,b)

    @unittest.skipUnless(HAS_REFERENCE, "Optional local DRGN-AI parity reference")
    def test_astigmatic_ctf_all_rounds(self):
        actual, expected = make_search(TracedSearch), make_search(HartleyReference)
        actual.ctf = expected.ctf = CTFCryoDRGN(32, 1.).to(DEVICE)
        batch, structures, _, _ = fixture(actual)
        batch.update(defocusU=torch.tensor([.8, 1., 1.2], device=DEVICE)[:,None,None],
                     defocusV=torch.tensor([.9, 1.3, 1.1], device=DEVICE)[:,None,None],
                     angleAstigmatism=torch.tensor([.2, .7, 1.1], device=DEVICE)[:,None,None])
        batch['proj'] = fourier_to_primal_2d(actual._apply_ctf(batch, primal_to_fourier_2d(batch['proj']))).real
        for search in [actual, expected]:
            np.random.seed(7)
            search.opt_theta_trans(batch, structures)
        for (scores, ids), (ref_scores, ref_ids) in zip(actual.trace, expected.trace):
            torch.testing.assert_close(scores, ref_scores, atol=.08, rtol=3e-5)
            torch.testing.assert_close(ids, ref_ids, atol=0, rtol=0)

    def test_predict_pose_cache_and_head_gradients(self):
        calls = []
        def search(batch, structure):
            calls.append(batch['idx'].clone())
            return torch.eye(3, device=DEVICE).expand(len(structure),-1,-1), structure.new_ones(len(structure),2)
        task = SimpleNamespace(pose_search=SimpleNamespace(opt_theta_trans=search),
            predicted_rots=torch.eye(3,device=DEVICE).repeat(5,1,1),
            predicted_trans=torch.zeros(5,2,device=DEVICE),
            predicted_pose_valid=torch.zeros(5,dtype=torch.bool,device=DEVICE),
            gmm_sigmas=torch.ones(1,device=DEVICE),gmm_amps=torch.ones(1,device=DEVICE),
            is_in_pose_search_step=True,use_pose_head=True,
            pose_head=PoseHeadSmallMLP(4).to(DEVICE))
        batch = dict(idx=torch.tensor([1,3],device=DEVICE),proj=torch.zeros(2,1,32,32,device=DEVICE))
        structure=torch.zeros(2,16,3,device=DEVICE)
        mu=torch.randn(2,4,device=DEVICE,requires_grad=True)
        rot, trans = CryoEMTask.predict_pose(task,batch,structure,mu)
        self.assertTrue(task.predicted_pose_valid[batch['idx']].all())
        task.is_in_pose_search_step=False
        rot, trans = CryoEMTask.predict_pose(task,batch,structure,mu)
        self.assertEqual(len(calls),1)
        rot[:,0,1].sum().backward()
        self.assertGreater(float(task.pose_head.net[-1].weight.grad.abs().sum()),0)
        batch['idx']=torch.tensor([3,4],device=DEVICE)
        CryoEMTask.predict_pose(task,batch,structure,mu)
        torch.testing.assert_close(calls[-1],batch['idx'][1:])
        self.assertEqual(len(calls),2)

    def test_known_pose_and_chunk_invariance(self):
        search = make_search()
        batch, structures, truth, shifts = fixture(search)
        np.random.seed(7)
        rotations, translations = search.opt_theta_trans(batch, structures)
        errors = torch.rad2deg(torch.acos(((rotations @ truth.transpose(-1,-2)).diagonal(dim1=-2,dim2=-1).sum(-1).sub(1)/2).clamp(-1,1)))
        print('NOISELESS', errors.tolist(), 'SHIFT', (translations-shifts).norm(dim=-1).tolist(), flush=True)
        self.assertLess(float(errors.max()), 5.)
        self.assertLess(float((translations-shifts).norm(dim=-1).max()), .6)
        search.particle_chunk_size = 1
        np.random.seed(7)
        single = search.opt_theta_trans(batch, structures)
        torch.testing.assert_close(rotations, single[0])
        torch.testing.assert_close(translations, single[1])

    def test_observed_fft_count_and_autocast(self):
        search = make_search(hps_particle_chunk_size=2)
        batch, structures, _, _ = fixture(search)
        observed = []
        def counted_fft(images):
            if images.ndim == 3:
                observed.append(len(images))
            return primal_to_fourier_2d(images)
        with patch('train_atom_with_pose.primal_to_fourier_2d', side_effect=counted_fft):
            with torch.autocast(device_type=DEVICE, enabled=DEVICE == 'cuda'):
                rotations, shifts = search.opt_theta_trans(batch, structures)
        self.assertEqual(sum(observed),len(structures))
        self.assertEqual(rotations.dtype,torch.float32)
        self.assertTrue(torch.isfinite(shifts).all())

    def test_zero_iterations_and_zero_extent(self):
        search = make_search(n_iter=0, t_extent=0.)
        batch, structures, _, _ = fixture(search)
        rot, trans = search.opt_theta_trans(batch, structures)
        self.assertEqual(rot.shape, (3,3,3))
        torch.testing.assert_close(trans, torch.zeros_like(trans))
        torch.testing.assert_close(rot @ rot.transpose(-1,-2), torch.eye(3,device=DEVICE).expand(3,-1,-1), atol=1e-5, rtol=1e-5)


if __name__ == '__main__':
    unittest.main(verbosity=2)
