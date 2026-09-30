"""Indexed differentiable poses and portable SO(3) subdivision."""
import numpy as np
import torch
from torch import nn
from cryodyna.utils import so3_grid, lie_tools


class PoseTable(nn.Module):
    """Particle IDs -> proper rotations [B,3,3], YX shifts [B,2] in pixels.

    The first two matrix rows parameterize SO(3); Gram-Schmidt keeps every
    forward rotation orthogonal. Initialization and validity are checkpointed.
    """
    def __init__(self, count, translations=True):
        super().__init__()
        self.rotation = nn.Parameter(torch.eye(3)[:2].reshape(1, 6).repeat(count, 1))
        self.translation = nn.Parameter(torch.zeros(count, 2), requires_grad=translations)
        self.register_buffer("valid", torch.zeros(count, dtype=torch.bool))

    @torch.no_grad()
    def initialize(self, indices, rotations, translations):
        self.rotation[indices] = rotations[:, :2].reshape(-1, 6).to(self.rotation)
        self.translation[indices] = translations.reshape(-1, 2).to(self.translation)
        self.valid[indices] = True

    def forward(self, indices):
        if not self.valid[indices].all():
            raise RuntimeError("Initialize the requested pose-table rows before lookup")
        a, b = self.rotation[indices].reshape(-1, 2, 3).unbind(1)
        e1 = torch.nn.functional.normalize(a, dim=-1)
        e2 = torch.nn.functional.normalize(b - (e1 * b).sum(-1, keepdim=True) * e1, dim=-1)
        return torch.stack((e1, e2, torch.cross(e1, e2, dim=-1)), 1), self.translation[indices]


def subdivide(quat, q_ind, cur_res, device):
    """Generate 16 Hopf children and retain the eight nearest to each parent.

    Uses CryoDyna's grid conventions, including quaternion sign equivalence.
    Returns quaternions [N,8,4], grid IDs [N,8,2], matrices [N*8,3,3].
    """
    ids = np.asarray(q_ind)
    n = len(ids)
    s2 = ids[:, :1] * 4 + np.arange(4)
    theta, phi = so3_grid.pix2ang(2 ** (cur_res + 1), s2.reshape(-1), nest=True)
    theta, phi = theta.reshape(n, 4), phi.reshape(n, 4)
    s1 = ids[:, 1:] * 2 - 1 + np.arange(4)
    count = 6 * 2 ** (cur_res + 1)
    s1[s1 < 0] += count
    psi = (s1 + .5) * (2 * np.pi / count)
    theta = np.repeat(theta, 4, axis=1)
    phi = np.repeat(phi, 4, axis=1)
    psi = np.tile(psi, (1, 4))
    children = np.stack((np.cos(theta/2)*np.cos(psi/2),
                         np.cos(theta/2)*np.sin(psi/2),
                         np.sin(theta/2)*np.cos(phi+psi/2),
                         np.sin(theta/2)*np.sin(phi+psi/2)), -1)
    children = torch.as_tensor(children, dtype=torch.float32, device=device)
    distance = torch.minimum((children-quat[:, None]).square().sum(-1),
                             (children+quat[:, None]).square().sum(-1))
    nearest = distance.argsort(-1)[:, :8]
    rows = torch.arange(n, device=device)[:, None]
    result = children[rows, nearest]
    grid = np.stack((np.repeat(s2, 4, axis=1), np.tile(s1, (1, 4))), -1)
    grid = grid[np.arange(n)[:, None], nearest.cpu().numpy()]
    return result, grid, lie_tools.quaternions_to_SO3(result.reshape(-1, 4))
