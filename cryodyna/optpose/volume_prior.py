"""Fixed-density pose fitting for the homogeneous 80S control.

This is an explicit volume-prior variant of Optpose. It reuses PoseTable and
Hopf-grid subdivision; it does not represent a learned atomic/deformation model.
"""
from pathlib import Path
import numpy as np
import mrcfile
import torch
from torch import nn
from torch.nn import functional as F


class FourierVolume(nn.Module):
    def __init__(self, path, radius, ctf_params, image_side=128):
        super().__init__()
        with mrcfile.open(str(path)) as m:
            vol = torch.from_numpy(m.data.copy()).float()
            apix = float(m.voxel_size.x)
        if vol.shape != (image_side,) * 3 or abs(apix-float(ctf_params[1])) > 1e-4:
            raise ValueError("Reference volume and image lattice must match")
        n = image_side
        spectrum = torch.fft.fftshift(torch.fft.fftn(torch.fft.ifftshift(vol)))
        spectrum /= spectrum.abs().max()
        self.register_buffer("volume", torch.view_as_real(spectrum).permute(3, 0, 1, 2)[None].contiguous())
        axis = torch.arange(n).float()-n//2
        yy, xx = torch.meshgrid(axis, axis, indexing="ij")
        mask = (xx.square()+yy.square() <= radius**2) & (xx.square()+yy.square() > 0)
        self.register_buffer("mask", mask)
        coordinates = torch.stack([xx[mask], yy[mask], torch.zeros_like(xx[mask])], -1)
        self.register_buffer("coordinates", coordinates)
        self.n = n
        # Same physical CTF as the input metadata, using the public CryoDyna implementation.
        from cryodyna.utils.ctf import compute_ctf
        cp = torch.as_tensor(ctf_params, dtype=torch.float32)
        filt = compute_ctf(coordinates[:, :2]/(n*apix), *cp[2:])
        # The deposited 80S images retain the native sign (DRGN-AI inverts them).
        self.register_buffer("ctf", filt)

    def forward(self, rotations, translations_yx=None):
        xyz = self.coordinates[None] @ rotations
        # Even-sized FFT DC lies at index n/2; grid_sample uses inclusive corners.
        grid = (2*(xyz+self.n/2)/(self.n-1)-1).reshape(1, 1, -1, self.coordinates.shape[0], 3)
        values = F.grid_sample(self.volume, grid, align_corners=True, mode="bilinear")
        values = values[0, :, 0].permute(1, 2, 0).contiguous()
        spectrum = torch.view_as_complex(values) * self.ctf
        if translations_yx is not None:
            phase = -2*torch.pi*(translations_yx @ self.coordinates[:, [1, 0]].T)/self.n
            spectrum = spectrum * torch.exp(1j*phase)
        return spectrum

    def observations(self, images):
        ft = torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(images, dim=(-2, -1))), dim=(-2, -1))
        return ft[:, self.mask]


def normalized_features(spectra):
    real = torch.view_as_real(spectra).flatten(1)
    return F.normalize(real, dim=1)


def correlation_loss(prediction, observed):
    return 1-(normalized_features(prediction)*normalized_features(observed)).sum(1)
