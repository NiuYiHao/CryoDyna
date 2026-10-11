"""Pose-consistent correlation candidate; legacy baseline stays unchanged."""
import torch


def scaled_masked_cosine_loss(predicted, observed, mask=None):
    """Negative cosine, scaled to the legacy zero-mean unbiased-STD convention.

    Unlike an uncentered dot divided by centered standard deviations, this
    objective is stationary at an exact positive-scale match. Pixel scaling
    retains the old image-term magnitude for zero-mean predictions/targets.
    """
    if mask is not None:
        predicted, observed = mask(predicted), mask(observed)
        pixel_count = mask.num_masked
    else:
        pixel_count = predicted.shape[-2] * predicted.shape[-1]
    p, t = predicted.flatten(-2), observed.flatten(-2)
    denominator = (p.norm(dim=-1) * t.norm(dim=-1)).clamp_min(1e-12)
    cosine = (p * t).sum(-1) / denominator
    return -cosine.sum(-1).mean() * (p.shape[-1] - 1) / pixel_count
