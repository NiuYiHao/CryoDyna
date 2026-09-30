"""Metrics use full rotations, Angstrom coordinates and explicit gauge fitting."""
import numpy as np
from scipy.spatial.transform import Rotation


def validate_rotations(rotations):
    r = np.asarray(rotations, dtype=np.float64)
    if r.ndim != 3 or r.shape[1:] != (3, 3) or not np.isfinite(r).all():
        raise ValueError("Expected finite rotations [N,3,3]")
    if np.max(np.abs(r @ r.transpose(0, 2, 1)-np.eye(3))) > 2e-3 or np.max(np.abs(np.linalg.det(r)-1)) > 2e-3:
        raise ValueError("Rotations must belong to SO(3)")
    return r


def angular_error(pred, truth):
    p, t = validate_rotations(pred), validate_rotations(truth)
    if p.shape != t.shape:
        raise ValueError("Pose arrays must have matching particle counts")
    relative = p @ t.transpose(0, 2, 1)
    return np.degrees(np.arccos(np.clip((np.trace(relative, axis1=1, axis2=2)-1)/2, -1, 1)))


def star_rotations(frame):
    return Rotation.from_euler("ZYZ", -frame[["rlnAnglePsi", "rlnAngleTilt", "rlnAngleRot"]].to_numpy(float), degrees=True).as_matrix()


def perturb_rotations(rotations, angle, seed):
    if not np.isfinite(angle) or not 0 <= angle <= 180:
        raise ValueError("Perturbation angles must be in [0,180] degrees")
    axes = np.random.default_rng(seed).normal(size=(len(rotations), 3))
    axes /= np.linalg.norm(axes, axis=-1, keepdims=True)
    return Rotation.from_rotvec(axes * np.radians(angle)).as_matrix() @ rotations


def fit_gauge(pred, truth, fit_indices=None):
    """One object-frame gauge and one hand for the entire dataset.

    Optional calibration IDs keep the alignment fit separate from evaluation.
    No particle-specific alignment is used for the pose metric.
    """
    p, t = validate_rotations(pred), validate_rotations(truth)
    ids = np.arange(len(p)) if fit_indices is None else np.asarray(fit_indices)
    choices = []
    for mirror in (False, True):
        sign = -1. if mirror else 1.
        s = np.diag([1., 1., sign])
        source = s @ p
        u, _, vh = np.linalg.svd(np.einsum("nji,njk->ik", source[ids], t[ids]))
        correction = np.eye(3)
        correction[-1, -1] = sign * np.linalg.det(u @ vh)
        q = u @ correction @ vh
        aligned = source @ q
        score = float(np.sum((aligned[ids]-t[ids])**2))
        choices.append((score, aligned, {"mirror": mirror, "right_matrix": q.tolist(), "fit_count": len(ids)}))
    _, aligned, metadata = min(choices, key=lambda x: x[0])
    return aligned, metadata


def kabsch_rmsd(pred, truth):
    p, t = np.asarray(pred, dtype=np.float64), np.asarray(truth, dtype=np.float64)
    if p.shape != t.shape or p.ndim != 3 or p.shape[-1] != 3:
        raise ValueError("Structures must have matching [N,atoms,3] shapes")
    p, t = p-p.mean(1, keepdims=True), t-t.mean(1, keepdims=True)
    u, _, vh = np.linalg.svd(p.transpose(0, 2, 1) @ t)
    correction = np.tile(np.eye(3), (len(p), 1, 1))
    correction[:, -1, -1] = np.linalg.det(u @ vh)
    return np.sqrt(np.mean(np.sum((p @ (u @ correction @ vh)-t)**2, axis=-1), axis=-1))
