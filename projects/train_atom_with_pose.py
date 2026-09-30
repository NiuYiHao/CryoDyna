"""
原版依赖pose。在原版 ``train_atom`` 上增加可学习的旋转姿态修正。

数据流：粒子图像 -> encoder 得到 z ->
    1. 原 decoder 预测结构形变量；
    2. pose head 通过有界的两层 GELU MLP，根据 z 预测三维轴角修正量；
随后用修正后的旋转矩阵投影结构，并沿用原版 loss 训练。

开发阶段：加入HPS(Hierarchical pose search)模块，更新pose。
"""



from abc import ABC, abstractmethod
import collections
import os.path as osp
from typing import Optional

import biotite.structure as struc
import lightning.pytorch as pl
from lightning.pytorch.utilities import rank_zero_only
from torch.utils.data import DataLoader
from cryodyna.utils.dataio import StarfileDataSet, StarfileDatasetConfig, Mask
from torch import optim
from cryodyna.utils.polymer import Polymer
import numpy as np
import torch
from scipy.spatial.distance import cdist
from cryodyna.utils.pdb_tools import bt_save_pdb, extract_sec_ids_merge_small_blocks, build_metagraph_knn_from_centroids, get_kplus_neighbor_metanodes
from copy import deepcopy
from cryodyna.utils.losses import calc_cor_loss
from cryodyna.utils.fft_utils import primal_to_fourier_2d, fourier_to_primal_2d
from cryodyna.utils.misc import log_to_current, pl_init_exp, pretty_dict
from miscs import (
    VAE,
    calc_clash_loss,
    calc_pair_dist_loss,
    infer_ctf_params_from_config,
    low_pass_mask2d,
)
from cryodyna.gmm.deformer import E3Deformer
from mmengine import Config
from mmengine import mkdir_or_exist
import einops
from cryodyna.gmm.gmm import EMAN2Grid, batch_projection, Gaussian
from cryodyna.utils.ctf_utils import CTFCryoDRGN
from cryodyna.utils.dist_loss import (
    DistLoss,
    calc_dist_by_pair_indices,
    filter_same_chain_pairs,
    find_continuous_pairs,
    find_quaint_cutoff_pairs,
    find_range_cutoff_pairs,
    remove_duplicate_pairs,
)
from cryodyna.utils.latent_space_utils import (
    cluster_kmeans,
    get_nearest_point,
    get_pc_traj,
    run_pca,
    run_umap,
)
from cryodyna.utils.pl_utils import (
    filter_outputs_by_indices,
    get_1st_unique_indices,
    merge_step_outputs,
    squeeze_dict_outputs_1st_dim,
)
from cryodyna.utils.vis_utils import plot_z_dist, save_tensor_image

from cryodyna.utils.transforms import SpatialGridTranslate
from cryodyna.utils.rotation_conversion import axis_angle_to_matrix
from cryodyna.utils import so3_grid
from cryodyna.utils import lie_tools
from cryodyna.utils import shift_grid
from cryodyna.optpose.pose import PoseTable, subdivide


TASK_NAME = "atom"


class PoseHeadMLP(torch.nn.Module):
    """Predict a bounded local axis-angle correction from the latent code."""

    def __init__(
        self,
        in_dim: int,
        hidden_dims: tuple[int, ...] = (128, 128),
        max_angle_deg: float = 45.0,
    ) -> None:
        super().__init__()
        if len(hidden_dims) == 0:
            raise ValueError("hidden_dims must contain at least one layer")

        layers: list[torch.nn.Module] = [torch.nn.LayerNorm(in_dim)]
        prev_dim = in_dim
        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    torch.nn.Linear(prev_dim, hidden_dim),
                    torch.nn.GELU(),
                ]
            )
            prev_dim = hidden_dim
        layers.append(torch.nn.Linear(prev_dim, 3))
        self.net = torch.nn.Sequential(*layers)
        self.theta_max = float(np.deg2rad(max_angle_deg))

        # Keep the original zero-correction starting point while preserving
        # a non-zero gradient through the bounded output parameterisation.
        torch.nn.init.zeros_(self.net[-1].weight)
        torch.nn.init.zeros_(self.net[-1].bias)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        raw = self.net(z)
        # The bound is applied component-wise; the result remains in radians.
        return self.theta_max * torch.tanh(raw)


class PoseHeadSmallMLP(torch.nn.Module):
    """Predict a bounded local axis-angle correction with one hidden layer."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 64,
        max_angle_deg: float = 45.0,
    ) -> None:
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.LayerNorm(in_dim),
            torch.nn.Linear(in_dim, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_dim, 3),
        )
        self.theta_max = float(np.deg2rad(max_angle_deg))
        torch.nn.init.zeros_(self.net[-1].weight)
        torch.nn.init.zeros_(self.net[-1].bias)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.theta_max * torch.tanh(self.net(z))

class HierarchicalPoseSearch:
    """Hierarchical SO(3) and translation grid search following drgn-ai.

    生成SO3和shift grid → 在低频圆盘内打分 → 每图保留最佳候选。
    迭代细化：每个旋转生成8个子候选 → 缩小平移网格 → 扩大频率圆盘
    → 重新评分并筛选。

    Even N-pixel images use a symmetric (N+1) Fourier lattice, excluding DC.
    Scores retain the unnormalized FFT scale of GMM projections and observations.
    Coarse CTF-before-in-plane interpolation follows the local drgn-ai baseline.
    The training-facing entry point is CryoEMTask.predict_pose; shifts are YX pixels.
    """
    def __init__(self, configs, gmm_sigmas, gmm_amps, grid, ctf, translator):
        self.gmm_sigmas = gmm_sigmas
        self.gmm_amps = gmm_amps
        self.grid = grid
        self.ctf = ctf
        self.translator = translator
        self.chunk_size = int(configs.get("hps_chunk_size", 32))
        self.particle_chunk_size = int(configs.get("hps_particle_chunk_size", 4))
        self.ignore_dc = bool(configs.get("hps_ignore_dc", True))
        self.score_type = configs.get("hps_score", "l2")
        if self.score_type not in ("l2", "correlation"):
            raise ValueError("hps_score must be l2 or correlation")
        self._frequency_cache = {}
        self._masked_frequencies = {}
        if min(self.chunk_size, self.particle_chunk_size) < 1:
            raise ValueError("HPS chunk sizes must be positive")
        # SO3 grid层数
        self.base_healpy = int(configs.get("base_healpy", 2))
        # 迭代细化轮数
        self.niter = int(configs.get("n_iter", 4))
        # 中间层保留的候选pose个数
        self.nkeptposes = int(configs.get("n_kept_poses", 8))
        # 频域半径范围
        self.l_min = int(configs.get("l_start", 12))
        self.l_max = int(configs.get("l_end", 32))
        # shift grid
        self.t_extent = float(configs.get("t_extent", 20.0))
        self.t_n_grid = int(configs.get("t_n_grid", 7))
        self.t_xshift = float(configs.get("t_x_shift", 0.0))
        self.t_yshift = float(configs.get("t_y_shift", 0.0))
        shifts = shift_grid.base_shift_grid(
            self.base_healpy - 1,
            self.t_extent,
            self.t_n_grid,
            xshift=self.t_xshift,
            yshift=self.t_yshift,
        )
        self.base_shifts = torch.from_numpy(shifts).float().flip(-1)  # XY 网格转为返回接口的 YX 像素顺序
        # so3 grid
        self.so3_base_quat = torch.from_numpy(so3_grid.grid_SO3(self.base_healpy))
        self.base_rot = lie_tools.quaternions_to_SO3(torch.from_numpy(so3_grid.s2_grid_SO3(self.base_healpy)))
        self.base_inplane = torch.from_numpy(so3_grid.grid_s1(self.base_healpy))
        if self.niter < 0 or self.nkeptposes < 1 or not 0 < self.l_min <= self.l_max:
            raise ValueError("HPS requires n_iter >= 0, n_kept_poses >= 1 and 0 < l_start <= l_end")

    def _shared_projection(self, pred_struc, rot_mats):
        pred_images = batch_projection(
            gauss=Gaussian(
                mus=pred_struc,
                sigmas=self.gmm_sigmas.to(pred_struc).unsqueeze(0),
                amplitudes=self.gmm_amps.to(pred_struc).unsqueeze(0)),
            rot_mats=rot_mats,
            line_grid=self.grid.line(),
        )
        return einops.rearrange(pred_images, "b y x -> b 1 y x")

    def _apply_ctf(self, batch, f_proj):
        """在实验图像坐标系中给预测频谱施加 CTF。"""
        ctf_params = {
            key: batch[key]
            for key in ("defocusU", "defocusV", "angleAstigmatism")
            if key in batch
        }
        return self.ctf(f_proj, batch["idx"], ctf_params=ctf_params,
                        mode="gt", frequency_marcher=None)

    @torch.no_grad()
    def _predict_hps_projections(
        self, batch, pred_struc, rotations, image_indices=None, apply_ctf=True
    ):
        """按候选旋转生成含 CTF 的预测频谱。"""
        if image_indices is None:
            image_indices = torch.arange(
                pred_struc.shape[0], device=pred_struc.device
            )
        if rotations.ndim == 3:
            rotations = rotations.unsqueeze(0).expand(len(image_indices), -1, -1, -1)
        group_count, rotation_count = rotations.shape[:2]
        flat_rotations = rotations.reshape(-1, 3, 3)
        flat_image_indices = image_indices.repeat_interleave(rotation_count)
        spectra = []
        for start in range(0, len(flat_rotations), self.chunk_size):
            chunk_indices = flat_image_indices[start:start + self.chunk_size]
            projections = self._shared_projection(
                pred_struc[chunk_indices], flat_rotations[start:start + self.chunk_size]
            )
            projections = primal_to_fourier_2d(projections)
            if apply_ctf:
                projections = projections * batch["_hps_ctf"][chunk_indices]
            projections = self._symmetric_spectrum(projections)
            spectra.append(projections.squeeze(1))
        spectra = torch.cat(spectra, dim=0)
        return spectra.reshape(group_count, rotation_count, *spectra.shape[-2:])

    def get_l(self, step: int, res: int) -> int:
        """当前轮次的频域半径"""
        fraction = step / self.niter if self.niter > 0 else 1.0
        radius = self.l_min + int(fraction * (self.l_max - self.l_min))
        return min(radius, res // 2)

    @staticmethod
    def _symmetric_spectrum(spectra):
        """Append periodic Nyquist endpoints for the upstream odd lattice."""
        if spectra.shape[-1] % 2 == 0:
            spectra = torch.cat((spectra, spectra[..., :1, :]), dim=-2)
            spectra = torch.cat((spectra, spectra[..., :1]), dim=-1)
        return spectra

    def _frequency_grid(self, res, radius, device):
        key = (res, radius, str(device))
        if key not in self._frequency_cache:
            axis = torch.arange(res, device=device) - res // 2
            yy, xx = torch.meshgrid(axis, axis, indexing="ij")
            mask = xx.square() + yy.square() <= radius ** 2
            if self.ignore_dc:
                mask[res // 2, res // 2] = False
            freqs_yx = torch.stack((yy[mask], xx[mask]), -1).float() / (res - 1)
            self._frequency_cache[key] = mask, freqs_yx
            self._masked_frequencies[id(mask)] = freqs_yx
        return self._frequency_cache[key]

    def get_frequency_mask(self, res, radius, device):
        return self._frequency_grid(res, radius, device)[0]

    def translate_images(self, images_ft, shifts, mask):
        """Complex spectra [B,D,D] and pixel YX shifts → [B,T,P]."""
        freqs = self._masked_frequencies[id(mask)]
        phase = -2 * torch.pi * (shifts.to(images_ft.real) @ freqs.T)
        return images_ft[:, None, mask] * torch.polar(torch.ones_like(phase), phase)

    @torch.no_grad()
    def compute_err(self, images_ft, projections_ft):
        """Hartley-equivalent score [B,T,Q] on symmetric frequencies."""
        dots = (images_ft @ projections_ft.conj().transpose(-1, -2)).real
        if self.score_type == "correlation":
            image_norm = images_ft.abs().square().sum(-1).sqrt().clamp_min(1e-12)
            projection_norm = projections_ft.abs().square().sum(-1).sqrt().clamp_min(1e-12)
            return -dots / (image_norm[:, :, None] * projection_norm[:, None, :])
        norm = projections_ft.abs().square().sum(-1) / 2
        return norm[:, None, :] - dots

    @torch.no_grad()
    def eval_grid(self, images, rotations, shifts, mask, batch, pred_struc):
        """输入对称复数频谱 [B,D,D]、YX 平移，返回评分 [B,T,Q]。"""
        shifted_images = self.translate_images(images, shifts, mask)
        losses = []
        for start in range(0, len(rotations), self.chunk_size):
            projections_ft = self._predict_hps_projections(
                batch, pred_struc, rotations[start:start + self.chunk_size]
            )
            losses.append(self.compute_err(shifted_images, projections_ft[..., mask]))
        return torch.cat(losses, dim=-1)

    def _rotate_base_spectra(self, spectra, angles, mask):
        """[B,S,H,W] → [B,S*A,P]；方向优先、面内角次之，对齐 SO(3) 网格编号。"""
        batch_size, directions, res, _ = spectra.shape
        axis = torch.arange(res, device=spectra.device, dtype=spectra.real.dtype) - res // 2
        yy, xx = torch.meshgrid(axis, axis, indexing="ij")
        coords = torch.stack((xx[mask], yy[mask]), dim=-1)
        omega = torch.zeros((len(angles), 3), device=spectra.device)
        omega[:, 2] = angles
        # R候选 = Rz(angle) @ R基准；对应频谱采样坐标为 k @ Rz(angle)。
        sample_coords = coords @ axis_angle_to_matrix(omega)[:, :2, :2]
        # Match drgn-ai interpolate exactly, including align_corners=False.
        grid = sample_coords * (2 / (res - 1))
        source = spectra[..., mask].reshape(batch_size * directions, -1)
        channels = torch.view_as_real(spectra * mask).permute(0, 1, 4, 2, 3)
        channels = channels.reshape(batch_size * directions, 2, res, res)
        sampled = torch.nn.functional.grid_sample(
            channels, grid[None].expand(len(channels), -1, -1, -1), align_corners=False
        )
        rotated = torch.complex(sampled[:, 0], sampled[:, 1])
        # 对齐 drgn-ai rotate_images：补偿插值造成的频谱幅度衰减。
        # Hartley statistics preserve the upstream amplitude compensation.
        source_std = (source.real - source.imag).std(-1)
        rotated_std = (rotated.real - rotated.imag).std(-1)
        scale = source_std[:, None] / rotated_std.clamp_min(1e-12)
        rotated = rotated * scale[..., None]
        return rotated.reshape(batch_size, directions * len(angles), -1)

    @torch.no_grad()
    def _eval_base_grid(self, images, shifts, mask, batch, pred_struc, start_angle):
        """粗搜只投影 S2 方向，复用二维旋转生成面内候选；返回 [B,T,S*A]。"""
        angles = self.base_inplane.to(images.real)
        omega = images.real.new_zeros(3)
        omega[2] = start_angle
        rotations = axis_angle_to_matrix(omega) @ self.base_rot.to(images.real)
        shifted_images = self.translate_images(images, shifts, mask)
        losses = []
        for start in range(0, len(rotations), self.chunk_size):
            spectra = self._predict_hps_projections(
                batch, pred_struc, rotations[start:start + self.chunk_size]
            )
            candidates = self._rotate_base_spectra(spectra, angles - start_angle, mask)
            losses.append(self.compute_err(shifted_images, candidates))
        return torch.cat(losses, dim=-1)

    def keep_matrix(self, loss, batch_size, max_poses):
        """loss：粗搜索 [B,T,Q]，细化[B*上一轮保留数,T,8]；
        max_poses 为本轮每张图保留数。
        返回行号、平移编号、旋转编号，三个张量均为 [B*max_poses]。
        """
        # 筛选误差最小的平移id
        best_loss, best_trans = loss.min(dim=1)
        # 将每张图的候选误差整理到同一行
        # 粗搜索：[B, Q] → [B, Q]；细化：[B * 保留数, 8] → [B, 保留数 * 8]
        errors_per_image = best_loss.reshape(batch_size, -1)
        selected_indices = errors_per_image.topk(max_poses, dim=1, largest=False).indices
        candidates_per_image = errors_per_image.shape[1]
        # 每张图的候选起始编号，例如每图 64 个时为 0、64、128……
        image_offsets = torch.arange(batch_size, device=loss.device) * candidates_per_image
        flat_indices = (selected_indices + image_offsets[:, None]).reshape(-1)
        parent_indices = flat_indices // loss.shape[-1]
        rotation_indices = flat_indices % loss.shape[-1]
        translation_indices = best_trans[parent_indices, rotation_indices]
        return parent_indices, translation_indices, rotation_indices

    @torch.no_grad()
    def opt_theta_trans(self, batch, pred_struc):
        """Bounded particle batches; one observed FFT and CTF per particle."""
        results = []
        batch_size = pred_struc.shape[0]
        if batch["proj"].shape[-1] % 2:
            raise ValueError("HPS uses even-sized images and an odd symmetric Fourier lattice")
        start_angle = self.base_inplane[np.random.randint(len(self.base_inplane))]
        with torch.autocast(device_type=pred_struc.device.type, enabled=False):
            for start in range(0, batch_size, self.particle_chunk_size):
                stop = start + self.particle_chunk_size
                particle = {
                    key: value[start:stop] if isinstance(value, torch.Tensor)
                    and value.ndim > 0 and value.shape[0] == batch_size else value
                    for key, value in batch.items()
                }
                particle["proj"] = particle["proj"].float()
                particle["_hps_ctf"] = self._apply_ctf(
                    particle, torch.ones_like(particle["proj"], dtype=torch.complex64)
                )
                results.append(self._search_batch(particle, pred_struc[start:stop].float(), start_angle))
        return tuple(torch.cat(items, dim=0) for items in zip(*results))

    @torch.no_grad()
    def _search_batch(self, batch, pred_struc, start_angle):
        """粗搜索 → 多轮细化；输入实域图，返回 YX 平移。"""
        images = self._symmetric_spectrum(primal_to_fourier_2d(batch["proj"].squeeze(1)))
        # 读取批量大小、图像尺寸和计算设备
        batch_size = images.shape[0]
        res = images.shape[-1]
        device = images.device
        # 准备粗搜索的旋转和平移候选
        quaternions = self.so3_base_quat.to(device)
        shifts = self.base_shifts.to(device)
        # 生成粗搜索使用的低频圆盘
        radius = self.get_l(0, res)
        mask = self.get_frequency_mask(res, radius, device)
        # 计算候选误差，每张图保留最好的几个姿态
        loss = self._eval_base_grid(images, shifts, mask, batch, pred_struc, start_angle)
        keep_count = self.nkeptposes if self.niter > 0 else 1
        image_indices, translation_indices, rotation_indices = self.keep_matrix(
            loss, batch_size, keep_count
        )
        # 取出保留的旋转、网格编号和平移，作为细化起点
        quaternions = quaternions[rotation_indices]
        grid_indices = so3_grid.get_base_ind(
            rotation_indices.cpu().numpy(), self.base_healpy
        )
        translations = shifts[translation_indices]
        for step in range(1, self.niter + 1):
            # 每个保留旋转生成 8 个更细的子旋转
            quaternions, grid_indices, rotations = subdivide(
                quaternions, grid_indices, self.base_healpy + step - 1, device
            )
            # 平移网格缩小一半，并移到各候选平移附近
            shifts = shifts / 2
            candidate_translations = translations[:, None, :] + shifts[None, :, :]
            # 扩大频率圆盘，逐渐加入高频细节
            radius = self.get_l(step, res)
            mask = self.get_frequency_mask(res, radius, device)
            # 为每组子旋转匹配实验图，并生成对应预测投影
            rotations = rotations.reshape(-1, 8, 3, 3)
            losses = []
            for start in range(0, len(image_indices), self.particle_chunk_size):
                stop = start + self.particle_chunk_size
                indices = image_indices[start:stop]
                projections_ft = self._predict_hps_projections(
                    batch, pred_struc, rotations[start:stop], indices
                )
                shifted_images = self.translate_images(
                    images[indices], candidate_translations[start:stop], mask
                )
                losses.append(self.compute_err(shifted_images, projections_ft[..., mask]))
            loss = torch.cat(losses)
            # 中间轮保留多个候选，最后一轮选出最佳姿态
            keep_count = self.nkeptposes if step < self.niter else 1
            parent_indices, translation_indices, rotation_indices = self.keep_matrix(
                loss, batch_size, keep_count
            )
            quaternions = quaternions[parent_indices, rotation_indices]
            grid_indices = grid_indices[
                parent_indices.cpu().numpy(), rotation_indices.cpu().numpy()
            ]
            translations = candidate_translations[parent_indices, translation_indices]
            image_indices = image_indices[parent_indices]
        best_rotations = lie_tools.quaternions_to_SO3(quaternions)
        return best_rotations, translations


class AbstractCryoEMTask(pl.LightningModule, ABC):
    """CryoEM 推理流水线必须实现的计算接口。"""

    @abstractmethod
    def _shared_forward(
        self,
        images: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """由粒子图像预测结构形变量和潜变量。"""
        raise NotImplementedError

    @abstractmethod
    def _shared_projection(
        self,
        pred_struc: torch.Tensor,
        rot_mats: torch.Tensor,
    ) -> torch.Tensor:
        """将批量三维结构按旋转矩阵投影为二维图像。"""
        raise NotImplementedError

    @abstractmethod
    def _apply_ctf(
        self,
        batch: dict[str, torch.Tensor],
        real_proj: torch.Tensor,
        freq_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """对预测投影应用 CTF。"""
        raise NotImplementedError

    @abstractmethod
    def _shared_infer(
        self,
        batch: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """组织推理流程；返回 gt 图、预测图、预测结构和潜变量。"""
        raise NotImplementedError

    @abstractmethod
    def get_batch_pose(
        self,
        batch: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """从 batch 读取旋转矩阵和平移矩阵。"""
        raise NotImplementedError

    @abstractmethod
    def _prepare_structural_loss_dependencies(self, meta: Polymer) -> None:
        """从参考结构建立各结构约束 loss 的固定依赖。"""
        raise NotImplementedError

    @abstractmethod
    def _calculate_connect_loss(self, pred_struc: torch.Tensor) -> torch.Tensor:
        """计算加权的相邻原子连接 loss。"""
        raise NotImplementedError

    @abstractmethod
    def _calculate_sse_loss(self, pred_struc: torch.Tensor) -> torch.Tensor:
        """计算加权的二级结构保持 loss。"""
        raise NotImplementedError

    @abstractmethod
    def _calculate_dist_loss(self, pred_struc: torch.Tensor) -> torch.Tensor:
        """计算加权的非相邻原子距离保持 loss。"""
        raise NotImplementedError

    @abstractmethod
    def _calculate_clash_loss(self, pred_struc: torch.Tensor) -> torch.Tensor:
        """计算加权的原子碰撞 loss。"""
        raise NotImplementedError

    # 姿态估计接口
    @abstractmethod
    def predict_pose(
        self, batch: dict[str, torch.Tensor], pred_struc: torch.Tensor, mu: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """根据训练阶段返回旋转矩阵和 YX 平移。"""
        raise NotImplementedError


class InitTask(pl.LightningModule):
    """预训练 VAE，使初始预测形变量接近零。"""

    def __init__(self, em_task: "CryoEMTask"):
        super().__init__()
        self.cfg = em_task.cfg
        self.em_task = em_task
        self.loss_deque = collections.deque([10.0], maxlen=20)

    def training_step(
        self,
        batch: dict[str, torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        """最小化形变量，使初始预测结构贴近参考 PDB。"""
        del batch_idx
        images = batch["proj"]
        pred_deformation, _ = self.em_task.model(
            prepare_images(images, self.cfg.model.input_space)
        )
        return torch.mean(pred_deformation.flatten(start_dim=-2).pow(2))

    def on_train_batch_end(
        self,
        outputs: dict[str, torch.Tensor],
        batch: dict[str, torch.Tensor],
        batch_idx: int,
    ) -> None:
        """最近 20 步平均 loss 足够小时提前结束初始化。"""
        del batch, batch_idx
        self.loss_deque.append(outputs["loss"].item())
        if np.mean(self.loss_deque) <= 1e-4:
            self.trainer.should_stop = True
        self.trainer.should_stop = self.trainer.strategy.broadcast(
            self.trainer.should_stop
        )

    def configure_optimizers(self) -> optim.Optimizer:
        """更新VAE参数。"""
        return optim.AdamW(self.em_task.model.parameters(), lr=1e-4)

    def on_fit_end(self) -> None:
        """记录初始化结束时的平均形变量 loss。"""
        log_to_current(f"Init finished with loss {np.mean(self.loss_deque)}")


class CryoEMTask(AbstractCryoEMTask):
    gmm_centers: torch.Tensor

    
    # pylance显示问题的处理
    gmm_sigmas: torch.Tensor
    gmm_amps: torch.Tensor
    
    def __init__(self, cfg:Config, dataset: StarfileDataSet):
        super().__init__()
        cfg = deepcopy(cfg)
        self.cfg = cfg
        if cfg.get("pose_init", "given") == "hps" and cfg.get("n_imgs_pose_search", 500000) <= 0:
            raise ValueError("pose_init=hps requires a positive pose-search budget")
        # HPS caches initialize optional differentiable per-particle poses.
        self.use_pose_table = cfg.get("use_pose_table", False)
        self.pose_table = PoseTable(len(dataset), cfg.get("optimize_translations", True)) if self.use_pose_table else None
        self.register_buffer("predicted_rots", torch.eye(3).repeat(len(dataset), 1, 1))
        self.register_buffer("predicted_trans", torch.zeros(len(dataset), 2))
        self.register_buffer("predicted_pose_valid", torch.zeros(len(dataset), dtype=torch.bool))
        # HPS setup
        n_imgs_pose_search = cfg.get("n_imgs_pose_search", 500000)
        self.epochs_pose_search = (
            max(2, n_imgs_pose_search // len(dataset) + 1)
            if n_imgs_pose_search > 0 else 0
        )

        self.mask = Mask(cfg.data_process.down_side_shape, rad=cfg.loss.mask_rad_for_image_loss)
        
        meta = Polymer.from_pdb(cfg.dataset_attr.ref_pdb_path)
        self.template_pdb = meta.to_atom_arr()
        num_pts = len(meta)
        in_dim = cfg.data_process.down_side_shape ** 2
        
        # low-pass filtering，返回一个二维低通滤波掩码，形状与图像相同；低频区域：1 高频区域：0
        if hasattr(cfg.data_process, "low_pass_bandwidth"):
            log_to_current(f"Use low-pass filtering w/ {cfg.data_process.low_pass_bandwidth} A")
            lp_mask2d = low_pass_mask2d(cfg.data_process.down_side_shape, cfg.data_process.down_apix,
                                        cfg.data_process.low_pass_bandwidth)
            self.register_buffer("lp_mask2d", torch.from_numpy(lp_mask2d).float())
        else:
            self.lp_mask2d = None

        # 处理二级结构标签
        sec_ids, _ = extract_sec_ids_merge_small_blocks(cfg.dataset_attr.ref_pdb_path ,min_block_len=3)
        # 二级结构块的连接关系
        meta_edge_index, centroids = build_metagraph_knn_from_centroids(pos=meta.coord,sec_ids=sec_ids,k=cfg.knn_num)
        # 这些连接对应的距离。
        centroid_distances = cdist(centroids, centroids)
        edge_dist = centroid_distances[meta_edge_index[0], meta_edge_index[1]]
        # 原子的位置编码
        ref_centers = torch.from_numpy(meta.coord).float()
        self.register_buffer("gmm_centers", ref_centers)
        pe_vector = np.array([centroids[sec_id] - ref_centers[i] for i,sec_id in enumerate(sec_ids)])
        # 二级结构块到原子的连接关系。
        meta_2_node_edge, meta_2_node_vector = get_kplus_neighbor_metanodes(ref_centers,sec_ids,centroids)

        self.model = VAE(in_dim=in_dim,
                out_dim=num_pts * 3,
                sec_ids = torch.from_numpy(np.array(sec_ids)).long(),
                meta_edge_index = meta_edge_index.long(),
                edge_dist = torch.from_numpy(edge_dist).float(),
                pe_vector = torch.from_numpy(pe_vector),
                meta_2_node_edge = meta_2_node_edge.long(),
                meta_2_node_vector = meta_2_node_vector,
                **cfg.model.model_cfg)
        self.use_pose_head = cfg.model.get("use_pose_head", True)
        pose_head_type = cfg.model.get("pose_head_type", "mlp")
        pose_max_angle_deg = cfg.model.get("pose_max_angle_deg", 45.0)
        if pose_head_type == "small_mlp":
            self.pose_head = PoseHeadSmallMLP(
                in_dim=cfg.model.model_cfg.z_dim,
                hidden_dim=cfg.model.get("pose_head_small_hidden_dim", 64),
                max_angle_deg=pose_max_angle_deg,
            )
        elif pose_head_type == "mlp":
            self.pose_head = PoseHeadMLP(
                in_dim=cfg.model.model_cfg.z_dim,
                hidden_dims=tuple(cfg.model.get("pose_head_hidden_dim", (128, 128))),
                max_angle_deg=pose_max_angle_deg,
            )
        else:
            raise ValueError(f"Unknown pose_head_type: {pose_head_type}")
        
        self.deformer = E3Deformer()
        self._prepare_structural_loss_dependencies(meta)
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.history_saved_dirs = []
        
        grid = EMAN2Grid(side_shape=cfg.data_process.down_side_shape, voxel_size=cfg.data_process.down_apix)
        self.grid = grid
        
        # GMM参数
        ref_amps = torch.from_numpy(meta.num_electron).float()
        ref_sigmas = torch.ones_like(ref_amps)
        ref_sigmas.fill_(2.)
        log_to_current(f"1st GMM blob amplitude {ref_amps[0].item()}, sigma {ref_sigmas[0].item()}")
        self.register_buffer("gmm_sigmas", ref_sigmas)
        self.register_buffer("gmm_amps", ref_amps)
        # self.gmm_sigmas = ref_sigmas
        # self.gmm_amps = ref_amps
        
        self.apix = self.cfg.data_process.down_apix
        ctf_params = infer_ctf_params_from_config(cfg)
        self.ctf = CTFCryoDRGN(**ctf_params, num_particles=len(dataset))
        self.translator = SpatialGridTranslate(D=cfg.data_process.down_side_shape, device=self.device)

        # 是否开启HPS
        self.pose_search = (
            HierarchicalPoseSearch(
                cfg, self.gmm_sigmas, self.gmm_amps, self.grid, self.ctf, self.translator
            )
            if self.epochs_pose_search > 0 else None
        )
        self.pose_head.requires_grad_(self.use_pose_head)

    def _prepare_structural_loss_dependencies(self, meta: Polymer) -> None:
        """从参考结构生成连接、距离、二级结构和碰撞 loss 所需索引。"""
        cfg = self.cfg
        connect_pairs = find_continuous_pairs(
            meta.chain_id, meta.res_id, meta.atom_name
        )
        cutoff_pairs = find_quaint_cutoff_pairs(
            meta.coord,
            meta.chain_id,
            meta.res_id,
            cfg.loss.intra_chain_cutoff,
            cfg.loss.inter_chain_cutoff,
            cfg.loss.intra_chain_res_bound,
        )
        cutoff_pairs = remove_duplicate_pairs(cutoff_pairs, connect_pairs)

        if cfg.loss.sse_weight != 0.0:
            sse_pairs = find_quaint_cutoff_pairs(
                meta.coord,
                meta.chain_id,
                meta.res_id,
                cfg.loss.intra_chain_cutoff,
                0,
                20,
            )
            cutoff_pairs = remove_duplicate_pairs(cutoff_pairs, sse_pairs)
            self.register_buffer("sse_pairs", torch.from_numpy(sse_pairs).long())
            sse_dists = calc_dist_by_pair_indices(meta.coord, sse_pairs)
            self.register_buffer("sse_dists", torch.from_numpy(sse_dists).float())
            log_to_current(f"found {len(sse_pairs)} sse_pairs")

        clash_pairs = find_range_cutoff_pairs(
            meta.coord, cfg.loss.clash_min_cutoff
        )
        clash_pairs = remove_duplicate_pairs(clash_pairs, connect_pairs)

        if len(connect_pairs) > 0:
            self.register_buffer(
                "connect_pairs", torch.from_numpy(connect_pairs).long()
            )
            connect_dists = calc_dist_by_pair_indices(meta.coord, connect_pairs)
            self.register_buffer(
                "connect_dists", torch.from_numpy(connect_dists).float()
            )
            log_to_current(f"found {len(connect_pairs)} connect_pairs")
        else:
            log_to_current("connect_pairs is empty")

        if len(cutoff_pairs) > 0:
            cutoff_dists = calc_dist_by_pair_indices(meta.coord, cutoff_pairs)
            self.dist_loss_fn = DistLoss(cutoff_pairs, cutoff_dists, reduction=None)
            cutoff_chain_mask = filter_same_chain_pairs(
                cutoff_pairs, meta.chain_id
            )
            self.register_buffer(
                "cutoff_chain_mask", torch.from_numpy(cutoff_chain_mask)
            )
            log_to_current(f"found {len(cutoff_pairs)} cutoff_pairs")
        else:
            log_to_current("cutoff_pairs is empty")

        if len(clash_pairs) > 0:
            self.register_buffer(
                "clash_pairs", torch.from_numpy(clash_pairs).long()
            )
            log_to_current(f"found {len(clash_pairs)} clash_pairs")
        else:
            log_to_current("clash_pairs is empty")

    def _calculate_connect_loss(self, pred_struc: torch.Tensor) -> torch.Tensor:
        """计算加权的相邻原子连接 loss。"""
        if not hasattr(self, "connect_pairs"):
            return pred_struc.new_tensor(0.0)
        connect_loss = calc_pair_dist_loss(
            pred_struc, self.connect_pairs, self.connect_dists
        )
        return self.cfg.loss.connect_weight * connect_loss

    def _calculate_sse_loss(self, pred_struc: torch.Tensor) -> torch.Tensor:
        """计算加权的二级结构保持 loss。"""
        if not hasattr(self, "sse_pairs"):
            return pred_struc.new_tensor(0.0)
        sse_loss = calc_pair_dist_loss(
            pred_struc, self.sse_pairs, self.sse_dists
        )
        return self.cfg.loss.connect_weight * sse_loss

    def _calculate_dist_loss(self, pred_struc: torch.Tensor) -> torch.Tensor:
        """计算加权的非相邻原子距离保持 loss。"""
        if not hasattr(self, "dist_loss_fn"):
            return pred_struc.new_tensor(0.0)

        dist_loss = self.dist_loss_fn(pred_struc)
        all_dist_loss = self.all_gather(dist_loss)
        all_dist_loss = all_dist_loss.reshape(-1, dist_loss.shape[-1])
        # 单粒子尾批跳过依赖跨粒子方差的距离损失。
        if all_dist_loss.shape[0] < 2:
            return dist_loss.sum() * 0.
        with torch.no_grad():
            keep_mask = torch.ones(
                dist_loss.shape[-1],
                dtype=torch.bool,
                device=dist_loss.device,
            )
            for chain_mask in self.cutoff_chain_mask:
                pair_indices = chain_mask.nonzero(as_tuple=True)[0]
                pair_variance = all_dist_loss.index_select(
                    dim=1, index=pair_indices
                ).var(dim=0)
                chain_keep_mask = pair_variance.lt(
                    torch.quantile(pair_variance, self.cfg.loss.dist_keep_ratio)
                )
                keep_mask[chain_mask] *= chain_keep_mask
            keep_mask = keep_mask.unsqueeze(0).repeat(dist_loss.size(0), 1)
        if not keep_mask.any():
            return dist_loss.sum() * 0.
        return self.cfg.loss.dist_weight * torch.mean(dist_loss[keep_mask])

    def _calculate_clash_loss(self, pred_struc: torch.Tensor) -> torch.Tensor:
        """计算加权的原子碰撞 loss。"""
        if not hasattr(self, "clash_pairs"):
            return pred_struc.new_tensor(0.0)
        clash_loss = calc_clash_loss(
            pred_struc, self.clash_pairs, self.cfg.loss.clash_min_cutoff
        )
        return self.cfg.loss.clash_weight * clash_loss

    @property
    def is_in_pose_search_step(self):
        return 0 <= self.current_epoch < self.epochs_pose_search

    def predict_pose(
        self, batch: dict[str, torch.Tensor], pred_struc: torch.Tensor, mu: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """先 HPS 搜索；后阶段按开关用 head 修正。返回旋转和 YX 平移。"""
        if self.pose_search is None:
            rot_mats, trans_mats = self.get_batch_pose(batch)
        else:
            indices = batch["idx"].long()
            needs_search = (
                torch.ones_like(indices, dtype=torch.bool)
                if self.is_in_pose_search_step else ~self.predicted_pose_valid[indices]
            )
            if needs_search.any():
                search_batch = {
                    key: value[needs_search] if isinstance(value, torch.Tensor)
                    and value.ndim > 0 and value.shape[0] == len(indices) else value
                    for key, value in batch.items()
                }
                # Refresh references after module device moves or checkpoint loading.
                self.pose_search.gmm_sigmas = self.gmm_sigmas
                self.pose_search.gmm_amps = self.gmm_amps
                rotations, translations = self.pose_search.opt_theta_trans(search_batch, pred_struc[needs_search])
                selected = indices[needs_search]
                self.predicted_rots[selected] = rotations.to(self.predicted_rots)
                self.predicted_trans[selected] = translations.to(self.predicted_trans)
                self.predicted_pose_valid[selected] = True
            rot_mats = self.predicted_rots[indices]
            trans_mats = self.predicted_trans[indices]
        if getattr(self, "use_pose_table", False) and not self.is_in_pose_search_step:
            indices = batch["idx"].long()
            missing = ~self.pose_table.valid[indices]
            if missing.any():
                self.pose_table.initialize(indices[missing], rot_mats[missing], trans_mats[missing])
            rot_mats, trans_mats = self.pose_table(indices)
        if self.use_pose_head and not self.is_in_pose_search_step:
            rot_mats = axis_angle_to_matrix(self.pose_head(mu)) @ rot_mats
        return rot_mats, trans_mats


    def _shared_forward(
        self,
        images: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pred_deformation, mu = self.model(prepare_images(images, self.cfg.model.input_space))
        return pred_deformation, mu
    
    def _shared_projection(
        self,
        pred_struc: torch.Tensor,
        rot_mats: torch.Tensor,
    ) -> torch.Tensor:
        pred_images = batch_projection(
            gauss=Gaussian(
                mus=pred_struc,
                sigmas=self.gmm_sigmas.unsqueeze(0),  # (b, num_centers)
                amplitudes=self.gmm_amps.unsqueeze(0)),
            rot_mats=rot_mats,
            line_grid=self.grid.line())
        pred_images = einops.rearrange(pred_images, 'b y x -> b 1 y x')
        return pred_images
    
    def get_batch_pose(
        self,
        batch: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        rot_mats = batch["rotmat"]
        # yx order
        trans_mats = torch.stack((batch["shiftY"].reshape(-1), batch["shiftX"].reshape(-1)), dim=1)
        trans_mats /= self.apix
        return rot_mats, trans_mats
    
    def _apply_ctf(
        self,
        batch: dict[str, torch.Tensor],
        real_proj: torch.Tensor,
        freq_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        f_proj = primal_to_fourier_2d(real_proj)

        pred_ctf_params = {k: batch[k] for k in ('defocusU', 'defocusV', 'angleAstigmatism') if k in batch}
        f_proj = self.ctf(f_proj, batch['idx'], ctf_params=pred_ctf_params, mode="gt", frequency_marcher=None)
        if freq_mask is not None:
            f_proj = f_proj * self.lp_mask2d

        # Note: here only use the real part
        proj = fourier_to_primal_2d(f_proj).real
        return proj
    
    def _shared_infer(
        self,
        batch: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        '''gt图->pred_struc+mu（隐变量）->pred图;
        然后gt图->平移矫正gt图'''
        gt_images = batch["proj"]
        pred_deformation, mu  = self._shared_forward(gt_images)
        pred_struc = self.deformer.transform(pred_deformation, self.gmm_centers)
        rot_mats, trans_mats = self.predict_pose(batch, pred_struc, mu)
        # get gmm projections
        pred_gmm_images = self._shared_projection(pred_struc, rot_mats)
        # apply ctf, low-pass
        pred_gmm_images = self._apply_ctf(batch, pred_gmm_images, self.lp_mask2d)
        
        if trans_mats is not None:
            res = gt_images.shape[-1]
            freq = torch.fft.fftshift(torch.fft.fftfreq(res, device=gt_images.device))
            phase = -2 * torch.pi * (
                trans_mats[:, 0, None, None] * freq[None, :, None]
                + trans_mats[:, 1, None, None] * freq[None, None, :]
            )
            gt_images = fourier_to_primal_2d(
                primal_to_fourier_2d(gt_images) * torch.exp(1j * phase[:, None])
            ).real

        
        return gt_images, pred_gmm_images, pred_struc, mu

    def low_pass_images(self, images: torch.Tensor) -> torch.Tensor:
        '''低通滤波：滤波前gt_images->滤波后lp_gt_images'''
        if self.lp_mask2d is not None:
            f_images = primal_to_fourier_2d(images)
            f_images = f_images * self.lp_mask2d
            images = fourier_to_primal_2d(f_images).real
        return images

    def _save_ckpt(self, ckpt_path: str) -> None:
        """按原版格式保存模型和 GMM 参数。"""
        torch.save(
            {
                "model": self.model.state_dict(),
                "pose_head": self.pose_head.state_dict(),
                "pose_table": self.pose_table.state_dict() if self.pose_table is not None else None,
                "encoder_unregistered": {
                    name: [layer.state_dict() for layer in getattr(self.model.encoder, name, [])]
                    for name in ("attn_layers", "post_norm")
                },
                "gmm_sigmas": self.gmm_sigmas.data,
                "gmm_amps": self.gmm_amps.data,
                "hps_pose": {
                    "rotations": self.predicted_rots.detach().cpu(),
                    "translations_yx": self.predicted_trans.detach().cpu(),
                    "valid": self.predicted_pose_valid.detach().cpu(),
                    "epoch": self.current_epoch,
                },
            },
            ckpt_path,
        )

    def _get_save_dir(self) -> str:
        """返回并创建原版的 epoch_step 输出目录。"""
        save_dir = osp.join(
            self.cfg.work_dir,
            f"{self.current_epoch:04d}_{self.global_step:07d}",
        )
        mkdir_or_exist(save_dir)
        return save_dir

    def _shared_decoding(self, z: torch.Tensor) -> torch.Tensor:
        """把潜变量解码为三维结构。"""
        with torch.no_grad():
            z = z.float().to(self.device)
            pred_deformation = self.model.decoder(z)
            pred_struc = self.deformer.transform(
                pred_deformation, self.gmm_centers
            )
        return pred_struc.squeeze(0)

    def _save_batched_strucs(
        self,
        pred_strucs: torch.Tensor,
        save_path: str,
    ) -> None:
        """把一批预测坐标保存为多模型 PDB。"""
        atom_arrays = []
        for pred_struc in pred_strucs:
            atom_array = self.template_pdb.copy()
            atom_array.coord = pred_struc.cpu().numpy()
            atom_arrays.append(atom_array)
        bt_save_pdb(save_path, struc.stack(atom_arrays))

    def _shared_image_check(self, total: int = 25) -> None:
        """保存原版同名的输入图和预测投影图。"""
        loader = self.trainer.val_dataloaders or self.trainer.test_dataloaders
        if loader is None:
            return

        was_training = self.model.training
        gt_images_list = []
        pred_images_list = []
        sample_count = 0
        self.model.eval()
        with torch.no_grad():
            for batch in loader:
                batch = self.trainer.strategy.batch_to_device(batch)
                gt_images, pred_images, _, _ = self._shared_infer(batch)
                gt_images_list.append(gt_images)
                pred_images_list.append(pred_images)
                sample_count += gt_images.shape[0]
                if sample_count >= total:
                    break
        self.model.train(mode=was_training)

        save_dir = self._get_save_dir()
        gt_images = torch.cat(gt_images_list, dim=0)[:total]
        pred_images = torch.cat(pred_images_list, dim=0)[:total]
        save_tensor_image(gt_images, osp.join(save_dir, "input_image.png"))
        save_tensor_image(
            pred_images,
            osp.join(save_dir, "pred_gmm_image.png"),
            self.mask.mask,
        )

    def validation_step(
        self,
        batch: dict[str, torch.Tensor],
        batch_idx: int,
    ) -> None:
        """收集验证集的潜变量，供 epoch 结束时分析和保存。"""
        del batch_idx
        images = batch["proj"]
        z = self.model.encode(
            prepare_images(images, self.cfg.model.input_space)
        )
        self.validation_step_outputs.append({"z": z, "idx": batch["idx"]})

    def on_validation_start(self) -> None:
        """验证开始时保存与原版同名的检查点。"""
        log_to_current(
            f"Epoch {self.current_epoch} Step {self.global_step} start validation"
        )
        state_dict = self.trainer.strategy.broadcast(self.model.state_dict())
        self.model.load_state_dict(state_dict)
        if self.trainer.is_global_zero:
            save_dir = self._get_save_dir()
            self._save_ckpt(osp.join(save_dir, "ckpt.pt"))
            self.history_saved_dirs.append(save_dir)

    def on_validation_epoch_end(self) -> None:
        """按原版目录结构保存潜变量、投影图和结构轨迹。"""
        all_outputs = merge_step_outputs(self.validation_step_outputs)
        all_outputs = self.all_gather(all_outputs)
        # DDP 的 all_gather 会增加进程维；普通单卡不会。
        if all_outputs["idx"].ndim > 1:
            all_outputs = squeeze_dict_outputs_1st_dim(all_outputs)

        if self.trainer.is_global_zero and len(all_outputs) > 0:
            self._shared_image_check()
            save_dir = self._get_save_dir()

            indices = get_1st_unique_indices(all_outputs["idx"])
            log_to_current(f"Total {len(indices)} unique samples")
            all_outputs = filter_outputs_by_indices(all_outputs, indices)
            z_list = all_outputs["z"].cpu().numpy()
            np.save(osp.join(save_dir, "z.npy"), z_list)

            _, centers = cluster_kmeans(z_list, self.cfg.analyze.cluster_k)
            centers, _ = get_nearest_point(z_list, centers)
            if z_list.shape[-1] > 2 and not self.cfg.analyze.skip_umap:
                log_to_current("Running UMAP...")
                z_emb, reducer = run_umap(z_list)
                centers_emb = reducer.transform(centers)
                try:
                    plot_z_dist(
                        z_emb,
                        extra_cluster=centers_emb,
                        save_path=osp.join(save_dir, "z_distribution.png"),
                    )
                except Exception as error:
                    log_to_current(str(error))
            elif z_list.shape[-1] <= 2:
                try:
                    plot_z_dist(
                        z_list,
                        extra_cluster=centers,
                        save_path=osp.join(save_dir, "z_distribution.png"),
                    )
                except Exception as error:
                    log_to_current(str(error))

            pred_struc = self._shared_decoding(torch.from_numpy(centers))
            self._save_batched_strucs(
                pred_struc, osp.join(save_dir, "pred.pdb")
            )

            pc, pca = run_pca(z_list)
            pca_count = min(3, self.cfg.model.model_cfg.latent_dim)
            for pca_dim in range(1, pca_count + 1):
                start = np.percentile(pc[:, pca_dim - 1], 5)
                stop = np.percentile(pc[:, pca_dim - 1], 95)
                z_pc_traj = get_pc_traj(
                    pca, z_list.shape[1], 10, pca_dim, start, stop
                )
                z_pc_traj, _ = get_nearest_point(z_list, z_pc_traj)
                pred_struc = self._shared_decoding(
                    torch.from_numpy(z_pc_traj)
                )
                self._save_batched_strucs(
                    pred_struc,
                    osp.join(save_dir, f"pca-{pca_dim}.pdb"),
                )

        self.trainer.strategy.barrier()
        self.validation_step_outputs.clear()

    def on_train_start(self) -> None:
        """训练开始时生成原版的初始投影检查图。"""
        if self.trainer.is_global_zero:
            self._shared_image_check()
        self.trainer.strategy.barrier()

    def training_step(
        self,
        batch: dict[str, torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        cfg = self.cfg
        
        # proj loss        
        gt_images, pred_gmm_images, pred_struc, mu = self._shared_infer(batch)
        
        # 低通滤波
        lp_gt_images = self.low_pass_images(gt_images)
        
        gmm_proj_loss = calc_cor_loss(pred_gmm_images, lp_gt_images, self.mask)
        weighted_gmm_proj_loss = cfg.loss.gmm_cryoem_weight * gmm_proj_loss

        weighted_connect_loss = self._calculate_connect_loss(pred_struc)
        weighted_sse_loss = self._calculate_sse_loss(pred_struc)
        weighted_dist_loss = self._calculate_dist_loss(pred_struc)
        weighted_clash_loss = self._calculate_clash_loss(pred_struc)
        # Axis-angle vectors are in radians; average over particles, not coordinates.
        pose_reg_loss = mu.new_zeros(())
        if self.use_pose_head and not self.is_in_pose_search_step:
            delta_omega = self.pose_head(mu)
            pose_reg_loss = self.cfg.loss.get("pose_reg_weight", 0.5) * delta_omega.square().sum(dim=-1).mean()
        if self.use_pose_table and not self.is_in_pose_search_step:
            pose_reg_loss = pose_reg_loss + self.pose_table_regularization(batch)
        
        loss = (
            weighted_gmm_proj_loss
            + weighted_connect_loss
            + weighted_dist_loss
            + weighted_sse_loss
            + weighted_clash_loss
            + pose_reg_loss
        )

        tmp_metric = {
            "loss": loss.item(),
            "cryoem(gmm)": weighted_gmm_proj_loss.item(),
            "con": weighted_connect_loss.item(),
            "sse": weighted_sse_loss.item(),
            "dist": weighted_dist_loss.item(),
            "clash": weighted_clash_loss.item(),
            "pose_reg": pose_reg_loss.item(),
        }
        self.training_step_outputs.append(
            {name: value for name, value in tmp_metric.items() if name != "loss"}
        )
        if self.global_step % cfg.runner.log_every_n_step == 0:
            self.log_dict(tmp_metric)
            metric_text = pretty_dict(tmp_metric, 5) or ""
            log_to_current(
                f"epoch {self.current_epoch} "
                f"[{batch_idx}/{self.trainer.num_training_batches}] | {metric_text}"
            )
        
        return loss

    def on_train_epoch_end(self) -> None:
        """按原版格式打印当前 epoch 的平均 loss。"""
        step_num = len(self.training_step_outputs)
        if step_num == 0:
            return
        average = {
            name: sum(item[name] for item in self.training_step_outputs) / step_num
            for name in self.training_step_outputs[0]
        }
        metric_text = pretty_dict(average, 6) or ""
        log_to_current(
            f"epoch {self.current_epoch} Average | {metric_text}"
        )
        self.training_step_outputs.clear()

    def configure_optimizers(self) -> optim.Optimizer:
        params = [p for p in [*self.model.parameters(), *self.pose_head.parameters()] if p.requires_grad]

        groups = [{"params": params, "lr": self.cfg.optimizer.lr}]
        if self.pose_table is not None:
            groups.append({"params": self.pose_table.parameters(),
                           "lr": self.cfg.optimizer.get("pose_lr", 1e-3), "weight_decay": 0.})
        optimizer = optim.AdamW(groups)
        return optimizer

    def pose_table_regularization(self, batch):
        """Regularize displacement from input/HPS poses, using training inputs only."""
        indices = batch["idx"].long()
        rotations, translations = self.pose_table(indices)
        if self.pose_search is None:
            base_rotations, base_translations = self.get_batch_pose(batch)
        else:
            base_rotations, base_translations = self.predicted_rots[indices], self.predicted_trans[indices]
        # Half the squared chordal distance tends to squared radians at zero.
        rotation_penalty = .5 * (rotations-base_rotations).square().sum((-1, -2)).mean()
        translation_penalty = (translations-base_translations).square().sum(-1).mean()
        return (self.cfg.loss.get("pose_reg_weight", 1.) * rotation_penalty
                + self.cfg.loss.get("translation_reg_weight", .01) * translation_penalty)


def train():
    # 处理cfg&dataset
    cfg = pl_init_exp(exp_prefix=TASK_NAME, backup_list=[
        __file__,
    ], inplace=False)

    
    dataset = StarfileDataSet(
    StarfileDatasetConfig(
        dataset_dir=cfg.dataset_attr.dataset_dir,
        starfile_path=cfg.dataset_attr.starfile_path,
        apix=cfg.dataset_attr.apix,
        side_shape=cfg.dataset_attr.side_shape,
        down_side_shape=cfg.data_process.down_side_shape,
        mask_rad=cfg.data_process.mask_rad,
        power_images=1.0,
        ignore_rots=cfg.get("pose_init", "given") == "hps",
        ignore_trans=cfg.get("pose_init", "given") == "hps", ))
    
    # 若未指定降采样尺寸：大图默认降至 128，小图保持原尺寸
    if cfg.data_process.down_side_shape is None:
        if dataset.side_shape > 256:
            cfg.data_process.down_side_shape = 128
            dataset.down_side_shape = 128
        else:
            cfg.data_process.down_side_shape = dataset.down_side_shape
    # 根据降采样比例更新像素尺寸，保证图像的实际物理尺寸不变        
    cfg.data_process["down_apix"] = dataset.apix
    if dataset.down_side_shape != dataset.side_shape:
        cfg.data_process.down_apix = dataset.side_shape * dataset.apix / dataset.down_side_shape

    # pl_init_exp 最先保存的是原始配置；这里覆盖为补全降采样参数后的配置。
    rank_zero_only(cfg.dump)(osp.join(cfg.work_dir, "config.py"))
    log_to_current(
        f"Set down-sample side_shape {dataset.down_side_shape} "
        f"with apix {cfg.data_process.down_apix}"
    )

    
    
    train_loader = DataLoader(dataset,
                              batch_size=cfg.data_loader.train_batch_per_gpu,
                              shuffle=True,
                              drop_last=False,
                              num_workers=cfg.data_loader.workers_per_gpu)

    test_loader = DataLoader(
        dataset,
        batch_size=cfg.data_loader.val_batch_per_gpu,
        shuffle=False,
        drop_last=False,
        num_workers=cfg.data_loader.workers_per_gpu,
    )
    
    em_task = CryoEMTask(cfg, dataset)

    # 正式训练前先让模型输出接近零形变，与原版初始化流程一致。
    if not cfg.eval_mode and cfg.do_ref_init:
        init_task = InitTask(em_task)
        init_trainer = pl.Trainer(
            max_epochs=3,
            devices=cfg.trainer.devices,
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            precision=cfg.trainer.precision,
            logger=False,
            enable_checkpointing=False,
            enable_model_summary=False,
            enable_progress_bar=False,
            num_sanity_val_steps=0,
        )
        init_trainer.fit(init_task, train_dataloaders=train_loader)

    em_trainer = pl.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        logger=False,
        enable_checkpointing=False,
        enable_model_summary=False,
        enable_progress_bar=False,
        **cfg.trainer,
    )
    em_trainer.fit(
        model=em_task,
        train_dataloaders=train_loader,
        val_dataloaders=test_loader,
    )
    
    return

def prepare_images(images: torch.FloatTensor, space: str):
    assert space in ("real", "fourier")
    if space == "real":
        model_input = einops.rearrange(images, "b 1 ny nx -> b (1 ny nx)")
    else:
        fimages = primal_to_fourier_2d(images)
        model_input = einops.rearrange(torch.view_as_real(fimages), "b 1 ny nx c2 -> b (1 ny nx c2)", c2=2)
    return model_input

if __name__ == "__main__":
    train()
