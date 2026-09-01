"""1AKE 姿态优化与已知真值的姿态扰动恢复实验。

使用示例：
    python optimize_pose_real.py --num-particles 32 --iterations 50

查看全部参数：
    python optimize_pose_real.py --help
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional, Union, cast

import pandas as pd
import numpy as np
import starfile
import torch
from mmengine import Config
from torch.utils.data import DataLoader, Dataset

from cryodyna.gmm.gmm import EMAN2Grid, Gaussian, batch_projection
from cryodyna.utils.losses import calc_cor_loss
from cryodyna.utils.polymer import Polymer
from cryodyna.utils.transforms import SpatialGridTranslate


def euler2rotmat_relion(poses: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
    """把 RELION 欧拉角转换为可求梯度的旋转矩阵。

    参数：poses 为 ``(N, 3)``，三列依次是 Rot、Tilt、Psi，单位为度。
    返回：形状为 ``(N, 3, 3)`` 的旋转矩阵。
    """
    if not torch.is_tensor(poses):
        poses = torch.as_tensor(poses, dtype=torch.float32)
    poses = poses.float()
    if poses.ndim != 2 or poses.shape[1] != 3:
        raise ValueError(f"poses must have shape (N, 3), got {tuple(poses.shape)}")

    alpha, beta, gamma = torch.deg2rad(poses).unbind(dim=1)
    ca, cb, cg = torch.cos(alpha), torch.cos(beta), torch.cos(gamma)
    sa, sb, sg = torch.sin(alpha), torch.sin(beta), torch.sin(gamma)
    cc, cs = cb * ca, cb * sa
    sc, ss = sb * ca, sb * sa

    rows = (
        torch.stack((cg * cc - sg * sa, -cg * cs - sg * ca, cg * sb), dim=1),
        torch.stack((sg * cc + cg * sa, -sg * cs + cg * ca, sg * sb), dim=1),
        torch.stack((-sc, ss, cb), dim=1),
    )
    return torch.stack(rows, dim=1)


def read_star_poses(
    starfile_path: Union[str, Path], apix: float, limit: Optional[int] = None
) -> np.ndarray:
    """从 STAR 文件读取姿态。

    参数：starfile_path 是 STAR 路径；apix 是像素尺寸；limit 限制读取数量。
    返回：``(N, 5)`` 数组，依次为 Rot、Tilt、Psi、OriginX、OriginY；
    平移统一保存为埃。
    """
    star_data = starfile.read(Path(starfile_path))
    if isinstance(star_data, dict):
        star_blocks = cast(Dict[str, Any], star_data)
        particles = cast(pd.DataFrame, star_blocks["particles"])
    else:
        particles = cast(pd.DataFrame, star_data)

    angles = particles.loc[:, ["rlnAngleRot", "rlnAngleTilt", "rlnAnglePsi"]].to_numpy(
        dtype=np.float32
    )
    if "rlnOriginXAngst" in particles:
        shifts = particles.loc[:, ["rlnOriginXAngst", "rlnOriginYAngst"]].to_numpy(
            dtype=np.float32
        )
    elif "rlnOriginX" in particles:
        shifts = (
            particles.loc[:, ["rlnOriginX", "rlnOriginY"]].to_numpy(dtype=np.float32)
            * float(apix)
        )
    else:
        shifts = np.zeros((len(particles), 2), dtype=np.float32)

    poses = np.concatenate((angles, shifts), axis=1).astype(np.float32)
    return poses if limit is None else poses[:limit]


class PoseOptimizationDataset(Dataset):
    """向 DataLoader 提供目标图像、初始姿态和可选真值姿态。

    target_images：``(N, H, W)`` 或 ``(N, 1, H, W)``。
    initial_poses：待优化的 ``(N, 5)`` 姿态。
    true_poses：可选的 ``(N, 5)`` 真值，仅用于评价。
    """

    def __init__(
        self,
        target_images: torch.Tensor,
        initial_poses: Union[np.ndarray, torch.Tensor],
        true_poses: Optional[Union[np.ndarray, torch.Tensor]] = None,
    ) -> None:
        target_images = torch.as_tensor(target_images, dtype=torch.float32).cpu()
        if target_images.ndim == 4 and target_images.shape[1] == 1:
            target_images = target_images[:, 0]
        if target_images.ndim != 3:
            raise ValueError("target_images must have shape (N, H, W)")

        self.target_images = target_images
        self.initial_poses = torch.as_tensor(initial_poses, dtype=torch.float32).cpu()
        if self.initial_poses.ndim != 2 or self.initial_poses.shape[1] != 5:
            raise ValueError("initial_poses must have shape (N, 5)")
        if len(self.target_images) != len(self.initial_poses):
            raise ValueError("target_images and initial_poses must have equal length")

        self.true_poses = None
        if true_poses is not None:
            self.true_poses = torch.as_tensor(true_poses, dtype=torch.float32).cpu()
            if self.true_poses.shape != self.initial_poses.shape:
                raise ValueError("true_poses must have the same shape as initial_poses")

    def __len__(self) -> int:
        return len(self.target_images)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        """按粒子编号返回一个样本，并保留编号供结果复原顺序。"""
        item = {
            "target_image": self.target_images[index],
            "initial_pose": self.initial_poses[index],
            "idx": torch.tensor(index, dtype=torch.long),
        }
        if self.true_poses is not None:
            item["true_pose"] = self.true_poses[index]
        return item


class PoseProjector:
    """把固定 GMM 按指定姿态投影成二维图像。
    """

    def __init__(
        self,
        gmm_centers: torch.Tensor,
        ref_sigmas: torch.Tensor,
        ref_amplitudes: torch.Tensor,
        side_shape: int,
        voxel_size: float,
        apix: float,
        gmm_mask: Optional[torch.Tensor] = None,
        device: Optional[Union[str, torch.device]] = None,
    ) -> None:
        self.device = torch.device(
            device
            if device is not None
            else ("cuda:0" if torch.cuda.is_available() else "cpu")
        )
        self._gmm_centers = torch.as_tensor(gmm_centers, dtype=torch.float32).to(self.device)
        self.ref_sigmas = torch.as_tensor(ref_sigmas, dtype=torch.float32).to(self.device)
        self.ref_amplitudes = torch.as_tensor(ref_amplitudes, dtype=torch.float32).to(self.device)
        if self._gmm_centers.ndim != 2 or self._gmm_centers.shape[1] != 3:
            raise ValueError("gmm_centers must have shape (num_centers, 3)")
        if len(self.ref_sigmas) != len(self._gmm_centers):
            raise ValueError("ref_sigmas and gmm_centers have different lengths")
        if len(self.ref_amplitudes) != len(self._gmm_centers):
            raise ValueError("ref_amplitudes and gmm_centers have different lengths")

        if gmm_mask is None:
            gmm_mask = torch.ones(len(self._gmm_centers), dtype=torch.float32)
        self.gmm_mask = torch.as_tensor(gmm_mask, dtype=torch.float32).to(self.device)
        if len(self.gmm_mask) != len(self._gmm_centers):
            raise ValueError("gmm_mask and gmm_centers have different lengths")

        self.grid = EMAN2Grid(side_shape=side_shape, voxel_size=voxel_size).to(self.device)
        self.translator = SpatialGridTranslate(D=side_shape, device=self.device)
        self.apix = float(apix)

    @property
    def gmm_centers(self) -> torch.Tensor:
        """返回 GMM 中心的副本。"""
        return self._gmm_centers.detach().clone()

    def project(self, poses: torch.Tensor) -> torch.Tensor:
        """计算一批姿态的投影，返回 ``(B, H, W)`` 图像。"""
        poses = poses.float().to(self.device)
        if poses.ndim != 2 or poses.shape[1] != 5:
            raise ValueError("poses must have shape (B, 5)")
        batch_size = poses.shape[0]

        # 整个 batch 只能使用初始化时指定的同一份 GMM。
        centers = self._gmm_centers.unsqueeze(0).expand(batch_size, -1, -1)

        # 模块 1：根据 mask 调整高斯宽度和振幅。
        scale = 4.0 - 3.0 * self.gmm_mask
        sigmas = (self.ref_sigmas * scale).unsqueeze(0).expand(batch_size, -1)
        amplitudes = (self.ref_amplitudes / scale.pow(3)).unsqueeze(0).expand(batch_size, -1)
        # 模块 2：欧拉角转旋转矩阵，再投影三维 GMM。
        rot_mats = euler2rotmat_relion(poses[:, :3])
        projected = batch_projection(
            Gaussian(mus=centers, sigmas=sigmas, amplitudes=amplitudes),
            rot_mats,
            self.grid.line(),
        )

        # 模块 3：STAR 平移为 X/Y 埃；平移器要求 Y/X 像素，因此换单位并调换顺序。
        trans_pixels = poses[:, 3:] / self.apix
        trans_yx = trans_pixels[:, [1, 0]].unsqueeze(1)
        return self.translator.transform(projected, trans_yx).squeeze(1)


class PoseOptimizer:
    """固定结构，仅优化每张图像的五维姿态参数。

    dataloader 提供目标图和初始姿态；projector 生成预测图；lr 是 Adam
    学习率；iterations 是每个 batch 的优化轮数。
    """

    def __init__(
        self,
        dataloader: DataLoader,
        projector: PoseProjector,
        lr: float = 1e-3,
        iterations: int = 50,
    ) -> None:
        self.dataloader = dataloader
        self.projector = projector
        self.lr = float(lr)
        self.iterations = int(iterations)

    def optimize_batch(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> Dict[str, Any]:
        """优化一个 batch，并返回优化前后姿态与 loss。"""
        target_images = batch["target_image"].to(self.projector.device)
        initial_pose = batch["initial_pose"].to(self.projector.device)
        # 模块 1：只把 pose 声明为可训练参数，结构和投影参数保持固定。
        pose = torch.nn.Parameter(initial_pose.detach().clone())
        optimizer = torch.optim.Adam([pose], lr=self.lr)

        # 模块 2：记录优化前 loss，作为效果基线。
        with torch.no_grad():
            initial_loss = calc_cor_loss(
                self.projector.project(pose), target_images
            ).item()

        # 模块 3：投影、计算相关性 loss、反向传播并更新 pose。
        for iteration in range(self.iterations):
            optimizer.zero_grad(set_to_none=True)
            predicted_images = self.projector.project(pose)
            loss = calc_cor_loss(predicted_images, target_images)
            loss.backward()
            optimizer.step()
            print(
                f"Batch {batch_idx:4d}  Iter {iteration:3d}: "
                f"Loss = {loss.detach().item():.6f}"
            )

        # 模块 4：使用最终 pose 重新计算 loss。
        with torch.no_grad():
            final_loss = calc_cor_loss(
                self.projector.project(pose), target_images
            ).item()

        return {
            "idx": batch["idx"].detach().cpu(),
            "initial_pose": initial_pose.detach().cpu(),
            "optimized_pose": pose.detach().cpu(),
            "initial_loss": initial_loss,
            "final_loss": final_loss,
        }

    def run(self, output_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
        """依次优化全部 batch；可把最终姿态保存到 ``output_path``。"""
        num_samples = len(self.dataloader.dataset)
        optimized_poses = np.empty((num_samples, 5), dtype=np.float32)
        initial_poses = np.empty_like(optimized_poses)
        initial_losses = []
        final_losses = []

        for batch_idx, batch in enumerate(self.dataloader):
            result = self.optimize_batch(batch, batch_idx)
            indices = result["idx"].numpy()
            optimized_poses[indices] = result["optimized_pose"].numpy()
            initial_poses[indices] = result["initial_pose"].numpy()
            initial_losses.append(result["initial_loss"])
            final_losses.append(result["final_loss"])

        output = {
            "initial_poses": initial_poses,
            "optimized_poses": optimized_poses,
            "initial_loss": float(np.mean(initial_losses)),
            "final_loss": float(np.mean(final_losses)),
        }
        if output_path is not None:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(output_path, optimized_poses)
        return output


class PosePerturbationExperiment:
    """检验被扰动的 1AKE 姿态能否向已知真值恢复。

    从 STAR 读取真值，用参考 PDB 合成目标图，再给真值添加角度和平移噪声。
    该实验验证优化模块的自洽性，不代表真实含噪粒子图像上的效果。
    """

    def __init__(
        self,
        config_path: Union[str, Path],
        num_particles: int = 32,
        batch_size: int = 8,
        iterations: int = 50,
        angle_noise_deg: float = 5.0,
        shift_noise_angst: float = 2.0,
        seed: int = 1,
        device: Optional[Union[str, torch.device]] = None,
    ) -> None:
        """初始化实验。

        num_particles：实验粒子数；batch_size：每批粒子数；iterations：优化轮数；
        angle_noise_deg：角度扰动标准差（度）；shift_noise_angst：平移扰动
        标准差（埃）；seed：随机种子；device：计算设备。
        """
        self.cfg = Config.fromfile(str(config_path))
        dataset_cfg = self.cfg.dataset_attr
        process_cfg = self.cfg.data_process
        self.apix = float(dataset_cfg.apix)
        side_shape = int(
            getattr(process_cfg, "down_side_shape", None) or dataset_cfg.side_shape
        )
        voxel_size = float(getattr(process_cfg, "down_apix", None) or self.apix)

        # 模块 1：从 STAR 读取已知真值姿态。
        self.true_poses = read_star_poses(
            dataset_cfg.starfile_path, self.apix, limit=num_particles
        )
        # 模块 2：从参考 PDB 建立固定 GMM 投影器。
        self.meta = Polymer.from_pdb(dataset_cfg.ref_pdb_path)
        centers = torch.from_numpy(np.asarray(self.meta.coord, dtype=np.float32))
        amplitudes = torch.from_numpy(
            np.asarray(self.meta.num_electron, dtype=np.float32)
        )
        sigmas = torch.ones_like(amplitudes)
        self.projector = PoseProjector(
            centers,
            sigmas,
            amplitudes,
            side_shape=side_shape,
            voxel_size=voxel_size,
            apix=self.apix,
            device=device,
        )

        # 模块 3：给真值姿态添加可复现的高斯扰动。
        rng = np.random.default_rng(seed)
        perturbation = np.zeros_like(self.true_poses)
        perturbation[:, :3] = rng.normal(
            0.0, angle_noise_deg, size=(len(self.true_poses), 3)
        )
        perturbation[:, 3:] = rng.normal(
            0.0, shift_noise_angst, size=(len(self.true_poses), 2)
        )
        self.perturbed_poses = self.true_poses + perturbation

        # 模块 4：用真值姿态合成目标图像，作为严格已知的优化目标。
        target_chunks = []
        for start in range(0, len(self.true_poses), batch_size):
            true_pose_batch = torch.from_numpy(self.true_poses[start : start + batch_size])
            target_chunks.append(self.projector.project(true_pose_batch).detach().cpu())
        target_images = torch.cat(target_chunks, dim=0)

        dataset = PoseOptimizationDataset(
            target_images=target_images,
            initial_poses=self.perturbed_poses,
            true_poses=self.true_poses,
        )
        self.dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        self.iterations = iterations

    def run(
        self, output_dir: Union[str, Path] = "test_pose/pose_perturbation"
    ) -> Dict[str, Any]:
        """按“优化 → 评价 → 保存”执行完整实验。"""
        result = self._optimize_poses()
        self._calculate_pose_metrics(result)
        self._save_results(result, output_dir)
        return result

    def _optimize_poses(self) -> Dict[str, Any]:
        """运行姿态优化并返回 loss 与优化后姿态。"""
        optimizer = PoseOptimizer(
            self.dataloader, self.projector, iterations=self.iterations
        )
        return optimizer.run()

    def _calculate_pose_metrics(self, result: Dict[str, Any]) -> None:
        """计算优化前后的角度和平移 RMSE，并写入 result。"""
        optimized_poses = result["optimized_poses"]
        initial_angle_delta = (
            self.perturbed_poses[:, :3] - self.true_poses[:, :3] + 180.0
        ) % 360.0 - 180.0
        initial_translation_delta = (
            self.perturbed_poses[:, 3:] - self.true_poses[:, 3:]
        )
        # 模块 5：计算考虑 360 度周期的角度误差和平移误差。
        angle_delta = (
            optimized_poses[:, :3] - self.true_poses[:, :3] + 180.0
        ) % 360.0 - 180.0
        translation_delta = optimized_poses[:, 3:] - self.true_poses[:, 3:]
        result.update(
            {
                "true_poses": self.true_poses,
                "initial_angle_rmse_deg": float(
                    np.sqrt(np.mean(initial_angle_delta**2))
                ),
                "initial_translation_rmse_angst": float(
                    np.sqrt(np.mean(initial_translation_delta**2))
                ),
                "angle_rmse_deg": float(np.sqrt(np.mean(angle_delta**2))),
                "translation_rmse_angst": float(
                    np.sqrt(np.mean(translation_delta**2))
                ),
            }
        )

    def _save_results(
        self, result: Dict[str, Any], output_dir: Union[str, Path]
    ) -> None:
        """保存真值、扰动姿态、优化姿态和评价指标。"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        np.save(output_dir / "true_poses.npy", self.true_poses)
        np.save(output_dir / "perturbed_poses.npy", self.perturbed_poses)
        np.save(output_dir / "optimized_poses.npy", result["optimized_poses"])
        with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
            json.dump(
                {
                    "initial_loss": result["initial_loss"],
                    "final_loss": result["final_loss"],
                    "initial_angle_rmse_deg": result["initial_angle_rmse_deg"],
                    "initial_translation_rmse_angst": result[
                        "initial_translation_rmse_angst"
                    ],
                    "angle_rmse_deg": result["angle_rmse_deg"],
                    "translation_rmse_angst": result["translation_rmse_angst"],
                },
                handle,
                indent=2,
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="运行 1AKE 已知真值姿态扰动恢复实验。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config-path",
        default="/media/nyh/ce66dee3-f568-4265-9a9c-a9c9b34c3973/group2/data/cryodyna/1ake/atom_1ake/config.py",
        help="1AKE 实验配置文件路径",
    )
    parser.add_argument("--num-particles", type=int, default=32, help="参与实验的粒子数")
    parser.add_argument("--batch-size", type=int, default=8, help="每个 batch 的粒子数")
    parser.add_argument("--iterations", type=int, default=50, help="每个 batch 的优化轮数")
    parser.add_argument("--angle-noise-deg", type=float, default=5.0, help="角度扰动标准差，单位为度")
    parser.add_argument("--shift-noise-angst", type=float, default=2.0, help="平移扰动标准差，单位为埃")
    parser.add_argument("--output-dir", default="test_pose/pose_perturbation", help="姿态和评价指标的输出目录")
    args = parser.parse_args()

    experiment = PosePerturbationExperiment(
        config_path=args.config_path,
        num_particles=args.num_particles,
        batch_size=args.batch_size,
        iterations=args.iterations,
        angle_noise_deg=args.angle_noise_deg,
        shift_noise_angst=args.shift_noise_angst,
    )
    result = experiment.run(args.output_dir)
    print(f"Initial loss: {result['initial_loss']:.6f}")
    print(f"Final loss:   {result['final_loss']:.6f}")
    print(
        f"Angle RMSE:   {result['initial_angle_rmse_deg']:.4f} -> "
        f"{result['angle_rmse_deg']:.4f} deg"
    )
    print(
        f"Shift RMSE:   {result['initial_translation_rmse_angst']:.4f} -> "
        f"{result['translation_rmse_angst']:.4f} Angstrom"
    )


if __name__ == "__main__":
    main()
