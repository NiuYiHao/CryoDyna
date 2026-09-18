"""
原版依赖pose。在原版 ``train_atom`` 上增加可学习的旋转姿态修正。

数据流：粒子图像 -> encoder 得到 z ->
    1. 原 decoder 预测结构形变量；
    2. pose head 通过有界的两层 GELU MLP，根据 z 预测三维轴角修正量；
随后用修正后的旋转矩阵投影结构，并沿用原版 loss 训练。

开发情况：重构原版
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

    # Pose 增量修正接口
    @abstractmethod
    def predict_pose(
        self, mu: torch.Tensor, rot_mats: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """预测轴角增量；返回修正后的旋转矩阵和 delta omega。"""
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
        return self.cfg.loss.dist_weight * torch.mean(dist_loss[keep_mask])

    def _calculate_clash_loss(self, pred_struc: torch.Tensor) -> torch.Tensor:
        """计算加权的原子碰撞 loss。"""
        if not hasattr(self, "clash_pairs"):
            return pred_struc.new_tensor(0.0)
        clash_loss = calc_clash_loss(
            pred_struc, self.clash_pairs, self.cfg.loss.clash_min_cutoff
        )
        return self.cfg.loss.clash_weight * clash_loss

    def predict_pose(
        self, mu: torch.Tensor, rot_mats: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """预测固定坐标系轴角增量，并左乘 STAR 旋转矩阵。"""
        delta_omega = self.pose_head(mu)
        delta_rot = axis_angle_to_matrix(delta_omega)
        return delta_rot @ rot_mats, delta_omega


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
        trans_mats = torch.concat((batch["shiftY"].unsqueeze(1), batch["shiftX"].unsqueeze(1)), dim=1)
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
        rot_mats, trans_mats = self.get_batch_pose(batch)

        gt_images = batch["proj"]
        pred_deformation, mu  = self._shared_forward(gt_images)
        pred_struc = self.deformer.transform(pred_deformation, self.gmm_centers)
        rot_mats, _ = self.predict_pose(mu, rot_mats)
        # get gmm projections
        pred_gmm_images = self._shared_projection(pred_struc, rot_mats)
        # apply ctf, low-pass
        pred_gmm_images = self._apply_ctf(batch, pred_gmm_images, self.lp_mask2d)
        
        if trans_mats is not None:
            gt_images = self.translator.transform(einops.rearrange(gt_images, "B 1 NY NX -> B NY NX"),
                                                einops.rearrange(trans_mats, "B C2 -> B 1 C2"))

        
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
                "gmm_sigmas": self.gmm_sigmas.data,
                "gmm_amps": self.gmm_amps.data,
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
        delta_omega = self.pose_head(mu)
        pose_reg_loss = self.cfg.loss.get("pose_reg_weight", 0.) * delta_omega.square().sum(dim=-1).mean()
        
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
        params = [*self.model.parameters(), *self.pose_head.parameters()]

        optimizer = optim.AdamW(params, lr=self.cfg.optimizer.lr)
        return optimizer


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
        ignore_rots=False,
        ignore_trans=False, ))
    
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
                              drop_last=True,
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
