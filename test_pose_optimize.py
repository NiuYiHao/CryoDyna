from typing import Tuple, Union, Optional
import torch
import dataclasses
import numpy as np
@dataclasses.dataclass
class Gaussian:
    mus: Union[torch.Tensor, np.ndarray]
    sigmas: Union[torch.Tensor, np.ndarray]
    amplitudes: Union[torch.Tensor, np.ndarray]
import einops
@dataclasses.dataclass
class Grid:
    coords: torch.Tensor  # (N, 1 or 2 or 3)
    shape: Tuple  # (side_shape, ) * 1 or 2 or 3
def batch_projection(gauss: Gaussian, rot_mats: torch.Tensor, line_grid:Grid, gmm_mask: Optional[torch.Tensor] = None, mask_type = 'sigma') -> torch.Tensor:
    """A quick version of e2gmm projection.

    Parameters
    ----------
    gauss: (b/1, num_centers, 3) mus, (b/1, num_centers) sigmas and amplitudes
    rot_mats: (b, 3, 3)
    line_grid: (num_pixels, 3) coords, (nx, ) shape

    Returns
    -------
    proj: (b, y, x) projections
    """
    if gmm_mask is None:
        gmm_mask = torch.ones(gauss.mus.shape[0],dtype=torch.float32)
        
    centers = einops.einsum(rot_mats, gauss.mus, "b c31 c32, b nc c32 -> b nc c31")

    sigmas = einops.rearrange(gauss.sigmas, 'b nc -> b 1 nc')
    
    amplitudes = gauss.amplitudes
    
    if mask_type == 'amplitude':
        amplitudes = amplitudes * gmm_mask[None, :]
    elif mask_type == 'sigma':
        sigmas = sigmas * (4 - 3* gmm_mask)
        amplitudes =  amplitudes / (4 - 3* gmm_mask)**3
        
    sigmas = 2 * sigmas**2

    
    proj_x = einops.rearrange(line_grid.coords.to(centers.device), "nx -> 1 nx 1") - einops.rearrange(centers[..., 0], "b nc -> b 1 nc")
    proj_x = torch.exp(-proj_x**2 / sigmas)

    proj_y = einops.rearrange(line_grid.coords.to(centers.device), "ny -> 1 ny 1") - einops.rearrange(centers[..., 1], "b nc -> b 1 nc")
    proj_y = torch.exp(-proj_y**2 / sigmas)

    proj = einops.einsum(amplitudes, proj_x, proj_y, "b nc, b nx nc, b ny nc -> b nx ny")
    proj = einops.rearrange(proj, "b nx ny -> b ny nx")
    
    return proj

import sys
from mmengine import Config
from cryodyna.utils.polymer import Polymer
import torch
import numpy as np
import pickle
import biotite.structure as struc
# from cryodyna.utils.align import get_rmsd_loss
sys.path.insert(0,'/lustre/grp/gyqlab/zhangcw/CryoDyna_sd/CryoDyna/')
# from cryostar.utils.dist_loss import  find_continuous_pairs
from scipy.spatial.distance import cdist
from cryodyna.utils.pdb_tools import bt_save_pdb, extract_sec_ids_merge_small_blocks, build_metagraph_knn_from_centroids ,get_kplus_neighbor_metanodes
# work_dir = '10792/atom_10792_10811'
# num_step = '0049_0045650'
work_dir = '/media/nyh/ce66dee3-f568-4265-9a9c-a9c9b34c3973/group2/data/cryodyna/1ake/atom_1ake'
num_step = '0029_0023430'
cfg = Config.fromfile(f'{work_dir}/config.py')
meta = Polymer.from_pdb(cfg.dataset_attr.ref_pdb_path)
ref_centers = torch.from_numpy(meta.coord).float()
ref_amps = torch.from_numpy(meta.num_electron).float()
ref_sigmas = torch.ones_like(ref_amps)
num_pts = len(meta)
sec_ids, _ = extract_sec_ids_merge_small_blocks(cfg.dataset_attr.ref_pdb_path ,min_block_len=3)
meta_edge_index, centroids = build_metagraph_knn_from_centroids( pos=meta.coord,sec_ids=sec_ids,k=cfg.knn_num)
sec_ids = torch.from_numpy(sec_ids).long()
meta_edge_index = meta_edge_index.long()
centroid_distances = cdist(centroids, centroids)
edge_dist = torch.from_numpy(centroid_distances[meta_edge_index[0], meta_edge_index[1]]).float()
pe_vector = torch.from_numpy(np.array([centroids[sec_id] - ref_centers[i] for i,sec_id in enumerate(sec_ids)]))
meta_2_node_edge, meta_2_node_vector = get_kplus_neighbor_metanodes(ref_centers,sec_ids,centroids)# node_meta_dist = cdist(ref_centers,centroids)
# in_dim = cfg.dataset_attr.side_shape ** 2
in_dim = 128 ** 2
from projects.miscs import VAE
model = VAE(in_dim=in_dim, sec_ids = sec_ids, meta_edge_index=meta_edge_index, edge_dist=edge_dist,out_dim = num_pts * 3, pe_vector=pe_vector ,meta_2_node_edge = meta_2_node_edge,meta_2_node_vector=meta_2_node_vector,**cfg.model.model_cfg)
z_list = np.load(f'{work_dir}/{num_step}/z.npy')
z = torch.from_numpy(z_list)
device = 'cuda:0'
weights = torch.load(f'{work_dir}/{num_step}/ckpt.pt')
model.load_state_dict(weights['model'],strict=False)
model.eval()
model.to(device)
batch_size = 500
z_batches = torch.split(z, batch_size)
pred_gmm = []
for i, sampled_z in enumerate(z_batches):
    with torch.no_grad():
        delta = model.eval_z(sampled_z.to(torch.float32).to(device))
    pred_struc = delta.reshape(-1, num_pts, 3) + ref_centers[None, :, :].to(device)
    pred_gmm.append(pred_struc.detach().cpu())
pred_gmm = torch.cat(pred_gmm, dim=0)
import numpy as np
# pred_gmm = np.load('test_pose/pred_gmm.npy')

import mrcfile
import torch
import torch.nn.functional as F
def check_points_in_mask_compact(points, mask, Apix=1.0):
    """紧凑版本"""
    mask = np.transpose(mask, (2, 1, 0)) 
    D = mask.shape[0]
    origin = -D // 2 * Apix
    
    # 计算索引并裁剪到有效范围
    indices = np.floor((points - origin) / Apix).astype(int)
    
    # 创建有效掩码
    valid = (
        (indices[:, 0] >= 0) & (indices[:, 0] < D) &
        (indices[:, 1] >= 0) & (indices[:, 1] < D) &
        (indices[:, 2] >= 0) & (indices[:, 2] < D)
    )
    
    # 初始化结果并赋值有效部分
    result = np.zeros(len(points), dtype=int)
    result[valid] = mask[indices[valid, 0], indices[valid, 1], indices[valid, 2]]
    
    return result
# mrc_data = mrcfile.read('/lustre/grp/gyqlab/share/cryoem_particles/10073/10073_arm_mask.mrc')
# mrc_data = F.interpolate(torch.from_numpy(mrc_data).unsqueeze(0).unsqueeze(0), size=(128,128,128), mode='trilinear', align_corners=False).squeeze(0).squeeze(0)
# mask_bool = mrc_data >= 1e-6
# mask_gmm = check_points_in_mask_compact(meta.coord, mask_bool.numpy(), Apix=4.15625)
mask_gmm = np.ones(len(meta.coord), dtype=int)

from cryodyna.utils.dataio import StarfileDataSet, StarfileDatasetConfig
dataset = StarfileDataSet(
        StarfileDatasetConfig(
            dataset_dir=cfg.dataset_attr.dataset_dir,
            starfile_path=cfg.dataset_attr.starfile_path,
            apix=cfg.dataset_attr.apix,
            side_shape=cfg.dataset_attr.side_shape,
            down_side_shape=cfg.data_process.down_side_shape,
            down_method="interp",
            mask_rad=cfg.data_process.mask_rad,
            power_images=1.0,
            ignore_rots=False,
            ignore_trans=False, ))

import starfile
from pathlib import Path
particle_df = starfile.read(Path(cfg.dataset_attr.starfile_path))
sel_particle_df = particle_df["particles"]
# print(sel_particle_df.keys())
sel_pose = np.column_stack((sel_particle_df["rlnAngleRot"],
                            sel_particle_df["rlnAngleTilt"],
                            sel_particle_df["rlnAnglePsi"],
                            np.array(sel_particle_df["rlnOriginXAngst"] , dtype=np.float32),
                            np.array(sel_particle_df["rlnOriginYAngst"] , dtype=np.float32)))

def euler2rotmat_relion(poses):
    """
    将RELION格式的欧拉角批量转换为旋转矩阵（PyTorch版）
    输入N×3的角度制欧拉角矩阵，输出N×3×3的旋转矩阵
    🌟 已内置角度转弧度，无需外部调用np.radians/torch.deg2rad 🌟
    
    Parameters
    ----------
    poses: torch.Tensor / np.ndarray
        角度制欧拉角矩阵，shape=(N, 3)，每一行是(alpha, beta, gamma)（单位：度）
    
    Returns
    -------
    rot_mats: torch.Tensor
        旋转矩阵，shape=(N, 3, 3)，每一个元素是对应欧拉角的3×3旋转矩阵
    """
    # 1. 输入类型/设备/维度校验 + 统一转为torch.float32张量
    if isinstance(poses, (list, tuple)):
        poses = torch.tensor(poses, dtype=torch.float32)
    elif isinstance(poses, np.ndarray):
        poses = torch.from_numpy(poses).float()
    elif not torch.is_tensor(poses):
        raise TypeError(f"输入必须是list/tuple/np.ndarray/torch.Tensor，当前类型：{type(poses)}")
    
    if poses.ndim != 2 or poses.shape[1] != 3:
        raise ValueError(f"输入必须是N×3的矩阵，当前形状：{poses.shape}")
    
    poses = poses.float()
    device = poses.device  # 保留输入设备（CPU/GPU）
    
    # 2. 内置角度转弧度（完全替代np.radians，纯PyTorch实现）
    poses_rad = torch.deg2rad(poses)  # 核心：角度→弧度，无需外部调用np.radians
    
    # 3. 拆分欧拉角（批量处理）
    alpha = poses_rad[:, 0]
    beta = poses_rad[:, 1]
    gamma = poses_rad[:, 2]
    
    # 4. 计算三角函数值（批量）
    ca = torch.cos(alpha)
    cb = torch.cos(beta)
    cg = torch.cos(gamma)
    sa = torch.sin(alpha)
    sb = torch.sin(beta)
    sg = torch.sin(gamma)
    
    # 预计算中间变量
    cc = cb * ca
    cs = cb * sa
    sc = sb * ca
    ss = sb * sa

    # 5. 构造批量旋转矩阵
    N = poses.shape[0]
    rot_mats = torch.zeros((N, 3, 3), dtype=torch.float32, device=device)
    
    # 按RELION约定填充
    rot_mats[:, 0, 0] = cg * cc - sg * sa
    rot_mats[:, 0, 1] = -cg * cs - sg * ca
    rot_mats[:, 0, 2] = cg * sb
    rot_mats[:, 1, 0] = sg * cc + cg * sa
    rot_mats[:, 1, 1] = -sg * cs + cg * ca
    rot_mats[:, 1, 2] = sg * sb
    rot_mats[:, 2, 0] = -sc
    rot_mats[:, 2, 1] = ss
    rot_mats[:, 2, 2] = cb
    
    return rot_mats

from cryodyna.utils.geom_utils import euler_angles2matrix
rotmats = torch.tensor([euler_angles2matrix(np.radians(pose[0]),np.radians(pose[1]),np.radians(pose[2])) for pose in sel_pose],dtype=torch.float32)
trans = sel_pose[:,3:]

from cryodyna.gmm.gmm import EMAN2Grid, Gaussian
grid = EMAN2Grid(side_shape=cfg.data_process.down_side_shape, voxel_size=cfg.data_process.down_apix)
def _shared_projection(gmm, rot_mats, ref_sigmas,ref_amps,grid,mask_gmm):
    pred_images = batch_projection(
        gauss=Gaussian(
            mus=gmm,
            sigmas=ref_sigmas.unsqueeze(0),  # (b, num_centers)
            amplitudes=ref_amps.unsqueeze(0)),
        rot_mats=rot_mats,
        line_grid= grid.line(),
        gmm_mask = mask_gmm,
        mask_type='sigma')
    pred_images = einops.rearrange(pred_images, 'b y x -> b 1 y x')
    return pred_images

# 以star pose+ref pdb制作target_image
from cryodyna.utils.transforms import SpatialGridTranslate
translator = SpatialGridTranslate(D=cfg.data_process.down_side_shape)
rotmats = torch.split(rotmats,500)
trans = torch.split(torch.from_numpy(trans),500)
target_images = []
for rotmat_, tran_ in zip(rotmats, trans):
    target_image = _shared_projection(torch.tensor(meta.coord)[None,...], rotmat_, ref_sigmas,ref_amps,grid,mask_gmm)
    target_image = translator.transform(einops.rearrange(target_image, "B 1 NY NX -> B NY NX"),
                                                  einops.rearrange(tran_, "B C2 -> B 1 C2"))
    target_images.append(target_image.detach())
    
target_images = torch.cat(target_images,dim=0)
# target_images = np.load('./test_pose/target_images.npy')

from torch.utils.data import Dataset, DataLoader
class PoseOptimizationDataset(Dataset):
    def __init__(
        self,
        target_images: torch.Tensor,       # 目标图像列表，每个 [H, W]
        deformed_gmm: torch.Tensor,           # 共享的3D结构 [N_points, 3]
        initial_poses: torch.Tensor,             # 初始姿态参数 [N_samples, 5]
    ):
        self.target_images = target_images
        self.deformed_gmm = deformed_gmm
        self.initial_poses = initial_poses
        
        assert len(target_images) == len(initial_poses), \
            f"图像数量({len(target_images)})和姿态数量({len(initial_poses)})必须相同"
    
    def __len__(self):
        return len(self.target_images)
    def __getitem__(self, idx):
        return {
            'target_image': self.target_images[idx],
            'deformed_gmm': self.deformed_gmm[idx],
            'initial_pose': self.initial_poses[idx]
        }
        
dataset = PoseOptimizationDataset(
        target_images=target_images,
        deformed_gmm=pred_gmm,
        initial_poses=sel_pose
    )
batch_size=32
num_workers=4
dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False
    )

from cryodyna.utils.losses import calc_cor_loss, calc_kl_loss
from cryodyna.utils.transforms import SpatialGridTranslate
import torch.optim as optim
device = 'cuda:0'
import copy
iter_num = 50
translator = SpatialGridTranslate(D=cfg.data_process.down_side_shape,device=device)
# optimizer = optim.Adam([{'params':pose_params, 'lr':1e-3}])
update_pose = []

# 提前开好文件夹
from pathlib import Path
Path("test_pose").mkdir(parents=True, exist_ok=True)

for batch_idx, batch in enumerate(dataloader):
        target_images,deformed_gmm, initial_poses = batch['target_image'],batch['deformed_gmm'],batch['initial_pose']
        target_images,deformed_gmm,initial_poses = target_images.to(device), deformed_gmm.to(device), initial_poses.to(device)
        # opt = optim.Adam([], lr=1e-3) 
        pose = torch.nn.Parameter(initial_poses, requires_grad=True)
        opt = optim.Adam([pose], lr =1e-3)
        # pose_params = torch.nn.Parameter(torch.from_numpy(copy.deepcopy(initial_poses)).float(),requires_grad=True).to(device)
        # rotmat = torch.tensor([euler_angles2matrix(np.radians(pose[0]),np.radians(pose[1]),np.radians(pose[2])) for pose in initial_poses],dtype=torch.float32)
        # trans = initial_poses[:,3:]
        # target_images,rotmat,trans = target_images.to(device), rotmat.to(device), trans.to(device)
        for it in range(iter_num):
                opt.zero_grad()
                # rotmat = torch.tensor([euler_angles2matrix(np.radians(pose[0]),np.radians(pose[1]),np.radians(pose[2])) for pose in initial_poses],dtype=torch.float32)
                rot_mats = euler2rotmat_relion(pose[:,:3])
                trans = pose[:,3:]
                pred_images = _shared_projection(deformed_gmm, rot_mats, ref_sigmas.to(device),ref_amps.to(device),grid, torch.from_numpy(mask_gmm).to(device))
                pred_images  = translator.transform(einops.rearrange(pred_images, "B 1 NY NX -> B NY NX"),
                                                  einops.rearrange(trans.float(), "B C2 -> B 1 C2"))
                loss = calc_cor_loss(pred_images, target_images)
                loss.backward()
                opt.step()
                print(f"Batch  {batch_idx:3d}   Iter {it:3d}: Loss = {loss.detach().item():.4f}")
        update_pose.append(pose.detach().cpu().numpy())

np.save('./test_pose/update_pose_mask_sigma.npy',np.vstack(update_pose))
