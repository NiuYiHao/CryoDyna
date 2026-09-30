"""Separate pose-search diagnostics; oracle structures are evaluation controls."""
from pathlib import Path
import time
import numpy as np
import torch
from .benchmark import resolve_config, make_dataset, make_task, seed_all, move, loader, read_star, write_json
from .geometry import angular_error


def diagnose(config, output, count=16, device="cuda", l_start=12, l_end=32, kept=8, atomic_template=False):
    seed_all(1)
    cfg = resolve_config(config)
    cfg.n_imgs_pose_search = 1
    cfg.use_pose_table = False
    cfg.model.use_pose_head = False
    cfg.t_extent = 0.; cfg.t_n_grid = 1
    cfg.l_start, cfg.l_end, cfg.n_kept_poses = l_start, l_end, kept
    dataset = make_dataset(cfg, "given")
    task = make_task(cfg, dataset, "opt", device)
    indices = np.sort(np.random.default_rng(101).choice(len(dataset), count, replace=False))
    batch = move(torch.utils.data.default_collate([dataset[int(i)] for i in indices]), device)
    truth = batch["rotmat"].cpu().numpy()
    reference = task.gmm_centers[None].expand(count, -1, -1)
    search = task.pose_search
    search.gmm_sigmas, search.gmm_amps = task.gmm_sigmas, task.gmm_amps
    from cryodyna.utils.polymer import Polymer
    _, frame = read_star(cfg.dataset_attr.starfile_path)
    classes = frame.iloc[indices].rlnClassNumber.to_numpy(int)
    pdb_dir = Path(cfg.dataset_attr.ref_pdb_path).parent
    oracle = torch.tensor(np.stack([Polymer.from_pdb(str(pdb_dir/f"1akeA_{i}.pdb")).coord for i in classes]), device=device)
    if atomic_template:
        import gemmi
        from cryodyna.utils.pdb_tools import bt_read_pdb
        atoms = bt_read_pdb(cfg.dataset_attr.ref_pdb_path)[0]
        reference = torch.tensor(atoms.coord, device=device)[None].expand(count, -1, -1)
        oracle = torch.tensor(np.stack([bt_read_pdb(pdb_dir/f"1akeA_{i}.pdb")[0].coord for i in classes]), device=device)
        task.gmm_sigmas = torch.ones(len(atoms), device=device)
        task.gmm_amps = torch.tensor([gemmi.Element(e).atomic_number for e in atoms.element], device=device, dtype=torch.float32)
        search.gmm_sigmas, search.gmm_amps = task.gmm_sigmas, task.gmm_amps
    rows = []
    with torch.no_grad():
        observed = batch["proj"].clone()
        for source, coordinates in [("reference", reference), ("oracle_true_structure_control", oracle)]:
            predicted = task._apply_ctf(batch, task._shared_projection(coordinates, batch["rotmat"]), task.lp_mask2d)
            target = task.low_pass_images(observed)
            p, t = predicted.flatten(1), target.flatten(1)
            corr = (p*t).sum(1) / (p.norm(dim=1)*t.norm(dim=1))
            rows.append(dict(case=source+"_known_pose", cosine=corr.cpu().tolist(),
                             image_std=target.std().item(), projection_std=predicted.std().item()))
            for score in ("l2", "correlation"):
                for image_kind in ("noiseless", "observed"):
                    search.score_type = score
                    batch["proj"] = predicted if image_kind == "noiseless" else observed
                    started = time.time()
                    np.random.seed(1)
                    rotations, trans = search.opt_theta_trans(batch, coordinates)
                    errors = angular_error(rotations.cpu().numpy(), truth)
                    row = dict(case=f"{source}_{image_kind}_{score}", mean_pose_deg=float(errors.mean()),
                               p90_pose_deg=float(np.quantile(errors, .9)), errors=errors.tolist(), seconds=time.time()-started)
                    rows.append(row)
                    print(row, flush=True)
    write_json(Path(output)/"diagnostics.json", {"particle_ids": indices.tolist(), "rows": rows, "atomic_template": atomic_template,
        "scope": "Known-pose and true-structure controls are diagnostic only; training uses no oracle structure."})
