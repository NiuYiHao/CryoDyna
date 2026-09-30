"""Run the explicitly labelled fixed-volume Optpose control on existing images.

Inputs contain images, CTF and a density reference. Ground-truth poses enter a
separate evaluator only. Output checkpoints carry particle identities.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import pickle
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import mrcfile
import numpy as np
import torch
from cryodyna.optpose.pose import PoseTable, subdivide
from cryodyna.optpose.volume_prior import FourierVolume, normalized_features, correlation_loss
from cryodyna.utils import so3_grid, lie_tools


def save_json(path, value):
    tmp = path.with_suffix(path.suffix+".tmp")
    tmp.write_text(json.dumps(value, indent=2)+"\n")
    tmp.replace(path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--particles", type=Path, required=True)
    p.add_argument("--ctf", type=Path, required=True)
    p.add_argument("--volume", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--count", type=int, default=100000)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--device", default="cpu")
    p.add_argument("--lr", type=float, default=.003)
    p.add_argument("--record-search-stages", action="store_true")
    args = p.parse_args()
    if (args.output/"manifest.json").exists():
        raise FileExistsError("Use a fresh output directory to preserve previous evidence")
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/"evaluation").mkdir(exist_ok=True)
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    ctf = np.asarray(pickle.load(args.ctf.open("rb")))
    if not (ctf == ctf[0]).all():
        raise ValueError("This shared-template control requires the deposited constant CTF")
    n = args.count
    device = torch.device(args.device)
    data = mrcfile.mmap(str(args.particles), mode="r")
    if not 0 < n <= len(data.data):
        raise ValueError("Count exceeds image stack")
    start = time.time()
    manifest = dict(method="CryoDyna-optpose (fixed-volume control)",
                    variant="Hopf HPS + existing PoseTable; fixed external density; pose-only fitting",
                    training_pose_access="none", initial_translation="zero; centered simulation condition",
                    particles=str(args.particles.resolve()), ctf=str(args.ctf.resolve()),
                    volume=str(args.volume.resolve()), volume_sha256=hashlib.sha256(args.volume.read_bytes()).hexdigest(),
                    particle_selection=f"first {n} in stored order", arguments=vars(args).copy(),
                    uncertainty="deterministic control conditional on fixed reference; no training-replicate claim")
    manifest["arguments"] = {k: str(v) if isinstance(v, Path) else v for k, v in manifest["arguments"].items()}
    save_json(args.output/"manifest.json", manifest)
    status_path = args.output/"status.json"
    def status(stage, **kw):
        row = dict(stage=stage, elapsed_s=time.time()-start, **kw)
        save_json(status_path, row); print(json.dumps(row), flush=True)
    models = {r: FourierVolume(args.volume, r, ctf[0]).to(device) for r in (12, 17, 22, 27, 32)}
    quats = torch.from_numpy(so3_grid.grid_SO3(2)).to(device)
    rots = lie_tools.quaternions_to_SO3(quats)
    with torch.no_grad():
        templates = torch.cat([normalized_features(models[12](r)) for r in rots.split(128)])
    chosen_q = torch.empty((n, 8, 4), device=device)
    chosen_ind = np.empty((n, 8, 2), dtype=int)
    final_rot = torch.empty((n, 3, 3), device=device)
    search_trace = {} if args.record_search_stages else None
    if search_trace is not None:
        for stage in range(5):
            search_trace[stage] = torch.empty((n, 3, 3), device=device)
    status("coarse_hps")
    with torch.no_grad():
        for a in range(0, n, 256):
            b = min(a+256, n)
            images = torch.as_tensor(data.data[a:b].copy(), device=device)
            y = normalized_features(models[12].observations(images))
            top = (y @ templates.T).topk(8, dim=1).indices
            chosen_q[a:b] = quats[top]
            if search_trace is not None:
                search_trace[0][a:b] = lie_tools.quaternions_to_SO3(quats[top[:, 0]])
            chosen_ind[a:b] = so3_grid.get_base_ind(top.cpu().numpy().reshape(-1), 2).reshape(b-a, 8, 2)
            if a % 4096 == 0: status("coarse_hps", particles=b, total=n)
        for a in range(0, n, 8):
            b = min(a+8, n); batch = b-a
            q, ind = chosen_q[a:b], chosen_ind[a:b]
            images = torch.as_tensor(data.data[a:b].copy(), device=device)
            for level, radius in enumerate((17, 22, 27, 32)):
                children, grids, rr = subdivide(q.reshape(-1, 4), ind.reshape(-1, 2), 2+level, device)
                count = children.numel()//(batch*4)
                pred = models[radius](rr).reshape(batch, count, -1)
                y = models[radius].observations(images)
                scores = (pred.conj()*y[:, None]).real.sum(-1) / pred.abs().square().sum(-1).sqrt().clamp_min(1e-20)
                keep = scores.topk(8, dim=1).indices
                q = children.reshape(batch, count, 4)[torch.arange(batch, device=device)[:, None], keep]
                ind = grids.reshape(batch, count, 2)[np.arange(batch)[:, None], keep.cpu().numpy()]
                if search_trace is not None:
                    search_trace[level+1][a:b] = lie_tools.quaternions_to_SO3(q[:, 0])
            final_rot[a:b] = lie_tools.quaternions_to_SO3(q[:, 0])
            if a % 512 == 0: status("local_hps", particles=b, total=n)
    if search_trace is not None:
        (args.output/"search").mkdir(exist_ok=True)
        for stage, rotations in search_trace.items():
            np.savez(args.output/"search"/f"stage{stage:02d}.npz",
                     rotations=rotations.cpu().numpy(), particle_ids=np.arange(n),
                     radius=(12, 17, 22, 27, 32)[stage],
                     phase="coarse_grid" if stage == 0 else "local_refinement")
    def save_epoch(epoch, r, trans):
        path = args.output/"evaluation"/f"epoch{epoch:03d}.npz"
        tmp = path.with_suffix(".tmp")
        with tmp.open("wb") as f:
            np.savez(f, rotations=r.detach().cpu().numpy(), translations_yx=trans.detach().cpu().numpy(), particle_ids=np.arange(n))
        tmp.replace(path)
    translation = torch.zeros((n, 2), device=device)
    save_epoch(0, final_rot, translation)
    state_rotation = final_rot[:, :2].reshape(n, 6).clone()
    moments = [torch.zeros_like(state_rotation), torch.zeros_like(state_rotation),
               torch.zeros_like(translation), torch.zeros_like(translation)]
    model = models[32]
    status("pose_optimization", epoch=0)
    for epoch in range(1, args.epochs+1):
        losses = []
        for a in range(0, n, args.batch):
            b = min(a+args.batch, n); batch = b-a
            table = PoseTable(batch, translations=True).to(device)
            with torch.no_grad():
                table.rotation.copy_(state_rotation[a:b]); table.translation.copy_(translation[a:b]); table.valid.fill_(True)
            images = torch.as_tensor(data.data[a:b].copy(), device=device)
            y = model.observations(images)
            r, tr = table(torch.arange(batch, device=device))
            loss = correlation_loss(model(r, tr), y)
            loss.sum().backward()
            with torch.no_grad():
                # Independent Adam states per particle; each row updates exactly once per pass.
                for parameter, moment, variance, target in [
                    (table.rotation, moments[0], moments[1], state_rotation),
                    (table.translation, moments[2], moments[3], translation)]:
                    grad = parameter.grad
                    moment[a:b].mul_(.9).add_(grad, alpha=.1)
                    variance[a:b].mul_(.999).addcmul_(grad, grad, value=.001)
                    target[a:b] = parameter-args.lr*(moment[a:b]/(1-.9**epoch))/(variance[a:b].sqrt()/(1-.999**epoch)**.5+1e-8)
                table.rotation.copy_(state_rotation[a:b]); table.translation.copy_(translation[a:b])
                final_rot[a:b] = table(torch.arange(batch, device=device))[0]
            losses.extend(loss.detach().cpu().tolist())
        save_epoch(epoch, final_rot, translation)
        status("pose_optimization", epoch=epoch, loss=float(np.mean(losses)), total_epochs=args.epochs)
    data.close()
    status("complete", epochs=args.epochs)


if __name__ == "__main__":
    main()
