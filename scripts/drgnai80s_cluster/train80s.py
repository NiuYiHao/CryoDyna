"""Memory adapter: preserve homogeneous network and global optimizer batch."""
import os,sys,json,copy
from pathlib import Path
import torch
from torch.utils.checkpoint import checkpoint
from src.models import HyperVolume
original_forward=HyperVolume.forward
CHUNK=32768

def memory_forward(self,x,z):
    if self.z_dim != 0 or z is not None:
        return original_forward(self,x,z)
    flat=x.reshape(-1,3)
    def f(a): return original_forward(self,a[None],None).reshape(-1)
    ys=[checkpoint(f,a,use_reentrant=False) if torch.is_grad_enabled() else f(a) for a in flat.split(CHUNK)]
    return torch.cat(ys).reshape(x.shape[:-1])

if '--verify' in sys.argv:
    torch.manual_seed(3); torch.set_num_threads(4)
    a=HyperVolume(129,0,3,256,'gaussian',64,.5,'hartley'); b=copy.deepcopy(a)
    x=(torch.rand(2,18000,3)-.5).requires_grad_(); y=x.detach().clone().requires_grad_()
    p=original_forward(a,x,None); q=memory_forward(b,y,None)
    p.square().mean().backward(); q.square().mean().backward()
    errs={'forward_max_abs':float((p-q).abs().max()),'coordinate_gradient_max_abs':float((x.grad-y.grad).abs().max()),'parameter_gradient_max_abs':max(float((u.grad-v.grad).abs().max()) for u,v in zip(a.parameters(),b.parameters()) if u.grad is not None)}
    assert torch.allclose(p,q,atol=2e-6,rtol=2e-5)
    assert torch.allclose(x.grad,y.grad,atol=2e-7,rtol=2e-4)
    for u,v in zip(a.parameters(),b.parameters()):
        if u.grad is not None: assert torch.allclose(u.grad,v.grad,atol=2e-6,rtol=2e-4)
    print(json.dumps(errs,indent=2)); sys.exit()
HyperVolume.forward=memory_forward
from src.command_line import run_cryodrgn_ai
if __name__ == "__main__":
    run_cryodrgn_ai()
