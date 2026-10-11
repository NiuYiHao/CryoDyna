#!/usr/bin/env python
"""Record dependencies and basic numerical compatibility on an allocated GPU."""
import argparse
import importlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import sys
import time

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", required=True)
args = parser.parse_args()
result = {"status": "running", "python": sys.version, "host": platform.node(),
          "packages": {}, "checks": {}, "scope": "allocated GPU environment preflight; no dataset training"}
started = time.time()
try:
    for module, package in [("torch","torch"),("numpy","numpy"),("scipy","scipy"),
        ("pandas","pandas"),("mrcfile","mrcfile"),("starfile","starfile"),
        ("biotite","biotite"),("gemmi","gemmi"),("mmengine","mmengine"),
        ("lightning","lightning"),("torch_geometric","torch-geometric"),
        ("einops","einops"),("healpy","healpy")]:
        try:
            importlib.import_module(module)
            result["packages"][package] = {"status":"pass", "version":importlib.metadata.version(package)}
        except Exception as exc:
            result["packages"][package] = {"status":"fail", "error":str(exc)}
    import torch
    result["cuda_available"] = torch.cuda.is_available()
    if not result["cuda_available"]:
        raise RuntimeError("Allocated GPU is unavailable to this environment")
    device = torch.cuda.get_device_properties(0)
    result["gpu"] = {"name":device.name,"memory_bytes":device.total_memory,
                     "capability":[device.major,device.minor],"torch_cuda":torch.version.cuda}
    torch.manual_seed(7)
    torch.backends.cuda.matmul.allow_tf32 = False
    x = torch.rand(64,64)
    y = torch.rand(64,64)
    error = float(((x.cuda()@y.cuda()).cpu()-x@y).abs().max())
    restored = torch.fft.irfft2(torch.fft.rfft2(x.cuda()),s=x.shape)
    fft_error = float((restored.cpu()-x).abs().max())
    result["checks"]["matmul_max_abs_difference"] = error
    result["checks"]["fft_roundtrip_max_abs_difference"] = fft_error
    assert error < 1e-4 and fft_error < 1e-5
    root = Path(__file__).resolve().parents[1]
    tests = subprocess.run([sys.executable,"-m","unittest","discover","-s","tests","-p","test_optpose.py","-v"],
                           cwd=root,capture_output=True,text=True,timeout=120)
    result["checks"]["optpose_geometry_tests"] = {"returncode":tests.returncode,
                                                 "output":(tests.stdout+tests.stderr)[-6000:]}
    result["status"] = "pass" if tests.returncode == 0 and all(p["status"]=="pass" for p in result["packages"].values()) else "dependencies_pending"
except Exception as exc:
    result["status"] = "fail"
    result["error"] = str(exc)
result["seconds"] = time.time()-started
out = Path(args.output)
out.parent.mkdir(parents=True,exist_ok=True)
out.write_text(json.dumps(result,indent=2)+"\n")
print(json.dumps(result,indent=2),flush=True)
sys.exit(0 if result["status"]=="pass" else 2)
