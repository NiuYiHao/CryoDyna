"""Snapshot complete DRGN-AI runs with verified pose files and weight hashes."""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess

REMOTE_INVENTORY = r'''
import hashlib,json,sys,time
from pathlib import Path
root=Path(sys.argv[1])
def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8388608),b''): h.update(b)
 return h.hexdigest()
records=[];files={};completed=[]
for seed in range(6):
 run=root/f'seed{seed}'
 log=run/'out/training.log'
 if not log.is_file() or 'Finished in ' not in log.read_text(): continue
 pairs=[(run/f'out/pose.{e}.pkl',run/f'out/weights.{e}.pkl') for e in range(30)]
 if not all(p.is_file() and w.is_file() for p,w in pairs): continue
 completed.append(seed)
 for epoch,(pose,weight) in enumerate(pairs):
  ph,wh=digest(pose),digest(weight)
  rel=str(pose.relative_to(root))
  files[rel]={'sha256':ph,'bytes':pose.stat().st_size}
  records.append(dict(seed=seed,epoch=epoch,pose=rel,pose_sha256=ph,
    weights=str(weight.relative_to(root)),weights_sha256=wh,weights_bytes=weight.stat().st_size))
 for p in [run/'configs.yaml',run/'out/drgnai-configs.yaml',log]:
  files[str(p.relative_to(root))]={'sha256':digest(p),'bytes':p.stat().st_size}
print(json.dumps(dict(schema=1,created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
 remote_root=str(root),completed_seeds=completed,required_seeds=list(range(6)),
 selection='all complete 30-pass runs at snapshot time; sequential seed order',
 checkpoints=records,files=files)))
'''


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--host', required=True)
    p.add_argument('--remote-root', required=True)
    p.add_argument('--remote-python', required=True)
    p.add_argument('--shared-local-root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    target = args.output/'snapshot.json'
    if target.exists():
        raise FileExistsError('Choose a fresh snapshot output')
    ssh = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=8', '-o', 'ClearAllForwardings=yes']
    command = shlex.quote(args.remote_python) + ' -c ' + shlex.quote(REMOTE_INVENTORY) + ' ' + shlex.quote(args.remote_root)
    manifest = json.loads(subprocess.check_output(ssh+[args.host, command], text=True))
    assert manifest['completed_seeds'], 'A completed run is required for this snapshot'
    manifest['source_host'] = args.host
    names = list(manifest['files'])
    assert all(not Path(n).is_absolute() and '..' not in Path(n).parts for n in names)
    listing = args.output/'transfer_files.txt'
    listing.write_text('\n'.join(names)+'\n')
    subprocess.run(['rsync', '-a', '--partial-dir=.rsync-partial', '--files-from='+str(listing),
                    '-e', shlex.join(ssh), args.host+':'+args.remote_root.rstrip('/')+'/',
                    str(args.output)+'/'], check=True)
    for name, expected in manifest['files'].items():
        file = args.output/name
        assert file.stat().st_size == expected['bytes'] and digest(file) == expected['sha256'], name
    for name in ('inputs', 'evaluation_truth'):
        source = (args.shared_local_root/name).resolve(strict=True)
        (args.output/name).symlink_to(source, target_is_directory=True)
    manifest.update(local_pose_verification='all downloaded files match remote SHA256',
                    local_role='evaluation snapshot; truth is linked after remote training finishes',
                    downloaded_bytes=sum(r['bytes'] for r in manifest['files'].values()))
    target.write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps(dict(completed_seeds=manifest['completed_seeds'],
                          pose_count=len(manifest['checkpoints']),
                          downloaded_bytes=manifest['downloaded_bytes'], manifest=str(target))))


if __name__ == '__main__':
    main()
