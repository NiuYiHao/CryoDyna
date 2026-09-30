"""Follow existing training checkpoints and render the measured comparison."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--drgnai-root", type=Path, required=True)
    p.add_argument("--optpose-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--evaluator", type=Path, default=Path(__file__).resolve().parents[1]/"cryodyna/optpose/volume_metrics.py")
    p.add_argument("--renderer", type=Path, default=Path(__file__).with_name("plot_curve_endpoint.py"))
    args = p.parse_args()
    args.output.mkdir(exist_ok=True, parents=True)
    command = [sys.executable, str(Path(__file__).with_name("figure_80s_comparison.py")),
               "--drgnai-root", str(args.drgnai_root), "--optpose-root", str(args.optpose_root),
               "--output", str(args.output), "--evaluator", str(args.evaluator),
               "--renderer", str(args.renderer), "--volume-control", "--progress"]
    previous = None
    while True:
        paths = list(args.drgnai_root.glob("seed*/out/weights.*.pkl"))
        paths += list((args.optpose_root/"evaluation").glob("epoch*.npz"))
        state = tuple(sorted((str(x), x.stat().st_size, x.stat().st_mtime_ns) for x in paths))
        if state != previous:
            result = subprocess.run(command)
            print(json.dumps(dict(time=time.time(), render_returncode=result.returncode)), flush=True)
            if result.returncode not in (0, 2, 3):
                raise RuntimeError("Comparison failed; see retained log")
            if result.returncode == 0 and os.environ.get("CRYODYNA_FIGURE_QA"):
                audits = Path(os.environ["CRYODYNA_FIGURE_QA"])
                for folder in ("viewing_direction", "full_so3"):
                    pdf = args.output/folder/"figure.pdf"
                    with (pdf.parent/"text_audit.json").open("w") as report:
                        subprocess.run([sys.executable, str(audits/"audit_pdf_text.py"),
                                        str(pdf), "--min-pt", "6", "--json"], stdout=report, check=True)
                    subprocess.run([sys.executable, str(audits/"audit_figure_collisions.py"),
                                    str(pdf), "--json-out", str(pdf.parent/"collision_audit.json")], check=True)
            previous = state
            status = json.loads((args.output/"status.json").read_text())
            if status.get("comparison_complete"):
                break
        run_status = args.drgnai_root/"status.txt"
        if run_status.exists() and run_status.read_text().startswith("failed"):
            (args.output/"upstream_failure.txt").write_text(run_status.read_text())
            break
        time.sleep(60)


if __name__ == "__main__": main()
