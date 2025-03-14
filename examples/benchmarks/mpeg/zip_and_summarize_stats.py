import json
import os
import subprocess
from collections import defaultdict
from typing import List

import numpy as np
import tyro


def main(results_dir: str, rps: List[str]):
    print("rps:", rps)
    stage = "compress"

    summary = defaultdict(dict)
    for rp in rps:
        rp_dir = os.path.join(results_dir, rp)

        if stage == "compress":
            zip_path = f"{rp_dir}/compression.zip"
            if os.path.exists(zip_path):
                subprocess.run(f"rm {zip_path}", shell=True)
            subprocess.run(f"zip -r {zip_path} {rp_dir}/compression/", shell=True)
            out = subprocess.run(
                f"stat -c%s {zip_path}", shell=True, capture_output=True
            )
            size = int(out.stdout) # Byte
            # summary["size"].append(size)
            summary[rp]["size"] = size
        
        try:
            with open(os.path.join(rp_dir, f"stats/{stage}_step29999.json"), "r") as f:
                stats = json.load(f)
                for k, v in stats.items():
                    summary[rp][k] = v
        except:
            with open(os.path.join(rp_dir, f"stats/{stage}_step0000.json"), "r") as f:
                stats = json.load(f)
                for k, v in stats.items():
                    summary[rp][k] = v           
    try:
        stage = "val"
        for rp in rps:
            rp_dir = os.path.join(results_dir, rp)
            try:
                with open(os.path.join(rp_dir, f"stats/{stage}_step29999.json"), "r") as f:
                    stats = json.load(f)
                    for k, v in stats.items():
                        if k in ['psnr', 'ssim', 'lpips']:
                            summary[rp]['val_'+k] = v
            except:
                with open(os.path.join(rp_dir, f"stats/{stage}_step-001.json"), "r") as f:
                    stats = json.load(f)
                    for k, v in stats.items():
                        if k in ['psnr', 'ssim', 'lpips']:
                            summary[rp]['val_'+k] = v            
    except:
        print("Could not find val stats, so do not include these val metrics in summary json.")

    # for k, v in summary.items():
    #     print(k, np.mean(v))

    # mean_summary = {k: np.mean(v) for k, v in summary.items()}

    with open(f"{results_dir}/rp_summary.json", "w") as fp:
        json.dump(summary, fp, indent=2)

    print(f"Summary result is saved to: {results_dir}/rp_summary.json")

if __name__ == "__main__":
    tyro.cli(main)
