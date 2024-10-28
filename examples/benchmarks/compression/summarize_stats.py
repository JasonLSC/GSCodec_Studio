import json
import os
import subprocess
from collections import defaultdict
from typing import List

import numpy as np
import tyro


def main(results_dir: str, scenes: List[str]):
    print("scenes:", scenes)
    stage = "compress"

    summary = defaultdict(list)
    for scene in scenes:
        scene_dir = os.path.join(results_dir, scene)

        if stage == "compress":
            zip_path = f"{scene_dir}/compression.zip"
            if os.path.exists(zip_path):
                subprocess.run(f"rm {zip_path}", shell=True)
            subprocess.run(f"zip -r {zip_path} {scene_dir}/compression/", shell=True)
            out = subprocess.run(
                f"stat -c%s {zip_path}", shell=True, capture_output=True
            )
            size = int(out.stdout)
            summary["size"].append(size)

        with open(os.path.join(scene_dir, f"stats/{stage}_step29999.json"), "r") as f:
            stats = json.load(f)
            for k, v in stats.items():
                summary[k].append(v)

    stage = "val"
    for scene in scenes:
        scene_dir = os.path.join(results_dir, scene)
        with open(os.path.join(scene_dir, f"stats/{stage}_step29999.json"), "r") as f:
            stats = json.load(f)
            for k, v in stats.items():
                if k in ['psnr', 'ssim', 'lpips']:
                    summary['val_'+k].append(v)

    for k, v in summary.items():
        print(k, np.mean(v))

    mean_summary = {k: np.mean(v) for k, v in summary.items()}

    with open(f"{results_dir}/comp_summary.json", "w") as fp:
        json.dump(mean_summary, fp)

if __name__ == "__main__":
    tyro.cli(main)
