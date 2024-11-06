import json
import os
import subprocess
from collections import defaultdict
from typing import List

from matplotlib.pyplot import step
import numpy as np
import torch
import tyro


def main(results_dir: str, scenes: List[str], num_frame: int):
    print("scenes:", scenes)
    stage = "compress"

    summary = defaultdict(list)
    for scene in scenes:
        scene_dir = os.path.join(results_dir, scene)

        # if use best_step
        try:
            best_step = torch.load(os.path.join(scene_dir, f"ckpts/ckpt_best_rank0.pt"))["step"]
        except:
            best_step = 29999

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

            bitrate = size * 8 / 1024**2 / num_frame * 30
            summary["bitrate"].append(bitrate)

            MB_per_frame = size / 1024**2 / num_frame
            summary["MB_per_frame"].append(MB_per_frame)

        with open(os.path.join(scene_dir, f"stats/{stage}_step{best_step}.json"), "r") as f:
            stats = json.load(f)
            for k, v in stats.items():
                summary[k].append(v)

    stage = "val"
    for scene in scenes:
        scene_dir = os.path.join(results_dir, scene)

        try:
            best_step = torch.load(os.path.join(scene_dir, f"ckpts/ckpt_best_rank0.pt"))["step"]
        except:
            best_step = 29999

        with open(os.path.join(scene_dir, f"stats/{stage}_step{best_step}.json"), "r") as f:
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
