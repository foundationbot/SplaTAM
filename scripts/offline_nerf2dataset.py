'''
Script to capture a dataset from the NeRFCapture iOS App. Code is adapted from instant-ngp/scripts/nerfcapture2nerf.py.
https://github.com/NVlabs/instant-ngp/blob/master/scripts/nerfcapture2nerf.py
'''
#!/usr/bin/env python3

import argparse
import os
import shutil
import sys
from pathlib import Path
import json
from importlib.machinery import SourceFileLoader
import re

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

sys.path.insert(0, _BASE_DIR)

import cv2
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="./configs/iphone/offline_nerfcapture.py", type=str, help="Path to config file.")
    return parser.parse_args()

def dataset_capture_loop(save_path: Path):

    manifest = json.load(open(save_path.joinpath("transforms.json"), "r"))

    # depth_scale = manifest["frames"][0]["depth_scale"]
    depth_scale = 10.0
    manifest["integer_depth_scale"] = float(depth_scale) / 65535.0

    all_img_dir = save_path.joinpath("images")
    all_images = sorted(os.listdir(all_img_dir), key=lambda x: int(re.search(r'\d+', x).group()))

    images_dir = save_path.joinpath("rgb")
    depth_dir = save_path.joinpath("depth")

    images_dir.mkdir(parents=True, exist_ok=True)
    depth_dir.mkdir(parents=True, exist_ok=True)

    total_frames = len(all_images)

    for img_path in all_images:
        print(f"Processing {img_path}")

        full_path = all_img_dir.joinpath(img_path)
        img = cv2.imread(str(full_path))
        img_num = int(re.search(r'\d+', img_path).group())

        h, w, _ = img.shape
        if "depth" in img_path:
            img_type = "depth"
            # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = img[:, :, 0]# + img[:, :, 1] * 256
            img = (img * manifest["integer_depth_scale"])#.astype(np.uint16)
            cv2.imwrite(str(depth_dir.joinpath(f"{img_num}.png")), img)
        else:
            img_type = "rgb"
            cv2.imwrite(str(images_dir.joinpath(f"{img_num}.png")), img)


if __name__ == "__main__":
    args = parse_args()

    # Load config
    experiment = SourceFileLoader(
        os.path.basename(args.config), args.config
    ).load_module()

    config = experiment.config
    dataset_capture_loop(Path(config['workdir']))
