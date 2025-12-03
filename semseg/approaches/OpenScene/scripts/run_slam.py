import argparse
import json
import yaml
import os
import glob
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from dataclasses import dataclass

import open3d as o3d

from typing import Dict, List, Optional, Union


@dataclass
class Intrinsic:
    """Camera intrinsics"""

    def __init__(self, width, height, fx, fy, cx, cy, depth_scale):
        self.width = width
        self.height = height
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.depth_scale = depth_scale

    def __repr__(self):
        return f"Intrinsic(\
            width={self.width}, \
            height={self.height}, \
            fx={self.fx}, \
            fy={self.fy}, \
            cx={self.cx}, \
            cy={self.cy}, \
            depth_scale={self.depth_scale}, \
        )"


class MappingDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_path,
        stride: Optional[int] = 1,
        start: Optional[int] = 0,
        end: Optional[int] = -1,
        load_semantics: bool = False,
    ):
        self._data_path = data_path
        self._slice = slice(start, end, stride)
        self._load_semantics = load_semantics

        self._rgb_paths = sorted(glob.glob(os.path.join(self._data_path, 'results/frame*.jpg')))
        self._depth_paths = sorted(glob.glob(os.path.join(self._data_path, 'results/depth*.png')))

        assert len(self._rgb_paths) == len(self._depth_paths)

        if self._load_semantics:
            self._semantic_paths = sorted(glob.glob(os.path.join(self._data_path, 'results/semantic*.png')))

            assert len(self._rgb_paths) == len(self._semantic_paths)

        self._poses = self._load_poses()

        assert len(self._poses) == len(self._rgb_paths)

        self._intrinsics = self._load_intrinsics()


    def __len__(self):
        return len(self._rgb_paths[self._slice])


    def __getitem__(self, index):
        rgb = o3d.io.read_image(self._rgb_paths[self._slice][index])
        depth = o3d.io.read_image(self._depth_paths[self._slice][index])

        if self._load_semantics:
            semantics = o3d.io.read_image(self._semantic_paths[self._slice][index])
        else:
            semantics = None

        pose = self._poses[self._slice][index]
        intrinsics = self._intrinsics

        return rgb, depth, semantics, pose, intrinsics


    def _load_poses(self):
        with open(os.path.join(self._data_path, "traj.txt"), "r") as file:
            poses = []
            for line in file:
                pose = np.fromstring(line, dtype=float, sep=" ")
                pose = np.reshape(pose, (4, 4))
                poses.append(pose)

        return poses


    def _load_intrinsics(self):
        yaml_file = os.path.join(self._data_path, "camera_params.yaml")

        with open(yaml_file, "r") as file:
            data = yaml.safe_load(file)

        camera_params = data["camera_params"]

        intrinsic = Intrinsic(
            width = camera_params["image_width"],
            height = camera_params["image_height"],
            fx = camera_params["fx"],
            fy = camera_params["fy"],
            cx = camera_params["cx"],
            cy = camera_params["cy"],
            depth_scale = camera_params["png_depth_scale"],
        )

        return intrinsic


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_root", type=Path, required=True,
    )
    parser.add_argument(
        "--scene_id", type=str, required=True
    )
    parser.add_argument(
        "--dataset_config", type=str, required=False,
        help="This path may need to be changed depending on where you run this script. "
    )
    # parser.add_argument("--image_height", type=int, default=480)
    # parser.add_argument("--image_width", type=int, default=640)

    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=-1)
    parser.add_argument("--stride", type=int, default=1)

    parser.add_argument("--downsample_rate", type=int, default=1)

    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--save_pcd", action="store_true", default=True)
    parser.add_argument("--save_ply", action="store_true")
    parser.add_argument("--save_h5", action="store_true")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")

    parser.add_argument("--load_semseg", action="store_true",
                        help="Load GT semantic segmentation and run fusion on them.")

    return parser


def create_semantic_point_cloud(
    rgb, depth, intrinsic, pose, semantics=None
):
    # Create RGBD image
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        o3d.geometry.Image(rgb),
        o3d.geometry.Image(depth),
        depth_scale=intrinsic.depth_scale,
        depth_trunc=np.inf,
        convert_rgb_to_intensity=False,
    )

    # Create point cloud
    color_pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd,
        o3d.camera.PinholeCameraIntrinsic(
            width=intrinsic.width,
            height=intrinsic.height,
            fx=intrinsic.fx,
            fy=intrinsic.fy,
            cx=intrinsic.cx,
            cy=intrinsic.cy,
        ),
    )

    color_pcd.transform(pose)

    if semantics is not None:
        # Create semantic image
        semantic_color = np.repeat(np.asarray(semantics, dtype=np.uint8)[..., None], 3, axis=-1)

        semantic_d = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(semantic_color),
            o3d.geometry.Image(depth),
            depth_scale=intrinsic.depth_scale,
            depth_trunc=np.inf,
            convert_rgb_to_intensity=False,
        )

        # Create semantic point cloud
        semantic_pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
            semantic_d,
            o3d.camera.PinholeCameraIntrinsic(
                width = intrinsic.width,
                height = intrinsic.height,
                fx = intrinsic.fx,
                fy = intrinsic.fy,
                cx = intrinsic.cx,
                cy = intrinsic.cy,
            ),
        )

        semantic_pcd.transform(pose)
    else:
        semantic_pcd = None

    return color_pcd, semantic_pcd


def main(args: argparse.Namespace):
    dataset = MappingDataset(
        data_path = os.path.join(args.dataset_root, args.scene_id),
        stride = args.stride,
        start = args.start,
        end = args.end,
        load_semantics = args.load_semseg
    )

    color_map = o3d.geometry.PointCloud()
    semantic_map = o3d.geometry.PointCloud() if args.load_semseg else None

    for (rgb, depth, semantics, pose, intrinsics) in tqdm(dataset):
        color_pcd, semantic_pcd = create_semantic_point_cloud(
            rgb, depth, intrinsics, pose, semantics
        )

        color_map += color_pcd

        if semantic_map is not None:
            semantic_map += semantic_pcd

    color_map = color_map.uniform_down_sample(every_k_points=args.downsample_rate)

    if semantic_map is not None:
        semantic_map = semantic_map.uniform_down_sample(every_k_points=args.downsample_rate)

    if args.visualize:
        o3d.visualization.draw_geometries([color_map])

        if semantic_map is not None:
            o3d.visualization.draw_geometries([semantic_map])

    dir_to_save_map = os.path.join(args.dataset_root, args.scene_id, "rgb_cloud")

    if args.save_pcd or args.save_ply:
        try:
            os.makedirs(dir_to_save_map, exist_ok=False)
        except Exception as _:
            pass

    if args.save_pcd:
        print(f'Saving .pcd files to "{dir_to_save_map}"')

        o3d.io.write_point_cloud(
            os.path.join(dir_to_save_map, "pointcloud.pcd"), 
            color_map
        )

        if semantic_map is not None:
            o3d.io.write_point_cloud(
                os.path.join(dir_to_save_map, "semantic.pcd"), 
                semantic_map
            )

    if args.save_ply:
        print(f'Saving .ply files to "{dir_to_save_map}"')

        o3d.io.write_point_cloud(
            os.path.join(dir_to_save_map, "pointcloud.ply"), 
            color_map
        )

        if semantic_map is not None:
            o3d.io.write_point_cloud(
                os.path.join(dir_to_save_map, "semantic.ply"), 
                semantic_map
            )


if __name__ == "__main__":
    args = get_parser().parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    main(args)