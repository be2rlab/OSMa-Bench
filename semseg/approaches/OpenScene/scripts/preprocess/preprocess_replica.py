import glob, os
import multiprocessing as mp
import numpy as np
import imageio
import cv2
import torch
from tqdm import tqdm
import yaml
import sys
import open3d as o3d
import math


def make_intrinsic(fx, fy, mx, my):
    '''Create camera intrinsics.'''

    intrinsic = np.eye(4)
    intrinsic[0][0] = fx
    intrinsic[1][1] = fy
    intrinsic[0][2] = mx
    intrinsic[1][2] = my
    return intrinsic

def adjust_intrinsic(intrinsic, intrinsic_image_dim, image_dim):
    '''Adjust camera intrinsics.'''

    if intrinsic_image_dim == image_dim:
        return intrinsic
    resize_width = int(math.floor(image_dim[1] * float(
                    intrinsic_image_dim[0]) / float(intrinsic_image_dim[1])))
    intrinsic[0, 0] *= float(resize_width) / float(intrinsic_image_dim[0])
    intrinsic[1, 1] *= float(image_dim[1]) / float(intrinsic_image_dim[1])
    # account for cropping here
    intrinsic[0, 2] *= float(image_dim[0] - 1) / float(intrinsic_image_dim[0] - 1)
    intrinsic[1, 2] *= float(image_dim[1] - 1) / float(intrinsic_image_dim[1] - 1)
    return intrinsic


def process_one_scene_2d(f_c, f_d):
    '''process one scene.'''

    # process RGB images
    img_id = int(int(f_c.split('frame')[-1].split('.')[0])/sample_freq)
    img = imageio.v3.imread(f_c)
    img = cv2.resize(img, img_dim, interpolation=cv2.INTER_LINEAR)
    imageio.imwrite(os.path.join(out_dir_color, str(img_id)+'.png'), img)

    # process depth images
    img_id = int(int(f_d.split('depth')[-1].split('.')[0])/sample_freq)
    depth = imageio.v3.imread(f_d).astype(np.uint16)
    depth = cv2.resize(depth, img_dim, interpolation=cv2.INTER_LINEAR)
    imageio.imwrite(os.path.join(out_dir_depth, str(img_id)+'.png'), depth)

    #process poses
    np.savetxt(os.path.join(out_dir_pose, str(img_id)+'.txt'), pose_list[img_id])
    

def process_one_scene_3d(fn):
    '''process one scene.'''

    color_pcd = o3d.io.read_point_cloud(os.path.join(fn, "pointcloud.pcd"))
    gt_classes = torch.tensor(np.load(os.path.join(fn, "semantic.npy")))
    
    colors = np.asarray(color_pcd.colors)
    coords = np.asarray(color_pcd.points)

    # gt_classes = (np.asarray(gt_pcd.colors)[:, 0] * 255).round()
    
    os.makedirs(os.path.join(out_dir, config), exist_ok=True)

    torch.save((coords, colors, gt_classes),
            os.path.join(out_dir, config, scene + '.pth'))
    print(fn)


#####################################
out_dir = os.path.join(sys.argv[3], "replica_2d")
in_path = sys.argv[1]
sample_freq = int(sys.argv[4])
#####################################

os.makedirs(out_dir, exist_ok=True)

scene = os.path.basename(in_path)
config = os.path.basename(os.path.dirname(in_path))

out_dir_color = os.path.join(out_dir, config, scene, 'color')
out_dir_depth = os.path.join(out_dir, config, scene, 'depth')
out_dir_pose = os.path.join(out_dir, config, scene, 'pose')
if not os.path.exists(out_dir_color):
    os.makedirs(out_dir_color)
if not os.path.exists(out_dir_depth):
    os.makedirs(out_dir_depth)
if not os.path.exists(out_dir_pose):
    os.makedirs(out_dir_pose)

# save the camera parameters to the folder
pose_dir = os.path.join(in_path, 'traj.txt')
poses = np.loadtxt(pose_dir).reshape(-1, 4, 4)
pose_list = poses[::sample_freq]

files_color = sorted(glob.glob(os.path.join(in_path, "results", 'frame*.jpg')))
files_depth = sorted(glob.glob(os.path.join(in_path, "results", 'depth*.png')))
files_color = files_color[::sample_freq] 
files_depth = files_depth[::sample_freq] 


# internal parameters loading

with open(os.path.join(in_path, "camera_params.yaml"), 'r') as file:
        data = yaml.safe_load(file)
    
camera_params = data.get('camera_params', {})


w = camera_params['image_width']
h = camera_params['image_height']

img_dim = (640, 360)
original_img_dim = (w, h)

cx = camera_params['cx']
cy = camera_params['cy']
fx = camera_params['fx']
fy = camera_params['fy']

intrinsics = make_intrinsic(fx, fy, cx, cy)

# save the intrinsic parameters of resized images
intrinsics = adjust_intrinsic(intrinsics, original_img_dim, img_dim)
np.savetxt(os.path.join(out_dir, config, scene, 'intrinsics.txt'), intrinsics)

process_one_scene_2d(files_color[0], files_depth[0])
args = tuple(zip(files_color, files_depth))

p = mp.Pool(processes=mp.cpu_count())
p.starmap(process_one_scene_2d, args)
p.close()


#####################################
out_dir = os.path.join(sys.argv[3], "replica_3d")
semantic_dir = sys.argv[2]
#####################################

os.makedirs(out_dir, exist_ok=True)

process_one_scene_3d(semantic_dir)