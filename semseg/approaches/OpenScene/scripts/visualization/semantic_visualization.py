import open3d as o3d
import sys
import os

input_dir = sys.argv[1]
scene_name = sys.argv[2]
feature_type = sys.argv[3]


gt = os.path.join(input_dir, scene_name, feature_type, "gt.ply")
pred = os.path.join(input_dir, scene_name, feature_type, "pred.ply")
color = os.path.join(input_dir, scene_name, feature_type, "input.ply")

pcd_gt = o3d.io.read_point_cloud(gt)
pcd_pred = o3d.io.read_point_cloud(pred)
pcd_color = o3d.io.read_point_cloud(color)


o3d.visualization.draw_geometries([pcd_color, pcd_pred])
o3d.visualization.draw_geometries([pcd_color, pcd_gt])