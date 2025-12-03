import argparse
import os

import torch
import numpy as np
import open3d as o3d

from pytorch3d.ops import knn_points
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import BallTree

from src.debug import debug_visualize_loaded_pointclouds


def compute_knn_associations(src_xyz, dst_xyz, k=1):
    knn_pred = knn_points(
        src_xyz.unsqueeze(0).cuda().contiguous().float(),
        dst_xyz.unsqueeze(0).cuda().contiguous().float(),
        lengths1=None,
        lengths2=None,
        return_nn=True,
        return_sorted=True,
        K=k,
    )
    
    dst_to_src_idx = knn_pred.idx.squeeze(0)
    
    return dst_to_src_idx


# def compute_knn_associations(src_xyz, dst_xyz, k=1):
#     src_np = src_xyz.cpu().numpy()
#     dst_np = dst_xyz.cpu().numpy()
    
#     tree = BallTree(dst_np, metric="minkowski")
#     dist, indices = tree.query(src_np, k=k)
    
#     dst_to_src_idx = torch.tensor(indices, device=src_xyz.device)
    
#     return dst_to_src_idx


def load_slam_reconstructed_gt(args, scene_id):
    '''Load the SLAM reconstruction results, to ensure fair comparison'''
    slam_path = os.path.join(args.replica_root, scene_id, "rgb_cloud")
    
    slam_pointclouds = o3d.io.read_point_cloud(os.path.join(slam_path, "pointcloud.pcd"))
    slam_xyz = torch.tensor(np.asarray(slam_pointclouds.points))
    
    return slam_xyz


def load_gt_pointcloud(gt_pc_path, semantic_info):
    gt_pc_ext = gt_pc_path.suffix
    
    if gt_pc_ext == '.ply':
        from src.pointcloud import load_gt_pointcloud_ply
        return load_gt_pointcloud_ply(gt_pc_path, semantic_info)
    elif gt_pc_ext == '.pcd':
        from src.pointcloud import load_gt_pointcloud_pcd
        return load_gt_pointcloud_pcd(gt_pc_path, semantic_info)
    else:
        raise ValueError(f"Unknown GT pointcloud extension: {gt_pc_path}")


def load_pred_pointcloud(approach_name, *args, **kwargs):
    if approach_name in ['cg', 'conceptgraphs']:
        from adaptors import conceptgraph as cg
        return cg.load_pred_pointcloud(*args, **kwargs)
    elif approach_name in ['bbq', 'beyondbarequeries']:
        from adaptors import bbq
        return bbq.load_pred_pointcloud(*args, **kwargs)
    elif approach_name in ['bbq_experimental']:
        from adaptors import bbq_experimental
        return bbq_experimental.load_pred_pointcloud(*args, **kwargs)
    elif approach_name in ['hovsg', 'hov-sg']:
        from adaptors import hovsg
        return hovsg.load_pred_pointcloud(*args, **kwargs)
    elif approach_name in ["openscene", "OpenScene"]:
        from adaptors import openscene
        return openscene.load_pred_pointcloud(*args, **kwargs)
    else:
        raise ValueError(f"Unknown approach name: {approach_name}")


def evaluate_scen(
    gt_pointcloud,
    pred_pointcloud, 
    class_feats,
    nn_count = 5
):
    gt_xyz, gt_class = gt_pointcloud
    pred_xyz, pred_color, pred_class = pred_pointcloud
    
    # debug_visualize_loaded_pointclouds(pred_class, class_feats['names'], pred_xyz, gt_xyz, gt_class, class_feats['ids'])

    pred_to_gt_idx = compute_knn_associations(gt_xyz, pred_xyz, k=nn_count).cpu()
    
    class_feats['ids'] = list(class_feats['ids']) + [-1]
    
    abandoned_gt_points_idx = torch.isin(gt_class, torch.tensor(class_feats['ids']))
    gt_class_mapped = gt_class[abandoned_gt_points_idx]
    pred_class_mapped = torch.mode(pred_class[pred_to_gt_idx], dim=-1)[0]
    pred_class_mapped = pred_class_mapped[abandoned_gt_points_idx]
    
    confmatrix = confusion_matrix(
        y_true = gt_class_mapped.cpu().numpy(),
        y_pred = pred_class_mapped.cpu().numpy(),
        labels = class_feats['ids']
    )
    
    # assert confmatrix.sum(0)[ignore_index].sum() == 0
    # assert confmatrix.sum(1)[ignore_index].sum() == 0
    
    return {
        "conf_matrix": torch.tensor(confmatrix),
        "labels": torch.tensor(class_feats['ids']),
    }


def eval_loop(args, class_feats, exclude_class, id_to_class_dict, class_to_id_dict):
    conf_matrices = {}
    scene_ids = list(args.scene_ids_str.split())
    
    for scene_id in scene_ids:
        print("Evaluating on:", scene_id)
        conf_matrix, keep_index = evaluate_scen(
            scene_id = scene_id,
            id_to_class_dict = id_to_class_dict,
            class_to_id_dict = class_to_id_dict,
            class_feats = class_feats,
            args = args,
            exclude_class_idx = exclude_class,
        )
        
        conf_matrix = conf_matrix.detach().cpu()

        conf_matrices[scene_id] = {
            "conf_matrix": conf_matrix,
            "keep_index": keep_index,
        }
    
    conf_matrix_all = np.sum([conf_matrix["conf_matrix"].numpy() for conf_matrix in conf_matrices.values()], axis=0)
    keep_index_all = np.unique([conf_matrix["keep_index"] for conf_matrix in conf_matrices.values()])
    
    conf_matrices["all"] = {
        "conf_matrix": torch.tensor(conf_matrix_all),
        "keep_index": torch.tensor(keep_index_all),
    }
    
    return conf_matrices