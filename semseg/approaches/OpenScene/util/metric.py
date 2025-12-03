'''IoU'''
import os
import json
import pickle
import pprint

import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
from dataset.label_constants import *

def compute_metrics(conf_matrix, class_names):
    tp = np.diag(conf_matrix)
    fp = conf_matrix.sum(axis=0) - tp
    fn = conf_matrix.sum(axis=1) - tp
    
    ious = tp / np.maximum(fn + fp + tp, 1e-7)
    miou = ious.mean()
    f_miou = (ious * (tp + fn) / conf_matrix.sum()).sum()
    
    precision = tp / np.maximum(tp + fp, 1e-7)
    recall = tp / np.maximum(tp + fn, 1e-7)
    
    f1score = 2 * precision * recall / np.maximum(precision + recall, 1e-7)

    mdict = {
        "class_names": class_names,
        "num_classes": len(class_names),
        "iou": ious.tolist(),
        "miou": miou.item(),
        "fmiou": f_miou.item(),
        "acc0.15": (ious > 0.15).sum().item(),
        "acc0.25": (ious > 0.25).sum().item(),
        "acc0.50": (ious > 0.50).sum().item(),
        "acc0.75": (ious > 0.75).sum().item(),
        "precision": precision.tolist(),
        "recall": recall.tolist(),
        "f1score": f1score.tolist()
    }

    return mdict


def get_metrics(conf_matrix, class_names):
    mdict = compute_metrics(conf_matrix, class_names)
    scene_name = os.environ["SCENE_NAME"]
    result = {
        "scene_id": scene_name,
        "miou": mdict["miou"] * 100.0,
        "mrecall": np.mean(mdict["recall"]) * 100.0,
        "mprecision": np.mean(mdict["precision"]) * 100.0,
        "mf1score": np.mean(mdict["f1score"]) * 100.0,
        "fmiou": mdict["fmiou"] * 100.0,
    }

    print("----------------------------------------")
    pprint.pprint(result, indent=2)
    print("----------------------------------------")
        
    df_result = pd.DataFrame([result])
    
    return df_result

def save_results(save_path, conf_matrix, df_result):
    scene_name = os.environ["SCENE_NAME"]
    config_name = os.environ["CONFIG_NAME"]
    save_path = os.path.join(os.path.dirname(save_path), "metrics", config_name, scene_name, f"{scene_name}_conf_matrix.pkl")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    try:
        existing_df = pd.read_csv(save_path)
        updated_df = pd.concat([existing_df, df_result], ignore_index=True)
    except FileNotFoundError:
        updated_df = df_result

    updated_df.to_csv(save_path, index=False)
    
    pickle.dump(conf_matrix, open(save_path, "wb"))
    

def get_labelset_from_habitat():
    scene_name = os.environ["SCENE_NAME"]
    config_name = os.environ["CONFIG_NAME"]    
    data_dir = os.environ["DATASET_DIR_PATH"]

    labelset = []
    json_file = os.path.join(data_dir, config_name, scene_name, "embed_semseg_classes.json")
    with open(json_file, 'r') as f:
            data = json.load(f)
            if 'classes' in data:
                for class_info in data['classes']:
                    if 'name' in class_info:
                        labelset.append(class_info['name'].replace('_', ' '))
    return labelset


def evaluate(pred_ids, gt_ids, stdout=False, dataset='scannet_3d', save_path=None):
    if stdout:
        print('evaluating', gt_ids.size, 'points...')
    if 'scannet_3d' in dataset:
        CLASS_LABELS = SCANNET_LABELS_20
    elif 'matterport_3d_40' in dataset:
        CLASS_LABELS = MATTERPORT_LABELS_40
    elif 'matterport_3d_80' in dataset:
        CLASS_LABELS = MATTERPORT_LABELS_80
    elif 'matterport_3d_160' in dataset:
        CLASS_LABELS = MATTERPORT_LABELS_160
    elif 'matterport_3d' in dataset:
        CLASS_LABELS = MATTERPORT_LABELS_21
    elif 'nuscenes_3d' in dataset:
        CLASS_LABELS = NUSCENES_LABELS_16
    elif 'replica_cad' in dataset:
        CLASS_LABELS = REPLICA_CLASSES
    elif 'replica' in dataset:
        CLASS_LABELS = get_labelset_from_habitat()
    elif 'hm_3d' in dataset:
        CLASS_LABELS = get_labelset_from_habitat()
    else:
        raise NotImplementedError
  
    
    conf_matrix = confusion_matrix(gt_ids, pred_ids, labels=list(range(0, len(REPLICA_CLASSES))))
    df_result = get_metrics(conf_matrix, CLASS_LABELS)
    save_results(save_path, conf_matrix, df_result)