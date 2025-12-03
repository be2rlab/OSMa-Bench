import os
import multiprocessing as mp
import numpy as np
import plyfile
import torch
import json


def load_gt_pointcloud_ply(gt_pc_path, scene_name_underscore):
    semantic_info = json.load(open("/data/gt/replica/" + scene_name_underscore + "/habitat/info_semantic.json"))

    plydata = plyfile.PlyData.read(gt_pc_path)

    object_to_class_mapping = {obj["id"]: obj["class_id"] for obj in semantic_info["objects"]}

    # Extract vertex data
    vertices = np.vstack([plydata["vertex"]["x"], plydata["vertex"]["y"], plydata["vertex"]["z"]]).T
    
    # Extract object_id and normalize it to use as color
    face_vertices = plydata["face"]["vertex_indices"]
    object_ids = plydata["face"]["object_id"]
    vertices1 = []
    object_ids1 = []
    for i, face in enumerate(face_vertices):
        vertices1.append(vertices[face])
        object_ids1.append(np.repeat(object_ids[i], len(face)))
    vertices1 = np.vstack(vertices1)
    object_ids1 = np.hstack(object_ids1)
    
    gt_xyz = np.array(vertices1)

    # semantic colors
    class_ids = []
    for object_id in object_ids1:
        if object_id in object_to_class_mapping.keys():
            class_ids.append(object_to_class_mapping[object_id])
        else:
            class_ids.append(0)
    gt_class = np.array(class_ids)
    
    return gt_xyz, gt_class


def process_one_scene(fn):
    '''process one scene.'''

    scene_name_underscore = fn.split('/')[-3]
    scene_name = scene_name_underscore.replace('_', '')
    gt_xyz, gt_class = load_gt_pointcloud_ply(fn, scene_name_underscore)

    torch.save((gt_xyz, 0, gt_class),
            os.path.join(out_dir,  scene_name + '.pth'))
    print(fn)


def process_txt(filename):
    with open(filename) as file:
        lines = file.readlines()
        lines = [line.rstrip() for line in lines]
    return lines

#! YOU NEED TO MODIFY THE FOLLOWING
scene_list = ['office0', 'office1', 'office2', 'office3',
              'office4', 'room0', 'room1', 'room2']
scene_names_underscore = ['office_0', 'office_1', 'office_2', 'office_3',
              'office_4', 'room_0', 'room_1', 'room_2']
#####################################
out_dir = '/data/openscene/replica/replica_3d'
in_path = '/data/gt/replica' # downloaded original replica data
#####################################

os.makedirs(out_dir, exist_ok=True)

files = []
for scene, scene_name_underscore in zip(scene_list, scene_names_underscore):
    files.append(os.path.join(in_path, scene_name_underscore, 'habitat/mesh_semantic.ply'))

p = mp.Pool(processes=mp.cpu_count())
p.map(process_one_scene, files)
p.close()
p.join()