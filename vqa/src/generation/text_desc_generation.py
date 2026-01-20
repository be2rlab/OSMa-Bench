import argparse
import logging
import os

import numpy as np
import scipy.spatial.transform as sst
from tqdm import tqdm

from src.models import create_foundation_model
from src.utils.api import post_with_retry
from src.utils.config_loader import load_configuration
from src.utils.json_utils import save_json


logger = logging.getLogger(__name__)


def get_parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path", help="Path to the YAML config.")
    parser.add_argument("--scene", required=True, help="Name of the scene folder.")
    parser.add_argument("--manual", action="store_true",
                        help="Use manual frames instead of automatic selection.")
    
    args = parser.parse_args()
    
    return args


def get_pathes(base_dir, scene_name):
    results_dir = os.path.join(base_dir, scene_name, "results")
    traj_path   = os.path.join(base_dir, scene_name, "traj.txt")
    vqa_dir     = os.path.join(base_dir, scene_name, "vqa")

    os.makedirs(vqa_dir, exist_ok=True)
    
    return results_dir, traj_path, vqa_dir


def load_traj(traj_path):
    traj = np.loadtxt(traj_path)
    matrices = traj.reshape(-1, 4, 4)
    
    return matrices


def check_view_similarity(lhs_pose, rhs_pose, pos_threshold, angle_threshold):
    """
    Checks if two camera frames are 'too similar' based on position + orientation.
    """
    pos_diff = np.linalg.norm(lhs_pose[:3, -1] - rhs_pose[:3, -1])
    
    lhs_rot = sst.Rotation.from_matrix(lhs_pose[:3, :3])
    rhs_rot = sst.Rotation.from_matrix(rhs_pose[:3, :3])

    relative_rot = lhs_rot * rhs_rot.inv()
    angle_diff = relative_rot.magnitude()     

    return (pos_diff < pos_threshold) and (angle_diff < angle_threshold)


def select_frames(frames_dir, traj_path, pos_threshold, angle_threshold):
    """
    Selects frames from scene_dir, skipping similar viewpoints.
    """
    all_frames = sorted(
        os.path.join(frames_dir, f) for f in os.listdir(frames_dir) if f.endswith(".jpg")
    )
    camera_poses = load_traj(traj_path)
    assert len(all_frames) == camera_poses.shape[0]
    
    # trans = camera_poses[:, :3, -1] 
    # scene_size = np.linalg.norm(trans.min(axis=0) - trans.max(axis=0))
    # frame_step = max(50, int(scene_size * 20)) # Adaptive step
    # frame_step = 1 # to remove

    # logger.info("Scene size: %.2f; using frame step = %d", scene_size, frame_step)

    selected_frames = []
    final_poses = []

    for i in range(0, len(all_frames)):
        frame = all_frames[i]
        cur_pose = camera_poses[i]

        if any(check_view_similarity(cur_pose, pose, pos_threshold, angle_threshold) for pose in final_poses):
            logger.debug("Frame %s excluded (similar to previous).", frame)
            continue

        selected_frames.append(frame)
        final_poses.append(cur_pose)

    logger.info("Selected %d frames out of %d total.", len(selected_frames), len(all_frames))

    return selected_frames, np.array(final_poses)


def get_frames_pathes(frames_dir=None, traj_path=None, manual_dir=None, pos_threshold=None, angle_threshold=None):
    if manual_dir is not None:       
        if not os.path.isdir(manual_dir):
            raise FileNotFoundError(f"Manual selected frames directory not found: {manual_dir}")
        
        selected_frames = sorted(
            os.path.join(manual_dir, f) for f in os.listdir(manual_dir) if f.endswith(".jpg")
        )
        
        poses = None
        
        logger.info("Using manual frames from %s", manual_dir)
    elif frames_dir is not None or traj_path is not None:           
        # automatic selection via trajectory
        selected_frames, poses = select_frames(frames_dir, traj_path, pos_threshold, angle_threshold)
        
        logger.info("Frames selection is completed")
    else:
        raise ValueError("frames_dir and traj_path or manual_dir must not be None")
        
    return selected_frames, poses


def describe_image(foundation_model, image_path, prompt, rejection_keyword):
    prompt = prompt.replace("{rejection_keyword}", rejection_keyword)

    answer = post_with_retry(foundation_model, prompt=prompt, images=[image_path])

    if answer is None:
        return rejection_keyword

    return answer


def generate_descriptions(foundation_model, selected_frames, prompt, rejection_keyword):
    descriptions = {}
    for image_path in tqdm(selected_frames, desc="Generating descriptions"):
        descriptions[image_path] = describe_image(
            foundation_model = foundation_model, 
            image_path = image_path,
            prompt = prompt, 
            rejection_keyword = rejection_keyword
        )
        
    return descriptions


def save_scene_data(output_path,
                    scene_name,
                    selected_frames,
                    poses=None,
                    descriptions=None,
                    rejection_keyword=None):
    """
    Save scene descriptions (and optional camera data) to JSON.
    If positions and orientations are provided, include a "camera" block
    for each frame; otherwise only save "frame" and "description".
    """
    scene_data = {"scene_name": scene_name, "parameters": []}

    for idx, frame in enumerate(selected_frames):
        desc = descriptions.get(frame, rejection_keyword)
        if rejection_keyword is not None and rejection_keyword in desc:
            logger.debug("Frame %s excluded (scene blocked).", frame)
            continue

        entry = {
            "frame": frame,
            "description": desc.strip()
        }

        if poses is not None:
            entry["camera"] = {
                "position": poses[idx, :3, -1].tolist(),
                "orientation": sst.Rotation.from_matrix(poses[idx, :3, :3]).as_quat().tolist()
            }

        scene_data["parameters"].append(entry)

    save_json(scene_data, output_path)
    logger.info("Scene description saved to: %s", output_path)


def main():  
    args = get_parser_args()
    
    config = load_configuration(yaml_path=args.config_path)
    
    results_dir, traj_path, vqa_dir = get_pathes(
        base_dir=config.data_dir, scene_name=args.scene
    )
        
    model_conf = config.foundation_model
    foundation_model = create_foundation_model(
        api_name = model_conf.api_name,
        api_key = model_conf.api_key,
        model = model_conf.model,
        llm = model_conf.llm,
        lvlm = model_conf.lvlm,
        )
    
    gen_config = config.description_generation
    selected_frames, poses = get_frames_pathes(
        frames_dir=results_dir, 
        traj_path=traj_path,
        pos_threshold = gen_config.pos_threshold, 
        angle_threshold = gen_config.angle_threshold
    )
    
    descriptions = generate_descriptions(
        foundation_model = foundation_model, 
        selected_frames = selected_frames, 
        prompt = gen_config.prompt, 
        rejection_keyword = gen_config.rejection_keyword
    )
    
    save_scene_data(
        output_path = os.path.join(vqa_dir, f"{args.scene}_descriptions.json"),
        scene_name = args.scene,
        selected_frames = selected_frames,
        poses = poses,
        descriptions = descriptions,
        rejection_keyword = gen_config.rejection_keyword
    )


if __name__ == "__main__":
    main()
