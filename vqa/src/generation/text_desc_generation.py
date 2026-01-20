import argparse
import base64
import os
import shutil
import time
import logging

import numpy as np
from tqdm import tqdm

from src.config import Configuration
from src.utils.api import post_with_retry
from src.utils.json_utils import save_json

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s %(message)s"
)
logger = logging.getLogger(__name__)

def encode_image(image_path):
    """Reads and encodes an image as base64."""
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode("utf-8")


def analyze_trajectory(traj_path):
    """Analyzes camera trajectory; returns positions, orientations, adaptive frame_step, and scene size."""
    positions, orientations = [], []
    with open(traj_path, 'r') as file:
        for line in file:
            vals = list(map(float, line.split()))
            # positions (x, y, z)
            positions.append(vals[3:6])
            # orientation data
            orientations.append(vals[:3] + vals[6:9] + vals[9:12])

    positions = np.array(positions)
    min_pos, max_pos = positions.min(axis=0), positions.max(axis=0)
    scene_size = np.linalg.norm(max_pos - min_pos)

    # Adaptive step
    frame_step = max(50, int(scene_size * 20))
    return np.array(positions), np.array(orientations), frame_step, scene_size


def compute_view_difference(pos1, pos2, ori1, ori2, pos_threshold=0.5, angle_threshold=10):
    """
    Checks if two camera frames are 'too similar' based on position + orientation.
    """
    pos_diff = np.linalg.norm(pos1 - pos2)

    # Normalize orientation vectors
    ori1_norm = np.linalg.norm(ori1[:3])
    ori2_norm = np.linalg.norm(ori2[:3])
    if ori1_norm == 0 or ori2_norm == 0:
        return False  # fallback if something is off

    ori1_unit = ori1[:3] / ori1_norm
    ori2_unit = ori2[:3] / ori2_norm

    cos_angle = np.dot(ori1_unit, ori2_unit)
    angle = np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))

    return (pos_diff < pos_threshold) and (angle < angle_threshold)


def select_frames(scene_dir, traj_path, vqa_dir):
    """
    Selects frames from scene_dir, skipping similar viewpoints,
    and saves them into vqa_dir/true_frames and vqa_dir/false_frames.
    """
    os.makedirs(vqa_dir, exist_ok=True)

    true_frames_dir = os.path.join(vqa_dir, "true_frames")
    false_frames_dir = os.path.join(vqa_dir, "false_frames")
    os.makedirs(true_frames_dir, exist_ok=True)
    os.makedirs(false_frames_dir, exist_ok=True)

    all_frames = sorted(f for f in os.listdir(scene_dir) if f.endswith(".jpg"))
    positions, orientations, frame_step, scene_size = analyze_trajectory(traj_path)

    logger.info("Scene size: %.2f; using frame step = %d", scene_size, frame_step)

    selected_frames = []
    final_positions = []
    final_orientations = []

    for i in range(0, len(all_frames), frame_step):
        if i >= len(positions):
            break

        frame = all_frames[i]
        pos1 = positions[i]
        ori1 = orientations[i]

        if any(compute_view_difference(pos1, p, ori1, o) for p, o in zip(final_positions, final_orientations)):
            logger.debug("Frame %s excluded (similar to previous).", frame)
            shutil.copy(os.path.join(scene_dir, frame), os.path.join(false_frames_dir, frame))
            continue

        selected_frames.append(frame)
        final_positions.append(pos1)
        final_orientations.append(ori1)
        shutil.copy(os.path.join(scene_dir, frame), os.path.join(true_frames_dir, frame))

    logger.info("Selected %d frames out of %d total.", len(selected_frames), len(all_frames))

    return selected_frames, np.array(final_positions), np.array(final_orientations)


def generate_description_gemini(config, image_path):
    time.sleep(2)
    api_url = f"{config.url}/{config.vlm}:generateContent?key={config.gemini_api_key}"
    prompt = config.gemini_scene_prompt.replace("{rejection_keyword}", config.rejection_keyword)

    payload = {
        "contents": [{
            "parts": [
                {"text": prompt},
                {"inlineData": {
                    "mimeType": "image/jpeg",
                    "data": encode_image(image_path)
                }}
            ]
        }]
    }

    response = post_with_retry(api_url, payload)

    if response is None:
        return config.rejection_keyword

    return response.json().get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "")


def save_scene_data(output_path,
                    scene_name,
                    selected_frames,
                    positions=None,
                    orientations=None,
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
        if rejection_keyword and rejection_keyword in desc:
            logger.debug("Frame %s excluded (scene blocked).", frame)
            continue

        entry = {
            "frame": frame,
            "description": desc
        }

        # Include camera info if available
        if positions is not None and orientations is not None:
            entry["camera"] = {
                "position": positions[idx].tolist(),
                "orientation": orientations[idx].tolist()
            }

        scene_data["parameters"].append(entry)

    save_json(scene_data, output_path)
    logger.info("Scene description saved to: %s", output_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path", help="Path to the YAML config.")
    parser.add_argument("--scene", required=True, help="Name of the scene folder.")
    parser.add_argument("--manual", action="store_true",
                        help="Use manual frames instead of automatic selection.")
    args = parser.parse_args()

    config = Configuration(yaml_path=args.config_path)
    base = config.base_scenes_dir
    scene = args.scene
    scene_dir = os.path.join(base, scene, "results")
    traj_path = os.path.join(base, scene, "traj.txt")
    vqa_dir = os.path.join(base, scene, "vqa")

    os.makedirs(vqa_dir, exist_ok=True)

    if args.manual or not os.path.isfile(traj_path):
        manual_dir = os.path.join(vqa_dir, "manual_frames")
        if not os.path.isdir(manual_dir):
            raise FileNotFoundError(f"Manual frames directory not found: {manual_dir}")
        # use frames placed manually
        selected_frames = sorted(f for f in os.listdir(manual_dir) if f.endswith(".jpg"))
        positions = orientations = None
        frames_source_dir = manual_dir
        logger.info("Using manual frames from %s", manual_dir)
    else:
        # automatic selection via trajectory
        selected_frames, positions, orientations = select_frames(scene_dir, traj_path, vqa_dir)
        frames_source_dir = scene_dir

    descriptions = {}
    for frame in tqdm(selected_frames, desc="Generating descriptions"):
        image_path = os.path.join(frames_source_dir, frame)
        descriptions[frame] = generate_description_gemini(config, image_path)

    output_json = os.path.join(vqa_dir, f"{args.scene}_descriptions.json")
    save_scene_data(
        output_json,
        args.scene,
        selected_frames,
        positions,
        orientations,
        descriptions,
        config.rejection_keyword
    )


if __name__ == "__main__":
    main()
