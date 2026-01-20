"""
CLI entry point for the pipeline and its subcommands.
"""
import argparse
import subprocess
import os
import sys


def generate_for_scene(config_path: str, scene: str, manual: bool):
    """
    Run text description, QA generation and QA validation for a single scene.
    """
    # Text description
    cmd = [sys.executable, '-m', 'src.generation.text_desc', config_path, '--scene', scene]
    if manual:
        cmd.append('--manual')
    subprocess.run(cmd, check=True)

    # QA generation
    cmd = [sys.executable, '-m', 'src.generation.qa_gen', config_path, '--scene', scene]
    if manual:
        cmd.append('--manual')
    subprocess.run(cmd, check=True)

    # QA validation
    cmd = [sys.executable, '-m', 'src.validation.qa_validate', config_path, '--scene', scene]
    if manual:
        cmd.append('--manual')
    subprocess.run(cmd, check=True)


def answer_for_scene(config_path: str, scene: str, graphs_dir: str, output_dir: str):
    """
    Answer validated questions using the scene graph for a single scene.
    """
    vqa_file = os.path.join('data', scene, 'vqa', f"{scene}_validated_questions.json")
    graph_file = os.path.join(graphs_dir, scene, 'scene_graph.json')
    output_file = os.path.join(output_dir, f"{scene}_answered.json")

    if not os.path.isfile(vqa_file) or not os.path.isfile(graph_file):
        print(f"[!] Skipping: missing VQA or graph for scene '{scene}'")
        return

    cmd = [sys.executable, '-m', 'src.evaluation.scene_graph_ans', config_path,
           '--questions', vqa_file,
           '--graph', graph_file,
           '--output', output_file]
    subprocess.run(cmd, check=True)


def evaluate_all(config_path: str):
    """
    Run graph evaluation across all answered files.
    """
    cmd = [sys.executable, '-m', 'src.evaluation.graphs_eval', config_path]
    subprocess.run(cmd, check=True)


def pipeline(args):
    """
    Full pipeline: generation, answering, evaluation.
    """
    scenes = [d for d in os.listdir(args.data_dir) if os.path.isdir(os.path.join(args.data_dir, d))]
    for scene in scenes:
        print(f"=== Processing scene: {scene} ===")
        generate_for_scene(args.config, scene, args.manual)

    for scene in scenes:
        print(f"=== Answering for scene: {scene} ===")
        answer_for_scene(args.config, scene, args.graphs_dir, args.output_dir)

    print("=== Evaluating all results ===")
    evaluate_all(args.config)


def main():
    parser = argparse.ArgumentParser(description='Scene QA Pipeline CLI')
    parser.add_argument('-c', '--config', required=True, help='Path to YAML config file')
    parser.add_argument('--data-dir', default='data', help='Base directory for scenes')
    parser.add_argument('--graphs-dir', default='graphs', help='Base directory for scene graphs')
    parser.add_argument('--output-dir', default='output', help='Directory to save answered JSONs')
    parser.add_argument('--manual', action='store_true', help='Use manual frame selection instead of trajectory')

    subparsers = parser.add_subparsers(dest='command', required=True)

    # pipeline command
    subparsers.add_parser('pipeline', help='Run full pipeline (generate, answer, evaluate)')

    # generate command
    sp_gen = subparsers.add_parser('generate', help='Run generation stages for one scene')
    sp_gen.add_argument('--scene', required=True, help='Scene name to process')

    # answer command
    sp_ans = subparsers.add_parser('answer', help='Run scene graph answering for one scene')
    sp_ans.add_argument('--scene', required=True, help='Scene name to answer')

    # evaluate command
    subparsers.add_parser('evaluate', help='Run evaluation of all answered outputs')

    args = parser.parse_args()

    if args.command == 'pipeline':
        pipeline(args)
    elif args.command == 'generate':
        generate_for_scene(args.config, args.scene, args.manual)
    elif args.command == 'answer':
        answer_for_scene(args.config, args.scene, args.graphs_dir, args.output_dir)
    elif args.command == 'evaluate':
        evaluate_all(args.config)


if __name__ == '__main__':
    main()
