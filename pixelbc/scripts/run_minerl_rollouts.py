import subprocess
from argparse import ArgumentParser
from multiprocessing import Pool
from pathlib import Path

import yaml

CHECKPOINT_FILE = "last.ckpt"
OUTPUT_DIR = "minerl_rollouts"
NUM_ROLLOUTS = 100
NUM_PARALLEL_RUNS = 4
FPS = 10
COMPLETED_STEPS = 60 * FPS  # 1 minute
CONFIG_PREFIX = "config_"
ACTIONS_PREFIX = "actions_"
PLUGIN_PREFIX = "plugin_data_"
VIDEO_PREFIX = "video_"


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("inputs", nargs="+")
    parser.add_argument("--output_dir", default=OUTPUT_DIR)
    parser.add_argument("--num_rollouts", type=int, default=NUM_ROLLOUTS)
    parser.add_argument("--parallel_runs", type=int, default=NUM_PARALLEL_RUNS)
    return parser.parse_args()


def get_model_name_from_checkpoint(checkpoint_path):
    return str(checkpoint_path).split("/")[-3]


def get_rollout_idx_from_output_file(output_file):
    return int(str(output_file.stem).split("_")[-1])


def get_completed_and_incomplete_rollouts(output_dir, model_name):
    model_output_dir = Path(output_dir) / model_name
    assert model_output_dir.exists(), f"Output directory {model_output_dir} does not exist"

    video_files = list(model_output_dir.glob(f"{VIDEO_PREFIX}*.mp4"))
    config_files = list(model_output_dir.glob(f"{CONFIG_PREFIX}*.yaml"))
    actions_files = list(model_output_dir.glob(f"{ACTIONS_PREFIX}*.yaml"))
    plugin_files = list(model_output_dir.glob(f"{PLUGIN_PREFIX}*.yaml"))

    completed_config_files = []
    for config_file in config_files:
        with open(config_file, "r") as f:
            config = yaml.safe_load(f)
        if config["num_steps"] == COMPLETED_STEPS:
            completed_config_files.append(config_file)

    video_file_indices = set([get_rollout_idx_from_output_file(video_file) for video_file in video_files])
    config_file_indices = set([get_rollout_idx_from_output_file(config_file) for config_file in completed_config_files])
    actions_file_indices = set([get_rollout_idx_from_output_file(actions_file) for actions_file in actions_files])
    plugin_file_indices = set([get_rollout_idx_from_output_file(plugin_file) for plugin_file in plugin_files])

    completed_rollout_indices = (
        video_file_indices.intersection(config_file_indices).intersection(actions_file_indices).intersection(plugin_file_indices)
    )
    incomplete_rollout_indices = video_file_indices - completed_rollout_indices
    return completed_rollout_indices, incomplete_rollout_indices


def cleanup_incomplete_rollouts(output_dir, model_name, incomplete_rollout_indices):
    model_output_dir = Path(output_dir) / model_name
    for idx in incomplete_rollout_indices:
        for prefix in [VIDEO_PREFIX, CONFIG_PREFIX, ACTIONS_PREFIX, PLUGIN_PREFIX]:
            if prefix == VIDEO_PREFIX:
                file = model_output_dir / f"{prefix}{idx}.mp4"
            else:
                file = model_output_dir / f"{prefix}{idx}.yaml"

            if file.exists():
                file.unlink()


def check_rollouts_complete_and_cleanup(output_dir, model_name, num_rollouts):
    completed_rollout_indices, incomplete_rollout_indices = get_completed_and_incomplete_rollouts(output_dir, model_name)
    last_completed_rollout_idx = max(completed_rollout_indices) + 1 if len(completed_rollout_indices) > 0 else 0

    print(f"{model_name}: {last_completed_rollout_idx}/{num_rollouts} rollouts completed ({len(incomplete_rollout_indices)} incomplete)")
    if last_completed_rollout_idx == num_rollouts:
        assert len(incomplete_rollout_indices) == 0, f"Found incomplete rollouts: {incomplete_rollout_indices}"
        return True
    else:
        if len(incomplete_rollout_indices) > 0:
            cleanup_incomplete_rollouts(output_dir, model_name, incomplete_rollout_indices)
        return False


def run_model_rollouts(input_dir, output_dir, num_rollouts):
    input_dir = Path(input_dir)
    assert input_dir.exists(), f"Input directory {input_dir} does not exist"
    output_dir = Path(output_dir)

    checkpoint_path = input_dir / CHECKPOINT_FILE
    model_name = get_model_name_from_checkpoint(checkpoint_path)
    output_model_dir = output_dir / model_name
    output_model_dir.mkdir(parents=True, exist_ok=True)

    while not check_rollouts_complete_and_cleanup(output_dir, model_name, num_rollouts):
        print(f"Running rollouts for {model_name}")
        subprocess.run(
            [
                "xvfb-run",
                "-a",
                "python",
                "online_rollout/rollout.py",
                "-g",
                "minerl",
                "-ckpt",
                checkpoint_path,
                "-fps",
                str(FPS),
                "-p",
                output_dir / model_name,
            ]
        )


def main():
    args = parse_args()

    assert len(args.inputs) > 0, "No input directories provided"

    with Pool(args.parallel_runs) as p:
        p.starmap(run_model_rollouts, [(input_dir, args.output_dir, args.num_rollouts) for input_dir in args.inputs])


if __name__ == "__main__":
    main()
