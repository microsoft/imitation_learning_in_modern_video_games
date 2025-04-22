# Plot bunch of MineRL results in one plot
# Input is bunch of directories.
#   We then search for all seeds (i.e. f"{directory_name}_seed_#/")

import json
from argparse import ArgumentParser
from collections import defaultdict
from pathlib import Path

import numpy as np
import seaborn as sns
import yaml
from scipy.stats import ttest_ind

sns.set_theme()

DEFAULT_SEEDS = [0, 1, 2]
EXPECTED_N_ROLLOUT_FILES = 3
ORIGINAL_FPS = 16
ORIGINAL_DURATION = 5 * 60  # 5 minutes

# We use .endswith to do the mapping
DIRECTORY_NAME_PART_TO_NICE_NAME = {
    "own_resnet_128": "ResNet (128) + Aug",
    "own_vit_custom_128": "ViT (128) + Aug",
    "own_vit_128": "ViT (128) + Aug",
    "impala_resnet_128_noaug": "Impala (128)",
    "impala_resnet_128": "Impala (128) + Aug",
    "resnet_128": "ResNet (128) + Aug",
    "resnet_128_noaug": "ResNet (128)",
    "resnet_256": "ResNet (256) + Aug",
    "resnet_256_noaug": "ResNet (256)",
    "vit_128": "ViT (128) + Aug",
    "vit_128_noaug": "ViT (128)",
    "vit_256": "ViT (256) + Aug",
    "vit_256_noaug": "ViT (256)",
    "vit_tiny": "ViT (Tiny) + Aug",
    "vit_tiny_noaug": "ViT (Tiny)",
    "dino_vits": "DINOv2 ViT-S/14",
    "dino_vitb": "DINOv2 ViT-B/14",
    "dino_vitl": "DINOv2 ViT-L/14",
    "clip_rn50": "CLIP RN50",
    "clip_vitb": "CLIP ViT-B/16",
    "clip_vitl": "CLIP ViT-L/16",
    "focal_large": "FocalNet Large",
    "focal_xlarge": "FocalNet XLarge",
    "focal_huge": "FocalNet Huge",
    "sd_vae": "StableDiffusion VAE",
}


def directory_name_to_nice_name(directory_name):
    for directory_name_part, nice_name in DIRECTORY_NAME_PART_TO_NICE_NAME.items():
        if directory_name.endswith(directory_name_part):
            return nice_name
    return directory_name


def get_kills_per_minute(rollout_file, config_file):
    """Get kills per minute"""
    # read config file as yaml
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)
    assert "fps" in config, f"Expected 'fps' in {config_file}"
    rollout_fps = config["fps"]

    if rollout_fps != ORIGINAL_FPS:
        assert (
            ORIGINAL_FPS % rollout_fps == 0
        ), f"Expected original fps to be divisible by rollout fps, but got {ORIGINAL_FPS} and {rollout_fps} for {rollout_file}"
        rollout_fps_ratio = ORIGINAL_FPS // rollout_fps
    else:
        rollout_fps_ratio = 1

    # Files are json files
    with open(rollout_file, "r") as f:
        rollout_data = json.load(f)
    times = rollout_data["time"]
    duration = times[-1] - times[0]

    # correct duration by factor of FPS (in case game was run at lower speed)
    effective_duration = duration / rollout_fps_ratio
    assert (
        abs(effective_duration - ORIGINAL_DURATION) < 5
    ), f"Expected effective duration to be close to {ORIGINAL_DURATION}, but got {effective_duration} for {rollout_file}"

    kills = rollout_data["kills"][-1]
    kills_per_minute = kills / effective_duration * 60
    return kills_per_minute


def get_times_and_cumulative_kills(rollout_file, config_file, interval=10):
    """
    Get cumulative kills over time
    :param rollout_file: Path to rollout file
    :param config_file: Path to config file
    :param num_bins: Number of bins to use
    :param interval: Get value every interval seconds
    :return: (times, cumulative_kills)
    """
    # read config file as yaml
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)
    assert "fps" in config, f"Expected 'fps' in {config_file}"
    rollout_fps = config["fps"]

    if rollout_fps != ORIGINAL_FPS:
        assert (
            ORIGINAL_FPS % rollout_fps == 0
        ), f"Expected original fps to be divisible by rollout fps, but got {ORIGINAL_FPS} and {rollout_fps} for {rollout_file}"
        rollout_fps_ratio = ORIGINAL_FPS // rollout_fps
    else:
        rollout_fps_ratio = 1

    # Files are json files
    with open(rollout_file, "r") as f:
        rollout_data = json.load(f)
    times = rollout_data["time"]
    duration = times[-1] - times[0]
    rel_times = np.array(times) - times[0]
    rel_times = rel_times[5:] / rollout_fps_ratio  # correct for rollout fps

    # sanity check
    effective_duration = duration / rollout_fps_ratio
    assert (
        abs(effective_duration - ORIGINAL_DURATION) < 5
    ), f"Expected effective duration to be close to {ORIGINAL_DURATION}, but got {effective_duration} for {rollout_file}"

    kills = rollout_data["kills"][5:]
    assert len(kills) == len(rel_times), f"Expected same number of kills and times, but got {len(kills)} and {len(rel_times)} for {rollout_file}"

    # Get relative times and kills every interval seconds
    filtered_rel_times = []
    filtered_kills = []
    interval_i = 0
    for time, kill in zip(rel_times, kills):
        if time > interval_i * interval:
            filtered_rel_times.append(time)
            filtered_kills.append(kill)
            interval_i += 1
    if len(filtered_rel_times) < ORIGINAL_DURATION / interval:
        filtered_rel_times.append(rel_times[-1])
        filtered_kills.append(kills[-1])
        print(f"[Warning] Expected {ORIGINAL_DURATION / interval} intervals, but got {len(filtered_rel_times)} for {rollout_file}")

    return filtered_rel_times, filtered_kills


def main(args):
    directory_to_seeds_to_rollout_results = defaultdict(dict)
    directory = Path(args.directory)
    for directory_path in directory.iterdir():
        # for directory_path in args.directories:
        if args.individual_seeds:
            directory_name = directory_path.name
            seed = 0
        else:
            try:
                directory_name, seed = directory_path.name.split("_seed_")
            except:
                directory_name, seed = directory_path.name.split("_seed")
            seed = int(seed)
        if seed not in DEFAULT_SEEDS:
            print(f"[Warning] Seed {seed} is not in {DEFAULT_SEEDS} for {directory_path}. Skipping")
            continue
        rollout_files = sorted(directory_path.glob("plugin_data*.json"))
        config_files = sorted(directory_path.glob("config*.yaml"))
        # rollout_files = sorted(glob.glob(os.path.join(directory_path, "plugin_data*.json")))
        # config_files = sorted(glob.glob(os.path.join(directory_path, "config*.yaml")))
        assert len(config_files) == len(
            rollout_files
        ), f"Expected same number of config and plugin data files, but got {len(config_files)} configs and {len(rollout_files)} "
        +f"plugin datas for {directory_path}"
        if len(rollout_files) != EXPECTED_N_ROLLOUT_FILES:
            print(
                f"[Warning] Expected {EXPECTED_N_ROLLOUT_FILES} rollout files, but found {len(rollout_files)} in "
                + f"{directory_path}. Skipping."
            )
            continue
        rollout_progression_result = [
            get_kills_per_minute(rollout_file, config_file) for rollout_file, config_file in zip(rollout_files, config_files)
        ]
        directory_to_seeds_to_rollout_results[directory_name][seed] = rollout_progression_result

    nice_name_to_kills_per_minute = {}

    for directory_name, seeds_to_rollout_results in directory_to_seeds_to_rollout_results.items():
        if args.individual_seeds:
            nice_name = directory_name
        else:
            nice_name = directory_name_to_nice_name(directory_name)
        all_rollout_results = []
        for rollout_results in seeds_to_rollout_results.values():
            rollout_results = np.array(rollout_results)
            all_rollout_results.append(rollout_results)
        if len(all_rollout_results) != len(DEFAULT_SEEDS):
            print(f"[Warning] Expected {len(DEFAULT_SEEDS)} seeds, but got {len(all_rollout_results)} for {directory_name}.")
        all_rollout_results = np.array(all_rollout_results)
        nice_name_to_kills_per_minute[nice_name] = {
            "kills_per_minute": all_rollout_results,
            "mean": np.mean(all_rollout_results),
            "std": np.std(all_rollout_results),
            "std_err": np.std(all_rollout_results) / np.sqrt(len(all_rollout_results)),
        }

    # Print out the overall success ratio and sort from highest to lowest
    print("Overall kills-per-minute (mean, std, std_err):")
    nice_name_and_kills_per_minute = sorted(nice_name_to_kills_per_minute.items(), key=lambda x: x[1]["mean"], reverse=True)
    for nice_name, kills_per_minute in nice_name_and_kills_per_minute:
        # Print as percentage and std err
        print(f"\t{nice_name:>30}: {kills_per_minute['mean']:>5.2f} ({kills_per_minute['std']:>4.2f}) ({kills_per_minute['std_err']:>4.2f})")

    # Print out t-test results for all models against each other
    if args.t_test:
        print("T-test results:")
        for a_model_name, a_kills_per_minute in nice_name_and_kills_per_minute:
            for b_model_name, b_kills_per_minute in nice_name_and_kills_per_minute:
                if a_model_name == b_model_name:
                    continue
                _, p_value = ttest_ind(
                    a_kills_per_minute["kills_per_minute"].ravel(),
                    b_kills_per_minute["kills_per_minute"].ravel(),
                    equal_var=False,
                )
                significance = ""
                if p_value < 0.05:
                    significance = "*"
                if p_value < 0.01:
                    significance = "**"
                if p_value < 0.001:
                    significance = "***"
                print(f"\t{a_model_name:>25} vs {b_model_name:<25}: {p_value:>5.4f} {significance}")

    # Plot lineplot for kills over time
    directory_to_seeds_to_times_and_kills = defaultdict(dict)
    if args.plot:
        for directory_path in directory.iterdir():
            if args.individual_seeds:
                directory_name = directory_path.name
                seed = 0
            else:
                directory_name, seed = directory_path.name.split("_seed_")
                seed = int(seed)
            if seed not in DEFAULT_SEEDS:
                print(f"[Warning] Seed {seed} is not in {DEFAULT_SEEDS} for {directory_path}. Skipping")
                continue
            rollout_files = sorted(directory_path.glob("plugin_data*.json"))
            config_files = sorted(directory_path.glob("config*.yaml"))
            assert len(config_files) == len(
                rollout_files
            ), f"Expected same number of config and plugin data files, but got {len(config_files)} configs and {len(rollout_files)} "
            + f"plugin datas for {directory_path}"
            if len(rollout_files) != EXPECTED_N_ROLLOUT_FILES:
                print(
                    f"[Warning] Expected {EXPECTED_N_ROLLOUT_FILES} rollout files, but found {len(rollout_files)} "
                    + f"in {directory_path}. Skipping."
                )
                continue
            times_and_kills = [
                get_times_and_cumulative_kills(rollout_file, config_file) for rollout_file, config_file in zip(rollout_files, config_files)
            ]
            times = np.array([times for (times, _) in times_and_kills])
            kills = np.array([kills for (_, kills) in times_and_kills])
            directory_to_seeds_to_times_and_kills[directory_name][seed] = times, kills

        nice_name_to_times_and_mean_kills_std_kills = {}

        for directory_name, seeds_to_rollout_results in directory_to_seeds_to_times_and_kills.items():
            if args.individual_seeds:
                nice_name = directory_name
            else:
                nice_name = directory_name_to_nice_name(directory_name)
            all_rollout_times = []
            all_rollout_kills = []
            for times, kills in seeds_to_rollout_results.values():
                all_rollout_times.append(times)
                all_rollout_kills.append(kills)
            if len(all_rollout_times) != len(DEFAULT_SEEDS):
                print(f"[Warning] Expected {len(DEFAULT_SEEDS)} seeds, but got {len(all_rollout_times)} for {directory_name}.")
            if len(all_rollout_times) == 1:
                # average/ std over runs per model
                times = np.mean(all_rollout_times[0], axis=0)
                kills_mean = np.mean(all_rollout_kills[0], axis=0)
                kills_std = np.std(all_rollout_kills[0], axis=0)
            else:
                # average over runs per seed and then average/ std over seeds
                times = np.array(all_rollout_times).mean(axis=0).mean(axis=0)
                kills = np.array(all_rollout_kills).mean(axis=1)
                kills_mean = kills.mean(axis=0)
                kills_std = kills.std(axis=0)
            nice_name_to_times_and_mean_kills_std_kills[nice_name] = (times, kills_mean, kills_std)

        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        for nice_name, (times, means, stds) in nice_name_to_times_and_mean_kills_std_kills.items():
            ax.plot(times, means, label=nice_name)
            ax.fill_between(times, means - stds, means + stds, alpha=0.2)
        ax.set_xlabel("Model")
        ax.set_ylabel("Kills per minute")
        ax.set_title("Kills per minute for different models")
        # legend above plot
        ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.05), ncol=4)
        plt.show()
        plt.savefig("kills_over_time.pdf", bbox_inches="tight")


if __name__ == "__main__":
    parser = ArgumentParser()
    # parser.add_argument("--directories", nargs="+", required=True, type=str, help="Directories to plot. They should end with '_seed_#'")
    parser.add_argument(
        "--directory",
        required=True,
        type=str,
        help="Directory containing results to plot as subdirectories ending with '_seed_#'",
    )
    parser.add_argument("--individual_seeds", action="store_true", help="Report results for individual seeds instead")
    parser.add_argument("--plot", action="store_true", help="Plot lineplot for kills over time")
    parser.add_argument("--t_test", action="store_true", help="Perform t-test of all models vs all models, and print out results.")
    args = parser.parse_args()
    main(args)
