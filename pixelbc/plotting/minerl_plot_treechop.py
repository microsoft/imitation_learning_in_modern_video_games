# Plot bunch of MineRL results in one plot
# Input is bunch of directories.
#   We then search for all seeds (i.e. f"{directory_name}_seed_#/")

import glob
import json
import os
from argparse import ArgumentParser
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import ttest_ind

sns.set_theme()

DEFAULT_SEEDS = [0, 1, 2]
DEFAULT_MAX_TREECHOPS = 5
EXPECTED_N_ROLLOUT_FILES = 100

# We use .endswith to do the mapping
DIRECTORY_NAME_PART_TO_NICE_NAME = {
    "impala_resnet_128": "Impala ResNet (128) + Aug",
    "impala_resnet_128_noaug": "Impala ResNet (128)",
    "own_resnet_128": "ResNet (128) + Aug",
    "own_resnet_128_noaug": "ResNet (128)",
    "own_resnet_256_noaug": "ResNet (256)",
    "own_resnet_256": "ResNet (256) + Aug",
    "own_vit_custom_128_noaug": "ViT (128)",
    "own_vit_custom_128": "ViT (128) + Aug",
    "own_vit_custom_256_noaug": "ViT (256)",
    "own_vit_custom_256": "ViT (256) + Aug",
    "own_vit_tiny_noaug": "Tiny ViT",
    "own_vit_tiny": "Tiny ViT + Aug",
    "clip_rn50": "CLIP ResNet50",
    "clip_vitb16": "CLIP ViT-B/16",
    "clip_vitl14": "CLIP ViT-L/14",
    "dino_vitb": "DINOv2 ViT-B/14",
    "dino_vitl": "DINOv2 ViT-L/14",
    "dino_vits": "DINOv2 ViT-S/14",
    "focal_huge": "FocalNet Huge FL4",
    "focal_large": "FocalNet Large FL4",
    "focal_xlarge": "FocalNet X-Large FL4",
    "sd_vae": "Stable Diffusion VAE",
    "dino_vitb_10percent": "DINOv2 ViT-B/14 (10%)",
    "dino_vitl_10percent": "DINOv2 ViT-L/14 (10%)",
    "dungeons_minerl_own_vit_custom_128_10percent": "ViT (128) + Aug (10%)",
    "minerl_rollouts/dungeons_minerl_own_vit_tiny_10percent": "Tiny ViT + Aug (10%)",
    "dungeons_minerl_own_vit_tiny_noaug_10percent": "Tiny ViT (10%)",
    "dungeons_minerl_own_vit_custom_256_noaug_10percent": "ViT (256) (10%)",
}


def directory_name_to_nice_name(directory_name):
    for directory_name_part, nice_name in DIRECTORY_NAME_PART_TO_NICE_NAME.items():
        if directory_name.endswith(directory_name_part):
            return nice_name
    return directory_name


def get_num_treechops(rollout_file):
    """Get number of tree logs chopped (not necessirely collected) in the rollout"""
    # Files are json files, they were just mistakenly called yaml
    with open(rollout_file, "r") as f:
        rollout_data = json.load(f)
    last_mine_block = rollout_data["mine_block"][-1]
    num_logs_mined = 0
    for item_name, item_count in last_mine_block.items():
        if "_log" in item_name:
            num_logs_mined += item_count
    return num_logs_mined


def main(args):
    directory_to_seeds_to_rollout_results = defaultdict(dict)
    for directory_path in args.directories:
        if args.individual_seeds:
            directory_name = directory_path
            seed = 0
        else:
            directory_name, seed = directory_path.split("_seed_")
            if seed.endswith("_10percent"):
                seed, name_postfix = seed.split("_")
                directory_name = f"{directory_name}_{name_postfix}"
            seed = int(seed)
        if seed not in DEFAULT_SEEDS:
            print(f"[Warning] Seed {seed} is not in {DEFAULT_SEEDS} for {directory_path}. Skipping")
            continue
        rollout_files = sorted(glob.glob(os.path.join(directory_path, "plugin_data*.yaml")))
        if len(rollout_files) != EXPECTED_N_ROLLOUT_FILES:
            print(f"[Warning] Expected {EXPECTED_N_ROLLOUT_FILES} rollout files, but found {len(rollout_files)} in {directory_path}. Skipping.")
            continue
        rollout_pregression_results = [get_num_treechops(rollout_file) for rollout_file in rollout_files]
        directory_to_seeds_to_rollout_results[directory_name][seed] = rollout_pregression_results

    # Show plot with x-axis the number of treechops, and y-axis the rate at which we get that number of treechops per directory
    # Do mean and standard deviation over seeds
    fig = plt.figure(figsize=(8, 6))
    ax = plt.gca()

    nice_name_to_mean_treechop = {}
    nice_name_to_overall_success_ratio = {}

    for directory_name, seeds_to_rollout_results in directory_to_seeds_to_rollout_results.items():
        if args.individual_seeds:
            nice_name = directory_name
        else:
            nice_name = directory_name_to_nice_name(directory_name)
        reach_progression_over_seeds = []
        all_rollout_results = []
        for rollout_results in seeds_to_rollout_results.values():
            rollout_results = np.array(rollout_results)
            all_rollout_results.append(rollout_results)
            reach_progression_over_seeds.append(
                [np.sum(rollout_results >= i) / EXPECTED_N_ROLLOUT_FILES for i in range(1, args.max_treechop_num + 1)]
            )
        if len(all_rollout_results) != len(DEFAULT_SEEDS):
            print(f"[Warning] Expected {len(DEFAULT_SEEDS)} seeds, but got {len(all_rollout_results)} for {directory_name}.")
        rollout_pregression_results = np.array(reach_progression_over_seeds)
        all_rollout_results = np.array(all_rollout_results)
        nice_name_to_mean_treechop[nice_name] = np.mean(all_rollout_results)
        nice_name_to_overall_success_ratio[nice_name] = {
            "success_ratios": rollout_pregression_results[:, 0],
            "mean": np.mean(rollout_pregression_results[:, 0]),
            "std": np.std(rollout_pregression_results[:, 0]),
            "std_err": np.std(rollout_pregression_results[:, 0]) / np.sqrt(len(rollout_pregression_results)),
        }

        # Mean over seeds
        mean_reach_progression = np.mean(rollout_pregression_results, axis=0)
        std_reach_progression = np.std(rollout_pregression_results, axis=0)

        # Plot mean and std
        x_range = np.arange(1, args.max_treechop_num + 1)
        ax.plot(x_range, mean_reach_progression, label=nice_name)
        ax.fill_between(
            x_range,
            mean_reach_progression - std_reach_progression,
            mean_reach_progression + std_reach_progression,
            alpha=0.2,
        )

    # Print out the mean scores and sort from highest to lowest
    print("Mean treechops:")
    nice_name_and_mean_treechop = sorted(nice_name_to_mean_treechop.items(), key=lambda x: x[1], reverse=True)
    for nice_name, mean_treechop in nice_name_and_mean_treechop:
        print(f"\t{nice_name:>30}: {mean_treechop:>5.2f}")

    # Print out the overall success ratio and sort from highest to lowest
    print("Overall success ratio (mean, std, std_err):")
    nice_name_and_overall_success_ratio = sorted(nice_name_to_overall_success_ratio.items(), key=lambda x: x[1]["mean"], reverse=True)
    for nice_name, overall_success_ratio in nice_name_and_overall_success_ratio:
        # Print as percentage and std err
        print(
            f"\t{nice_name:>30}: {overall_success_ratio['mean'] * 100:>5.2f}% ({overall_success_ratio['std'] * 100:>4.2f}%) ({overall_success_ratio['std_err'] * 100:>4.2f}%)"
        )

    # Print out t-test results for all models against each other
    if args.t_test:
        print("T-test results:")
        for a_model_name, a_overall_success_ratio in nice_name_and_overall_success_ratio:
            for b_model_name, b_overall_success_ratio in nice_name_and_overall_success_ratio:
                if a_model_name == b_model_name:
                    continue
                _, p_value = ttest_ind(
                    a_overall_success_ratio["success_ratios"],
                    b_overall_success_ratio["success_ratios"],
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

    ax.set_xlabel("Number of treechops")
    ax.set_ylabel("Fraction of rollouts")
    ax.set_xticks(np.arange(1, args.max_treechop_num + 1))
    ax.set_ylim([0, 0.35])
    ax.legend()

    if args.output is not None:
        fig.savefig(args.output)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--directories", nargs="+", required=True, type=str, help="Directories to plot. They should end with '_seed_#'")
    parser.add_argument(
        "--max_treechop_num",
        default=DEFAULT_MAX_TREECHOPS,
        type=int,
        help="Maximum number of treechops we show on the plot",
    )
    parser.add_argument("--output", default=None, type=str, help="Where to store the plot")
    parser.add_argument("--individual_seeds", action="store_true", help="Report results for individual seeds instead")
    parser.add_argument("--t_test", action="store_true", help="Perform t-test of all models vs all models, and print out results.")
    args = parser.parse_args()
    main(args)
