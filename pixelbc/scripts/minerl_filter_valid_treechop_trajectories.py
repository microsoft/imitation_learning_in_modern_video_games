# Script for filtering down bunch of MineRL VPT .jsonl files to find valid episodes for treechop task:
# starting from fresh world, gather a single log within specified timeframe.

import glob
import json
# This process was done on the MineRL VPT 6.13 dataset
import os

from tqdm import tqdm

RECORDING_FPS = 20

# One tick is _roughly_ 20ms (separate from recording_fps)
# A world within its first 20s is considered a fresh world
START_TICK_THRESHOLD = 20 * 1000 / 20

TREECHOP_MIN_ALLOWED_STEPS = 100

# Maximum length of treechop example
TREECHOP_TIMEOUT_IN_SECONDS = 60
TREECHOP_TIMEOUT_IN_STEPS = int(TREECHOP_TIMEOUT_IN_SECONDS * (1000 / RECORDING_FPS))

GLOB_PATTERN = "minerl_data/*.jsonl"
OUTPUT_FILE = "valid_treechop_files_and_steps_to_log.txt"

# Manually filtered files, which would be valid with the filtering here, but still are considered bad for "treechop"
# e.g., they start with non-fresh world
MANUAL_BAD_FILES = [
    "Player124-f153ac423f61-20210801-133315.jsonl",
    "Player169-f153ac423f61-20210828-153046.jsonl",
    "Player247-f153ac423f61-20210811-203504.jsonl",
    "Player26-ea950cf0ed6d-20210729-122438.jsonl",
    "Player274-bfbff4cc23c8-20210805-144338.jsonl",
    "Player293-f153ac423f61-20211101-200641.jsonl",
    "Player306-f153ac423f61-20210725-220234.jsonl",
    "Player38-f153ac423f61-20210727-120827.jsonl",
    "Player529-f153ac423f61-20210905-084124.jsonl",
    "Player529-f153ac423f61-20211110-114334.jsonl",
    "Player559-cbb1913deeb4-20211119-183155.jsonl",
    "Player560-f153ac423f61-20210821-163021.jsonl",
    "Player591-f153ac423f61-20210902-173508.jsonl",
    "Player7-f153ac423f61-20210830-010112.jsonl",
    "Player730-fec2ae3b32d7-20210902-122822.jsonl",
    "Player734-f153ac423f61-20211123-212040.jsonl",
    "Player743-f153ac423f61-20211125-193209.jsonl",
    "Player747-f153ac423f61-20210806-144743.jsonl",
    "Player756-b64c5e864d75-20210814-112321.jsonl",
    "Player933-ef857d085c6f-20210727-153952.jsonl",
    "Player934-c78eb9d230df-20211130-210433.jsonl",
    "treechop-f153ac423f61-20210905-103440.jsonl",
    "treechop-f153ac423f61-20210907-211427.jsonl",
    "treechop-f153ac423f61-20210910-132737.jsonl",
    "treechop-f153ac423f61-20210913-150243.jsonl",
    "treechop-f153ac423f61-20210919-203417.jsonl",
    "treechop-f153ac423f61-20211009-203344.jsonl",
    "treechop-f153ac423f61-20211015-175920.jsonl",
    "treechop-f153ac423f61-20211121-220300.jsonl",
]


def load_data():
    blog_files = glob.glob(GLOB_PATTERN)
    for blog_file in tqdm(blog_files, desc="Loading data"):
        file = []
        with open(blog_file, "r") as f:
            for line in f:
                blog = json.loads(line)
                file.append(blog)
        yield blog_file, file


def is_valid_treechop_example(datafile):
    """
    Returns true the datafile (list of states) is a valid example trajectory for treechop task.

    To be valid, it should:
    - Start from a fresh world (server ticks is low)
    - Player should get logs within given timelimit of the episode

    Returns two values:
    - True/False if the datafile is valid
    - The number of steps from start to first log (None if no logs are found)
    """
    start_tick = datafile[0]["serverTick"]
    if start_tick > START_TICK_THRESHOLD:
        return False, None

    # Check if player gets logs within timelimit
    has_logs = False
    steps_to_first_log = None
    for step_i, step in enumerate(datafile):
        if step_i > TREECHOP_TIMEOUT_IN_STEPS:
            break
        inventory = step["inventory"]
        for item in inventory:
            if "log" in item["type"]:
                has_logs = True
                steps_to_first_log = step_i
                break
        if has_logs:
            break
    if not has_logs:
        return False, None
    elif step_i < TREECHOP_MIN_ALLOWED_STEPS:
        # Player was very close to getting logs or already had logs in inventory
        # Skip, as this is not a very useful sample
        return False, None
    return True, steps_to_first_log


def main():
    valid_files = []
    for datafile_path, datafile in load_data():
        datafile_name = os.path.basename(datafile_path)
        if datafile_name in MANUAL_BAD_FILES:
            continue
        is_valid, num_steps_to_log = is_valid_treechop_example(datafile)
        if is_valid:
            valid_files.append((datafile_path, num_steps_to_log))
    print(f"Number of valid files: {len(valid_files)}")

    with open(OUTPUT_FILE, "w") as f:
        for datafile_path, num_steps_to_log in valid_files:
            f.write(f"{datafile_path} {num_steps_to_log}\n")


if __name__ == "__main__":
    main()
