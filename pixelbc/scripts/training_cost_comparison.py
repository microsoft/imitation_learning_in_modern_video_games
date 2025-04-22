# Compare training speeds of different configurations of the same model
from argparse import ArgumentParser
import os
import subprocess
import time
import threading

import numpy as np
from prettytable import PrettyTable
import pynvml


def get_args():
    parser = ArgumentParser()
    parser.add_argument("configs", nargs="+")
    parser.add_argument("--num_runs", type=int, default=3)
    parser.add_argument("--num_steps", type=int, default=1000)
    return parser.parse_args()


def monitor_vram_usage(interval, stop_event, vram_usage):
    """Monitor VRAM usage in MB. Stops when stop_event is set."""
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    base_usage = mem_info.used // 1024 // 1024
    while not stop_event.is_set():
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        vram_usage.append(mem_info.used // 1024 // 1024 - base_usage)
        time.sleep(interval)


def train(config_path, num_steps):
    cmd = f"python run_train.py {config_path} trainer.max_steps={num_steps}".split()
    out = subprocess.run(cmd)
    return out.returncode


def monitored_training_run(config_path, num_steps):
    # VRAM monitoring setup
    vram_usage = []
    stop_event = threading.Event()
    monitor_thread = threading.Thread(target=monitor_vram_usage, args=(10, stop_event, vram_usage))
    monitor_thread.start()

    # Run the training
    start_time = time.time()
    returncode = train(config_path, num_steps)

    if returncode != 0:
        # training crashed
        stop_event.set()
        monitor_thread.join()
        return None, None

    end_time = time.time()

    duration = end_time - start_time

    # Stop VRAM monitoring
    stop_event.set()
    monitor_thread.join()

    return duration, max(vram_usage)


def main(args):
    assert all(os.path.exists(config_path) for config_path in args.configs), "All config paths must exist"
    table = PrettyTable()
    table.field_names = ["Config name", "Time (min)", "VRAM (GB)"]
    for config_path in args.configs:
        print(f"Running {args.num_runs} training runs of {config_path} for {args.num_steps} steps")
        run_times = []
        vram_usages = []
        for i in range(args.num_runs):
            run_time, max_vram = monitored_training_run(config_path, args.num_steps)
            run_times.append(run_time)
            vram_usages.append(max_vram)
        
        config_name = os.path.basename(config_path)
        if any(run_time is None for run_time in run_times):
            print(f"Training failed for {config_path}")
            table.add_row([config_name, "Failed", "Failed"])
        else:
            mean_run_time_mins = np.mean(run_times) / 60
            mean_vram_usage_gb = np.mean(vram_usages) / 1024
            print(f"Results for {config_path} across {args.num_runs} runs:")
            print(f"\tRun time: {mean_run_time_mins:.2f} mins")
            print(f"\tVRAM usage: {mean_vram_usage_gb:.2f} GB")
            table.add_row([config_name, f"{mean_run_time_mins:.2f}", f"{mean_vram_usage_gb:.2f}"])
    print(table)


if __name__ == "__main__":
    args = get_args()
    main(args)
