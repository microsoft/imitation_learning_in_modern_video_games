import time
from collections import deque

import ffmpegcv
import numpy as np
from gym import Wrapper
from gym.wrappers.time_limit import TimeLimit
from minerl.herobraine.env_specs.human_survival_specs import HumanSurvival

from pixelbc.data.minerl_actions import minerl_model_output_to_minerl_action
from pixelbc.online_rollout.online_rollout import OnlineRollout

ENV_KWARGS = dict(
    fov_range=[70, 70],
    frameskip=1,
    gamma_range=[2, 2],
    guiscale_range=[1, 1],
    resolution=[640, 360],
    cursor_size_range=[16.0, 16.0],
)

NUM_ROLLOUTS = 100
FIXED_ENV_SEEDS = [1000 + i for i in range(NUM_ROLLOUTS)]
# One minute
EPISODE_TIMEOUT_IN_ENV_STEPS = 20 * 60

# Recreate env every now and then to make sure things work
RECREATE_EVERY_N_RESETS = 5
GAME_FPS = 20


class PeriodicallyRecreateWrapper(Wrapper):
    """
    Periodically recreate the environment upon reset.
    """

    def __init__(self, env, env_creation_fn, recreate_every_n_resets):
        super().__init__(env)
        self.env_creation_fn = env_creation_fn
        self.recreate_every_n_resets = recreate_every_n_resets
        self.reset_counter = 0

    def _recreate_env(self):
        self.env.close()
        self.env = self.env_creation_fn()
        # MineRL quick: reset env once to ensure seeding works
        _ = self.env.reset()

    def reset(self, **kwargs):
        if self.reset_counter == self.recreate_every_n_resets:
            self._recreate_env()
            self.reset_counter = 0
        self.reset_counter += 1
        return self.env.reset(**kwargs)


class DownsampleWrapper(Wrapper):
    """
    Wrapper to skip frames ("frameskip" or "downsample").
    1 = no frameskip
    2 = repeat action for two steps.
    3 = etc
    """

    def __init__(self, env, frameskip):
        super().__init__(env)
        self.frameskip = frameskip

    def step(self, action):
        reward = 0
        done = False
        for _ in range(self.frameskip):
            obs, r, d, info = self.env.step(action)
            reward += r
            done = done or d
            if done:
                break
        return obs, reward, done, info


class TreechopWrapper(Wrapper):
    """
    Wrapper that ends the episode when logs are obtained,
    and returns +1 reward
    """

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        for item_name, count in obs["inventory"]:
            if count == 0:
                continue
            if "log" in item_name:
                done = True
                reward = 1
        return obs, reward, done, info


def create_env_creation_fn(downsample):
    def create_env():
        env = HumanSurvival(**ENV_KWARGS).make()
        env = TimeLimit(env, max_episode_steps=EPISODE_TIMEOUT_IN_ENV_STEPS)
        env = DownsampleWrapper(env, downsample)
        return env

    return create_env


def remove_numpyness_and_remove_zeros(dict_with_numpy_arrays):
    # Recursively remove numpyness from a dictionary.
    # Remove zeros from the dictionary as well to save space.
    # From https://github.com/minerllabs/basalt-benchmark/blob/main/basalt/rollout_model.py (MIT)
    if isinstance(dict_with_numpy_arrays, dict):
        new_dict = {}
        for key, value in dict_with_numpy_arrays.items():
            new_value = remove_numpyness_and_remove_zeros(value)
            if new_value != 0:
                new_dict[key] = new_value
        return new_dict
    elif isinstance(dict_with_numpy_arrays, np.ndarray):
        if dict_with_numpy_arrays.size == 1:
            return dict_with_numpy_arrays.item()
        else:
            return dict_with_numpy_arrays.tolist()


class MineRLThreechopRollout(OnlineRollout):
    def __init__(
        self,
        checkpoint_path,
        save_dir,
        joystick_action_mode,
        trigger_action_mode,
        button_action_mode,
        fps,
        ignore_keyboard_inputs=False,
    ):
        super().__init__(
            checkpoint_path,
            save_dir,
            joystick_action_mode,
            trigger_action_mode,
            button_action_mode,
            fps,
            ignore_keyboard_inputs,
        )
        self.downsample = GAME_FPS / self.fps
        # Enforce downsample to be a an integer
        assert self.downsample == int(self.downsample), f"FPS {self.fps} must evenly divide {GAME_FPS}"
        self.downsample = int(self.downsample)
        env_creation_fn = create_env_creation_fn(self.downsample)
        self.env = PeriodicallyRecreateWrapper(env_creation_fn(), env_creation_fn, recreate_every_n_resets=RECREATE_EVERY_N_RESETS)

    def _reset_and_run_till_done(self, video_out):
        obs = self.env.reset()
        done = False
        with ffmpegcv.VideoWriter(video_out, None, self.fps) as video_out:
            while not done:
                # Flip to BGR for consistency (underlying code assumed opencv ordered data)
                raw_frame = obs["pov"][:, :, ::-1]
                stacked_frames, stacked_frames_tensor = self._process_frame(raw_frame)
                joystick_actions, trigger_actions, button_actions = self._get_actions(stacked_frames_tensor)

                minerl_action = minerl_model_output_to_minerl_action(joystick_actions, trigger_actions, button_actions)
                obs, reward, done, info = self.env.step(minerl_action)

                video_out.write(raw_frame)
                self.all_frames.append(stacked_frames)
                self.all_joystick_actions.append(joystick_actions)
                self.all_trigger_actions.append(trigger_actions)
                self.all_button_actions.append(button_actions)
                self.plugin_data.append(
                    {
                        "step": self.step,
                        "reward": reward,
                        "break_item": remove_numpyness_and_remove_zeros(obs["break_item"]),
                        "mine_block": remove_numpyness_and_remove_zeros(obs["mine_block"]),
                        "pickup": remove_numpyness_and_remove_zeros(obs["pickup"]),
                    }
                )
                self.step += 1

    def rollout_loop(self):
        for rollout_i in range(NUM_ROLLOUTS):
            self.model.init_for_rollout()

            # setup empty stack of frames for framestacking
            self.img_stack = deque(maxlen=self.hyperparameters["framestacking"])
            for _ in range(self.hyperparameters["framestacking"]):
                self.img_stack.append(np.zeros(self.preprocessed_image_shape, dtype=np.uint8))

            self.step = 0
            self.all_frames = []
            self.all_joystick_actions = []
            self.all_trigger_actions = []
            self.all_button_actions = []
            self.plugin_data = []
            self.start_time = time.time()

            self.env.seed(FIXED_ENV_SEEDS[rollout_i])
            video_path = self.save_dir / f"video_{rollout_i}.mp4"
            config_path = self.save_dir / f"config_{rollout_i}.yaml"
            plugin_path = self.save_dir / f"plugin_data_{rollout_i}.yaml"
            actions_path = self.save_dir / f"actions_{rollout_i}.yaml"
            self._reset_and_run_till_done(video_path)
            # Not really stopping, just saving results
            self._stop_and_save(config_path, plugin_path, actions_path, save_frames=False)
        self.env.close()
