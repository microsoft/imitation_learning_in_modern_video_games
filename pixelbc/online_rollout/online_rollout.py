import json
import time
from abc import abstractmethod
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from pixelbc.data.image_preprocessing import \
    get_preprocessing_function_and_image_shape
from pixelbc.models.utils.model_utils import count_parameters
from pixelbc.utils.load_checkpoint import load_checkpoint


class OnlineRollout:
    """Base class for rollouts"""

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
        """
        Online rollout of a trained model on a level.
        :param checkpoint_path: Path to the checkpoint of the model to rollout.
        :param save_dir: Directory to save the rollout results to.
        :param joystick_action_mode: Action mode to use for joystick actions ("stochastic" or "deterministic").
        :param trigger_action_mode: Action mode to use for trigger actions ("stochastic" or "deterministic").
        :param button_action_mode: Action mode to use for button actions ("stochastic" or "deterministic").
        :param fps: FPS to capture frames at.
        :param ignore_keyboard_inputs: Whether to ignore keyboard inputs (useful for running in background).
        """
        self.checkpoint_path = checkpoint_path
        self.save_dir = Path(save_dir)
        self.joystick_action_mode = joystick_action_mode
        self.trigger_action_mode = trigger_action_mode
        self.button_action_mode = button_action_mode
        self.fps = fps
        self.active_rollout = False
        self.ignore_keyboard_inputs = ignore_keyboard_inputs

        self.save_dir.mkdir(parents=True, exist_ok=True)

        # load model from checkpoint
        print(f"Loading model from checkpoint {checkpoint_path}...")
        self.model, self.hyperparameters = load_checkpoint(checkpoint_path, eval_mode=True)
        self.model.eval()
        print(self.model)
        trainable_params, total_params = count_parameters(self.model)
        trainable_params_str = f"{trainable_params / 1e6:.1f}M" if trainable_params > 1e6 else f"{trainable_params / 1e3:.1f}K"
        total_params_str = f"{total_params / 1e6:.1f}M" if total_params > 1e6 else f"{total_params / 1e3:.1f}K"
        print(f"Model has {trainable_params_str} trainable parameters and {total_params_str} total parameters")

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        self.model.to(self.device)

        # load hyperparameters from checkpoint
        self.num_joystick_actions = self.hyperparameters.num_joystick_actions
        if "num_trigger_actions" in self.hyperparameters:
            self.num_trigger_actions = self.hyperparameters.num_trigger_actions
        else:
            self.num_trigger_actions = 0
        if "num_button_actions" in self.hyperparameters:
            self.num_button_actions = self.hyperparameters.num_button_actions
        else:
            self.num_button_actions = self.hyperparameters.num_actions - self.num_joystick_actions - self.num_trigger_actions

        self.preprocessing_function, self.preprocessed_image_shape = get_preprocessing_function_and_image_shape(
            self.hyperparameters.encoder is None,
            self.hyperparameters.pretrained_encoder,
            self.hyperparameters.image_shape[1],
            self.hyperparameters.image_shape[2],
        )

    @abstractmethod
    def _start_level(self, plugin_exp_config):
        """
        Start a new level.
        :param plugin_exp_config: Plugin experiment config to use (specifies level and command to initialise the experiment).
        """
        pass

    def _process_frame(self, raw_frame):
        """
        Process a frame.
        :param raw_frame: Raw frame to process (in BGR)
        :return: Processed frame stack (as numpy array) and tensor (as torch tensor
        """
        # preprocess frame in RGB
        frame = self.preprocessing_function(raw_frame[:, :, ::-1])
        self.img_stack.append(frame)
        stacked_frames = np.concatenate(self.img_stack, axis=0)
        # convert to tensor and pass to model
        stacked_frames_tensor = torch.from_numpy(stacked_frames).to(self.device)
        return stacked_frames, stacked_frames_tensor

    def _get_actions(self, stacked_frames_tensor):
        """
        Get the action to take in the current frame.
        :param stacked_frames_tensor: Stacked frames tensor to pass to model.
        :return: Joytick, trigger and button actions to take (as numpy arrays)
        """
        with torch.no_grad():
            joystick_actions, trigger_actions, button_actions = self.model.act(
                stacked_frames_tensor,
                joystick_mode=self.joystick_action_mode,
                trigger_mode=self.trigger_action_mode,
                button_mode=self.button_action_mode,
            )
        # convert actions to numpy
        joystick_actions = joystick_actions.cpu().numpy()
        trigger_actions = trigger_actions.cpu().numpy()
        button_actions = button_actions.cpu().numpy()
        return joystick_actions, trigger_actions, button_actions

    @abstractmethod
    def rollout_loop(self, plugin_exp_config, **kwargs):
        """
        Start a rollout loop.
        :param plugin_exp_config: Plugin experiment config to use (specifies level and command to initialise the experiment).
        """
        pass

    def _stop(self):
        pass

    def _stop_and_save(self, config_path=None, plugin_path=None, actions_path=None, save_frames=True):
        if not config_path:
            config_path = self.save_dir / "config.yaml"
        if not plugin_path:
            plugin_path = self.save_dir / "plugin_data.json"
        if not actions_path:
            actions_path = self.save_dir / "actions.json"

        duration = time.time() - self.start_time
        recorded_fps = self.step / duration
        print(f"Recorded {self.step} steps in {duration:.2f} seconds ({recorded_fps:.2f} fps)")
        self._stop()
        # save config
        duration = time.time() - self.start_time
        recorded_fps = self.step / duration
        self._stop()
        config = OmegaConf.create(
            {
                "joystick_action_mode": self.joystick_action_mode,
                "trigger_action_mode": self.trigger_action_mode,
                "button_action_mode": self.button_action_mode,
                "fps": self.fps,
                "recorded_fps": recorded_fps,
                "checkpoint": self.checkpoint_path,
                "num_steps": self.step,
                "duration": duration,
                "timestamp": time.time(),
                "hyper_parameters": self.hyperparameters,
            }
        )
        OmegaConf.save(config, config_path)

        # save rollout data
        if save_frames:
            frames_path = self.save_dir / "frames.npy"
            np.save(frames_path, np.array(self.all_frames))
        with open(actions_path, "w") as f:
            json.dump(
                {
                    "joystick_actions": np.stack(self.all_joystick_actions).tolist(),
                    "trigger_actions": np.stack(self.all_trigger_actions).tolist(),
                    "button_actions": np.stack(self.all_button_actions).tolist(),
                },
                f,
            )

        # save plugin data to JSON
        plugin_dict = {
            key: [data[key].tolist() if isinstance(data[key], np.ndarray) else data[key] for data in self.plugin_data]
            for key in self.plugin_data[0].keys()
        }
        with open(plugin_path, "w") as f:
            json.dump(plugin_dict, f, indent=2)
