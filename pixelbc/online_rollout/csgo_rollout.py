import time
from collections import deque

import ffmpegcv
import mss
import numpy as np
import win32gui
from csgo_utils.config import wait_for_loop_end
from csgo_utils.key_input import mouse_check
from csgo_utils.key_output import (hold_left_click, mp_restartgame,
                                   release_left_click, set_pos)
from csgo_utils.meta_utils import server
from csgo_utils.screen_input import grab_window

from pixelbc.data.csgo_actions import (CSGO_ACTIONS,
                                       csgo_model_output_to_csgo_action)
from pixelbc.online_rollout.online_rollout import OnlineRollout

GAME_WINDOW_NAME = "Counter-Strike: Global Offensive - Direct3D 9"

CSGO_GAME_RESOLUTION = (1024, 768)
ORIGINAL_DATA_FPS = 16
NUM_ROLLOUTS = 3
ROLLOUT_TIME_IN_MINUTES = 5
ROLLOUT_TIME_IN_SECONDS = ROLLOUT_TIME_IN_MINUTES * 60


class CSGORollout(OnlineRollout):
    # Note: FPS here controls the speed at which rollout is done. We take actions at the fixed 16 actions per game second
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

        self.host_timescale = round(fps / ORIGINAL_DATA_FPS, 4)
        print("[Note] Looking for CSGO window...")
        self.hwin_csgo = 0
        while self.hwin_csgo == 0:
            self.hwin_csgo = win32gui.FindWindow(None, "Counter-Strike: Global Offensive - Direct3d 9")
            time.sleep(1)
        win32gui.SetForegroundWindow(self.hwin_csgo)
        time.sleep(1)

        # get info about resolution and monitors
        sct = mss.mss()
        if len(sct.monitors) == 3:
            self.Wd, self.Hd = sct.monitors[2]["width"], sct.monitors[2]["height"]
        else:
            self.Wd, self.Hd = sct.monitors[1]["width"], sct.monitors[1]["height"]

        print("capturing mouse position...")
        time.sleep(0.2)
        self.mouse_x_mid, self.mouse_y_mid = mouse_check()

    def _reset_and_run_till_done(self, video_out):
        release_left_click()
        win32gui.SetForegroundWindow(self.hwin_csgo)
        time.sleep(0.5)
        mp_restartgame(timescale=self.host_timescale)
        time.sleep(0.5)

        while True:
            server.handle_request()
            if not hasattr(server, "data_all") or server.data_all is None:
                print("Could not find data_all in server. Is server getting data from GSI?")
            elif "map" not in server.data_all.keys() or "player" not in server.data_all.keys():
                print('not running, "map" or "player" not in keys:', server.data_all.keys())
            else:
                break
            time.sleep(0.1)

        with ffmpegcv.VideoWriter(video_out, None, ORIGINAL_DATA_FPS) as video_out:
            start_time = time.time()
            is_left_click_down = False
            while (time.time() - start_time) < (ROLLOUT_TIME_IN_SECONDS / self.host_timescale):
                loop_start_time = time.time()
                raw_frame = grab_window(self.hwin_csgo, game_resolution=CSGO_GAME_RESOLUTION, SHOW_IMAGE=False)

                # Need to flip to BGR as underlying code assumes this
                stacked_frames, stacked_frames_tensor = self._process_frame(raw_frame[:, :, ::-1])
                joystick_actions, trigger_actions, button_actions = self._get_actions(stacked_frames_tensor)

                csgo_action = csgo_model_output_to_csgo_action(joystick_actions, trigger_actions, button_actions)

                # Set actions
                # Mouse
                mouse_x_smooth = csgo_action[CSGO_ACTIONS.MOUSE_X]
                mouse_y_smooth = csgo_action[CSGO_ACTIONS.MOUSE_Y]
                set_pos(self.mouse_x_mid + mouse_x_smooth / 1, self.mouse_y_mid + mouse_y_smooth / 1, self.Wd, self.Hd)

                # Click
                should_press_left_click = bool(csgo_action[CSGO_ACTIONS.SHOOT])
                if should_press_left_click and not is_left_click_down:
                    hold_left_click()
                    is_left_click_down = True
                elif not should_press_left_click and is_left_click_down:
                    release_left_click()
                    is_left_click_down = False

                server.handle_request()
                current_kills = server.data_all["player"]["match_stats"]["kills"]

                video_out.write(raw_frame)
                self.all_frames.append(None)
                self.all_joystick_actions.append(joystick_actions)
                self.all_trigger_actions.append(trigger_actions)
                self.all_button_actions.append(button_actions)
                self.plugin_data.append(
                    {
                        "step": self.step,
                        "time": loop_start_time,
                        "kills": current_kills,
                    }
                )
                wait_for_loop_end(loop_start_time, self.fps, n_loops=self.step, is_clear_decals=True)
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

            video_path = self.save_dir / f"video_{rollout_i}.mp4"
            config_path = self.save_dir / f"config_{rollout_i}.yaml"
            plugin_path = self.save_dir / f"plugin_data_{rollout_i}.json"
            actions_path = self.save_dir / f"actions_{rollout_i}.json"

            if video_path.exists() and config_path.exists() and plugin_path.exists() and actions_path.exists():
                print(f"Rollout {rollout_i} already exists. Skipping.")
                continue

            self._reset_and_run_till_done(video_path)
            # Not really stopping, just saving results
            self._stop_and_save(config_path, plugin_path, actions_path, save_frames=False)
        server.server_close()
        release_left_click()
