import numpy as np

from pixelbc.data.actions import discretise_individual_joystick_action

# map from xbox controller names to MineRL VPT buttons in JSON data.
# This is arbritraly chosen, as original MineRL VPT data is in keyboard/mouse space.
# Right joystick will be used for mouse movement.
# Left joystick will be ignored and instead direction buttons are used (both keys should be treated as separate buttons)
# If name is None, then the button is ignored
MINERL_ACTION_MAP = {
    "ABS_X": None,
    "ABS_Y": None,
    "ABS_RX": "camera.dx",
    "ABS_RY": "camera.dy",
    "ABS_RZ": None,
    # What a silly mapping this would be for humans :D
    "BTN_SOUTH": "key.keyboard.s",
    "BTN_EAST": "key.keyboard.d",
    "BTN_WEST": "key.keyboard.a",
    "BTN_NORTH": "key.keyboard.w",
    # Use/open things
    "BTN_TL": "mouse.1",
    # Attack
    "BTN_TR": "mouse.0",
    "BTN_SELECT": None,
    "BTN_START": None,
    # Sprint
    "BTN_THUMBL": "key.keyboard.left.control",
    # Jump
    "BTN_THUMBR": "key.keyboard.space",
}

MINERL_BUTTON_ORDERING = list(MINERL_ACTION_MAP.values())[5:]

# Mapping of MineRL keyboard buttons in data to MineRL env buttons
MINERL_JSON_BUTTON_NAME_TO_ENV = {
    "key.keyboard.escape": "ESC",
    "key.keyboard.s": "back",
    "key.keyboard.q": "drop",
    "key.keyboard.w": "forward",
    "key.keyboard.1": "hotbar.1",
    "key.keyboard.2": "hotbar.2",
    "key.keyboard.3": "hotbar.3",
    "key.keyboard.4": "hotbar.4",
    "key.keyboard.5": "hotbar.5",
    "key.keyboard.6": "hotbar.6",
    "key.keyboard.7": "hotbar.7",
    "key.keyboard.8": "hotbar.8",
    "key.keyboard.9": "hotbar.9",
    "key.keyboard.e": "inventory",
    "key.keyboard.space": "jump",
    "key.keyboard.a": "left",
    "key.keyboard.d": "right",
    "key.keyboard.left.shift": "sneak",
    "key.keyboard.left.control": "sprint",
    "key.keyboard.f": "swapHands",
    "mouse.0": "attack",
    "mouse.1": "use",
}


# From https://github.com/openai/Video-Pre-Training/blob/main/run_inverse_dynamics_model.py#L76
MINERL_CAMERA_SCALER = 360.0 / 2400.0
MINERL_CAMERA_CLIP_VALUE = 180.0
MINERL_CAMERA_MAX_VALUE = 10.0


def preprocess_minerl_action_line(line, discretize_continuous_actions):
    """
    Preprocess individual actions (one step with all action values).
    This is like preprocess_individual_actions, but for MineRL VPT dataset.

    For consistency, this returns same shape data as Dungeons data
    (i.e., buttons for all xbox buttons and joysticks), even if
    most of them are zeros.

    This follows logic in:
    https://github.com/openai/Video-Pre-Training/blob/main/run_inverse_dynamics_model.py#L80

    :param line: actions to preprocess as dict
    :param discretize_continuous_actions: discretise joystick actions into bins.
    :return: preprocessed actions as numpy array (num_actions,)
    """
    actions_data = []
    keyboard_keys = line["keyboard"]["keys"]
    mouse = line["mouse"]
    mouse_buttons = mouse["buttons"]
    for action, minerl_action_name in MINERL_ACTION_MAP.items():
        if minerl_action_name is None:
            # Just add 0, this is an ignored button
            actions_data.append(0)
            continue

        if "keyboard" in minerl_action_name:
            # keyboard_keys tells which buttons are pressed, so check
            # if button-to-be-checked is pressed.
            action_value = 1 if minerl_action_name in keyboard_keys else 0
        elif "camera" in minerl_action_name:
            # camera.dx and camera.dy are mouse movements
            mouse_value = mouse[minerl_action_name.split(".")[-1]]
            mouse_value *= MINERL_CAMERA_SCALER
            # Clip as per the original code
            if abs(mouse_value) > MINERL_CAMERA_CLIP_VALUE:
                mouse_value = 0
            # Normalize to roughly [-1, 1].
            # This is not exactly right, but this is what VPT also aimed for
            action_value = mouse_value / MINERL_CAMERA_MAX_VALUE
            # Discretize if needed
            if discretize_continuous_actions:
                # TODO could use some other bins than Dungeons ones, they are rather crude.
                action_value = discretise_individual_joystick_action(action_value)
        elif "mouse" in minerl_action_name:
            # Same as keyboard but with mouse buttons (integers)
            button_value = int(minerl_action_name.split(".")[-1])
            action_value = 1 if button_value in mouse_buttons else 0
        else:
            raise ValueError(f"Unknown action name {minerl_action_name}")

        actions_data.append(action_value)
    return np.array(actions_data, dtype=np.float32)


def minerl_model_output_to_minerl_action(joystick_actions, trigger_actions, button_actions):
    """
    Turn model outputs/predictions back into a valid MineRL action

    :param joystick_actions: joystick actions to process.
    :param trigger_actions: trigger actions to process.
    :param button_actions: button actions to process.
    :return: MineRL action as dict
    """
    minerl_action = {}

    # Denormalize camera actions
    # NOTE these index numbers are based on what is defined in MINERL_ACTION_MAP
    camera_x = joystick_actions[2] * MINERL_CAMERA_MAX_VALUE
    camera_y = joystick_actions[3] * MINERL_CAMERA_MAX_VALUE

    minerl_action["camera"] = np.array([camera_x, camera_y], dtype=np.float32)

    for button_i, button_number in enumerate(button_actions):
        if button_number == 0:
            continue
        button_keyboard_name = MINERL_BUTTON_ORDERING[button_i]
        if button_keyboard_name is None:
            continue
        button_env_name = MINERL_JSON_BUTTON_NAME_TO_ENV[button_keyboard_name]
        minerl_action[button_env_name] = 1

    return minerl_action
