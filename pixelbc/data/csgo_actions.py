from enum import Enum

import numpy as np


class CSGO_ACTIONS(Enum):
    SHOOT = 0
    MOUSE_X = 1
    MOUSE_Y = 2


CSGO_MOUSE_X_BIN_CENTERS = [
    -1000.0,
    -500.0,
    -300.0,
    -200.0,
    -100.0,
    -60.0,
    -30.0,
    -20.0,
    -10.0,
    -4.0,
    -2.0,
    -0.0,
    2.0,
    4.0,
    10.0,
    20.0,
    30.0,
    60.0,
    100.0,
    200.0,
    300.0,
    500.0,
    1000.0,
]
CSGO_MOUSE_Y_BIN_CENTERS = [
    -200.0,
    -100.0,
    -50.0,
    -20.0,
    -10.0,
    -4.0,
    -2.0,
    -0.0,
    2.0,
    4.0,
    10.0,
    20.0,
    50.0,
    100.0,
    200.0,
]

CSGO_MOUSE_X_NORMALIZER = max(np.abs(CSGO_MOUSE_X_BIN_CENTERS))
CSGO_MOUSE_Y_NORMALIZER = max(np.abs(CSGO_MOUSE_Y_BIN_CENTERS))

# For parsing the action vectors in the csgo dataset
# number of keyboard outputs, w,s,a,d,space,ctrl,shift,1,2,3,r
CSGO_N_KEYS = 11
# number of mouse buttons, left, right
CSGO_N_CLICKS = 2
CSGO_N_MOUSE_X = len(CSGO_MOUSE_X_BIN_CENTERS)
CSGO_N_MOUSE_Y = len(CSGO_MOUSE_Y_BIN_CENTERS)

CSGO_ACTION_MAP = {
    "ABS_X": None,
    "ABS_Y": None,
    "ABS_RX": CSGO_ACTIONS.MOUSE_X,
    "ABS_RY": CSGO_ACTIONS.MOUSE_Y,
    "ABS_RZ": None,
    "BTN_SOUTH": None,
    "BTN_EAST": None,
    "BTN_WEST": None,
    "BTN_NORTH": None,
    "BTN_TL": None,
    # Shoot
    "BTN_TR": CSGO_ACTIONS.SHOOT,
    "BTN_SELECT": None,
    "BTN_START": None,
    "BTN_THUMBL": None,
    "BTN_THUMBR": None,
}

CSGO_BUTTON_ORDERING = list(CSGO_ACTION_MAP.values())[5:]


def preprocess_csgo_action_line(line, discretize_continuous_actions):
    """Turn CSGO hdf5 action line into the same format used by Dungeons"""
    assert not discretize_continuous_actions, "Discretized continuous actions not supported for CSGO"
    actions_data = []
    mouse_x_onehot = line[CSGO_N_KEYS + CSGO_N_CLICKS : CSGO_N_KEYS + CSGO_N_CLICKS + CSGO_N_MOUSE_X]
    mouse_y_onehot = line[CSGO_N_KEYS + CSGO_N_CLICKS + CSGO_N_MOUSE_X : CSGO_N_KEYS + CSGO_N_CLICKS + CSGO_N_MOUSE_X + CSGO_N_MOUSE_Y]
    mouse_x_continuous_normalized = (mouse_x_onehot @ CSGO_MOUSE_X_BIN_CENTERS).item() / CSGO_MOUSE_X_NORMALIZER
    mouse_y_continuous_normalized = (mouse_y_onehot @ CSGO_MOUSE_Y_BIN_CENTERS).item() / CSGO_MOUSE_Y_NORMALIZER
    mouse_click = line[CSGO_N_KEYS].item()
    for action, csgo_action_name in CSGO_ACTION_MAP.items():
        if csgo_action_name is None:
            # Just add 0, this is an ignored button
            actions_data.append(0)
            continue

        if csgo_action_name == CSGO_ACTIONS.MOUSE_X:
            actions_data.append(mouse_x_continuous_normalized)
        elif csgo_action_name == CSGO_ACTIONS.MOUSE_Y:
            actions_data.append(mouse_y_continuous_normalized)
        elif csgo_action_name == CSGO_ACTIONS.SHOOT:
            actions_data.append(mouse_click)
        else:
            raise ValueError(f"Unknown action name {csgo_action_name}")
    return np.array(actions_data, dtype=np.float32)


def csgo_model_output_to_csgo_action(joystick_actions, trigger_actions, button_actions):
    """
    Turn model outputs/predictions back into a valid CSGO action

    :param joystick_actions: joystick actions to process.
    :param trigger_actions: trigger actions to process.
    :param button_actions: button actions to process.
    :return: CSGO action as dict
    """
    csgo_action = {}

    # Denormalize camera actions
    # NOTE these index numbers are based on what is defined in CSGO_ACTION_MAP
    csgo_action[CSGO_ACTIONS.MOUSE_X] = joystick_actions[2] * CSGO_MOUSE_X_NORMALIZER
    csgo_action[CSGO_ACTIONS.MOUSE_Y] = joystick_actions[3] * CSGO_MOUSE_Y_NORMALIZER

    for button_i, button_number in enumerate(button_actions):
        button_csgo_action = CSGO_BUTTON_ORDERING[button_i]
        if button_csgo_action is None:
            continue
        csgo_action[button_csgo_action] = int(button_number)

    return csgo_action
