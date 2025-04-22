import numpy as np

# entries smaller than this will be clipped to 0
JOYSTICK_DEADZONE = 0.2
# max and min values for joystick
JOYSTICK_MIN = -1.0
JOYSTICK_MAX = 1.0
# separating values for bins of joystick action discretisation
JOYSTICK_BINS = [-0.99, -0.7, -0.45, -0.2, 0.2, 0.45, 0.7, 0.99]

# trigger threshold for discretisation
TRIGGER_THRESHOLD = 0.2
# number of buttons on the controller
BUTTON_ACTIONS = 10
# map from xbox controller action names to more descriptive names
ACTION_MAP = {
    "ABS_X": "left_joystick_x",
    "ABS_Y": "left_joystick_y",
    "ABS_RX": "right_joystick_x",
    "ABS_RY": "right_joystick_y",
    "ABS_RZ": "right_trigger",
    "BTN_SOUTH": "A",
    "BTN_EAST": "B",
    "BTN_WEST": "X",
    "BTN_NORTH": "Y",
    "BTN_TL": "LB",
    "BTN_TR": "RB",
    "BTN_SELECT": "select",
    "BTN_START": "start",
    "BTN_THUMBL": "left_thumb",
    "BTN_THUMBR": "right_thumb",
}


def get_joystick_bins():
    """
    Get the values of the joystick bins.
    :return: values of the joystick bins.
    """
    joystick_bin_values = np.zeros(len(JOYSTICK_BINS) + 1)
    joystick_bin_values[0] = JOYSTICK_MIN
    joystick_bin_values[-1] = JOYSTICK_MAX
    for i in range(1, len(JOYSTICK_BINS)):
        joystick_bin_values[i] = (JOYSTICK_BINS[i - 1] + JOYSTICK_BINS[i]) / 2
    return joystick_bin_values


def discretise_individual_joystick_action(action):
    """
    Discretise single joystick action into bins.
    :param action: joystick action to discretise.
    :return: discretised joystick action.
    """
    for i, bin_boundary in enumerate(JOYSTICK_BINS):
        if action < bin_boundary:
            return i
    return len(JOYSTICK_BINS)


def set_individual_continuous_joystick_deadzone(action, deadzone=JOYSTICK_DEADZONE):
    """
    Set single joystick action within the deadzone to 0.
    :param action: joystick action to process.
    :param deadzone: deadzone threshold.
    :return: processed joystick action.
    """
    if action < deadzone and action > -deadzone:
        return 0.0
    return float(action)


def clip_individual_trigger_action(action, threshold=TRIGGER_THRESHOLD):
    """
    Clip single trigger action below the threshold to 0 and others to 1.
    :param action: trigger action to process.
    :param threshold: threshold for clipping.
    :return: processed trigger action.
    """
    if action < threshold:
        return 0
    return 1


def preprocess_individual_actions(actions, discretise_joystick=False):
    """
    Preprocess individual actions (one step with all action values)
    :param actions: actions to preprocess as dict
    :param discretise_joystick: discretise joystick actions into bins.
    :return: preprocessed actions as numpy array (num_actions,)
    """
    actions_data = []
    for action, name in ACTION_MAP.items():
        if action not in actions:
            # print(f"Warning: action {action} not found in actions file")
            action_value = 0
        else:
            action_value = float(actions[action])

        if "joystick" in name:
            # process joystick actions
            if discretise_joystick:
                action_value = discretise_individual_joystick_action(action_value)
            else:
                action_value = set_individual_continuous_joystick_deadzone(action_value)
        elif "trigger" in name:
            # process trigger actions
            action_value = clip_individual_trigger_action(action_value)
        else:
            # process button actions
            action_value = int(action_value)

        actions_data.append(action_value)
    return np.array(actions_data, dtype=np.float32)
