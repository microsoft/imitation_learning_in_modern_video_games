from typing import Dict

import torch
from torch import Tensor


def get_binary_metrics_independent(button_pred: Tensor, button_target: Tensor) -> Dict[str, float]:
    """
    Compute metrics for binary classification predictions (e.g. buttons) where each class is independently
        evaluated and metrics are aggregated over all classes
    :param button_pred: (batchsize, num_buttons)
    :param button_target: (batchsize, num_buttons)
    :return: metrics on the given button predictions
    """
    button_pred = button_pred.bool()
    button_target = button_target.bool()

    # compare individual classes and aggregate
    tp_rate = (button_pred & button_target).float().sum() / button_target.float().sum()
    tn_rate = (~button_pred & ~button_target).float().sum() / (~button_target).float().sum()
    fp_rate = (button_pred & ~button_target).float().sum() / (~button_target).float().sum()
    fn_rate = (~button_pred & button_target).float().sum() / button_target.float().sum()
    accuracy = (button_pred == button_target).float().mean()
    balanced_accuracy = torch.mean(torch.stack([tp_rate, tn_rate]))

    return {
        "tp_rate": tp_rate,
        "tn_rate": tn_rate,
        "fp_rate": fp_rate,
        "fn_rate": fn_rate,
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
    }


def get_binary_metrics_match_all(button_pred: Tensor, button_target: Tensor) -> Dict[str, float]:
    """
    Compute metrics for binary classification predictions (e.g. buttons) where all classes are jointly evaluated,
    e.g. a TP requires all present classes within a sample to be predicted correctly
    :param button_pred: (batchsize, num_buttons)
    :param button_target: (batchsize, num_buttons)
    :param match_all: whether to match all classes or compute metrics individually
    :return: metrics on the given button predictions
    """
    button_pred = button_pred.bool()
    button_target = button_target.bool()

    # all buttons have to match for a positive classification
    # if there are no denominator matches (no positive/ negative cases), return nan for those values
    # and mean over the rest
    positive_mask = button_target.float().sum(dim=-1) > 0
    negative_mask = (~button_target).float().sum(dim=-1) > 0
    tp_matches = (button_pred & button_target).float().sum(dim=-1) == button_target.float().sum(dim=-1)
    tp_rate = tp_matches[positive_mask].float().mean()
    tn_matches = (~button_pred & ~button_target).float().sum(dim=-1) == (~button_target).float().sum(dim=-1)
    tn_rate = tn_matches[negative_mask].float().mean()
    fp_matches = (button_pred & ~button_target).float().sum(dim=-1) == (~button_target).float().sum(dim=-1)
    fp_rate = fp_matches[negative_mask].float().mean()
    fn_matches = (~button_pred & button_target).float().sum(dim=-1) == button_target.float().sum(dim=-1)
    fn_rate = fn_matches[positive_mask].float().mean()
    accuracy = (button_pred == button_target).float().min(dim=-1).values.mean()

    balanced_accuracy = torch.mean(torch.stack([tp_rate, tn_rate]))

    return {
        "tp_rate": tp_rate,
        "tn_rate": tn_rate,
        "fp_rate": fp_rate,
        "fn_rate": fn_rate,
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
    }


def get_button_metrics(button_logits: Tensor, button_target: Tensor) -> Dict[str, float]:
    """
    Compute metrics on the button predictions (TP, TN, FP, FN, accuracy, balanced_accuracy)
    :param button_logits: (batchsize, num_buttons)
    :param button_target: (batchsize, num_buttons)
    :return: metrics on the button predictions
    """
    button_dist = torch.distributions.Bernoulli(logits=button_logits)
    # compute metrics for greedy button actions
    button_pred = button_dist.probs > 0.5
    ind_metrics = get_binary_metrics_independent(button_pred, button_target)
    all_metrics = get_binary_metrics_match_all(button_pred, button_target)
    metrics = {f"button_ind_{key}": value for key, value in ind_metrics.items()}
    metrics.update({f"button_all_{key}": value for key, value in all_metrics.items()})
    return metrics


def get_trigger_metrics(trigger_logits: Tensor, trigger_target: Tensor) -> Dict[str, float]:
    """
    Compute metrics on the trigger predictions (TP, TN, FP, FN, accuracy, balanced_accuracy)
    :param trigger_logits: (batchsize, num_triggers)
    :param trigger_target: (batchsize, num_triggers)
    :return: metrics on the trigger predictions
    """
    nan_filter = torch.isnan(trigger_target) == 0
    trigger_logits = trigger_logits[nan_filter]
    trigger_target = trigger_target[nan_filter]
    trigger_dist = torch.distributions.Bernoulli(logits=trigger_logits)
    # compute metrics for greedy trigger actions
    trigger_pred = trigger_dist.probs > 0.5
    metrics = get_binary_metrics_independent(trigger_pred, trigger_target)
    return {f"trigger_{key}": value for key, value in metrics.items()}


def get_multidiscrete_metrics(pred: Tensor, target: Tensor, num_discretisation_bins: int) -> Dict[str, float]:
    """
    Compute metrics on multidiscrete classification predictions
    :param pred: argmax class predictions (batchsize, actions)
    :param target: target class indices (batchsize, actions)
    :param num_discretisation_bins: number of discrete joystick action bins
    """
    num_actions = pred.shape[-1]
    assert num_actions == target.shape[-1], f"Number of actions in prediction and target must match: {num_actions} != {target.shape[-1]}"
    pred = pred.reshape(-1, num_actions)
    target = target.reshape(-1, num_actions)

    accuracies = (pred == target).float().mean(dim=0)
    balanced_accuracies = []
    for true_class_label in range(num_discretisation_bins):
        class_mask = target == true_class_label
        balanced_accuracies.append((pred[class_mask] == target[class_mask]).float().mean())
    balanced_accuracies = balanced_accuracies

    # aggregate metrics
    metrics = {
        "accuracy": accuracies.mean(),
        "balanced_accuracy": torch.mean(torch.stack(balanced_accuracies)),
    }

    # individual joystick action metrics
    num_joystick_actions = pred.shape[-1]
    if num_joystick_actions == 4:
        joystick_action_labels = ["lx", "ly", "rx", "ry"]
    elif num_joystick_actions == 2:
        joystick_action_labels = ["x", "y"]
    else:
        raise ValueError(f"Unsupported number of joystick actions: {num_joystick_actions}")
    for i, label in enumerate(joystick_action_labels):
        metrics[f"{label}_accuracy"] = accuracies[i]

        # compute balanced accuracy for individual joystick action
        balanced_accuracies_i = []
        for true_class_label in range(num_discretisation_bins):
            class_mask = target[:, i] == true_class_label
            balanced_accuracies_i.append((pred[:, i][class_mask] == target[:, i][class_mask]).float().mean())
        metrics[f"{label}_balanced_accuracy"] = torch.mean(torch.stack(balanced_accuracies_i))

    return metrics


def get_discrete_joystick_metrics(
    joystick_logits: Tensor,
    joystick_target: Tensor,
    num_joystick_actions: int,
    num_discretisation_bins: int,
) -> Dict[str, float]:
    """
    Compute metrics on the discretised joystick predictions (TP, TN, FP, FN, accuracy, precision, recall, f1)
    :param joystick_logits: (batchsize, logits)
    :param joystick_target: (batchsize, actions)
    :param num_joystick_actions: number of joystick actions
    :param num_discretisation_bins: number of bins to discretise the joystick actions into
    :return: metrics on the discretised joystick predictions
    """
    # reshape to (batch_size, num_joystick_actions, num_bins)
    joystick_logits = joystick_logits.swapaxes(-1, -2)
    dist = torch.distributions.Categorical(logits=joystick_logits)
    # compute metrics for greedy joystick actions
    joystick_pred = dist.probs.argmax(dim=-1)
    metrics = get_multidiscrete_metrics(joystick_pred, joystick_target, num_discretisation_bins)
    metrics = {f"joystick_{key}": value for key, value in metrics.items()}
    metrics.update({"joystick_entropy": dist.entropy().mean()})
    return metrics


def get_continuous_metrics(
    pred: Tensor,
    target: Tensor,
) -> Dict[str, float]:
    """
    Compute metrics for continuous regression predictions over continuous joystick predictions (MSE, R2)
    :param pred: (batchsize, actions)
    :param target: (batchsize, actions)
    :return: metrics on the given predictions
    """
    num_actions = pred.shape[-1]
    assert num_actions == target.shape[-1], f"Number of actions in prediction and target must match: {num_actions} != {target.shape[-1]}"
    pred = pred.reshape(-1, num_actions)
    target = target.reshape(-1, num_actions)

    # R^2 = Coefficient of determination
    # R^2 < 0: model is worse than predicting the mean
    # R^2 = 0: model is as good as predicting the mean
    # R^2 = 1: model is perfect
    target_mean = target.mean(axis=0)
    target_total_var = (target - target_mean).pow(2).sum(axis=0).detach()
    mse = (pred - target).pow(2).sum(axis=0).detach()
    rsquared = 1 - (mse / target_total_var.clip(min=1e-10))
    # Bounding the R^2 below since it might be very negative at the beginning
    clipped_rsquared = torch.clamp_min(rsquared, min=-2)

    metrics = {
        "mse": (pred - target).pow(2).mean(),
        "r2": clipped_rsquared.mean(),
    }

    if pred.shape[1] == 4:
        metrics.update(
            {
                "lx_r2": clipped_rsquared[0],
                "ly_r2": clipped_rsquared[1],
                "rx_r2": clipped_rsquared[2],
                "ry_r2": clipped_rsquared[3],
            }
        )
    elif pred.shape[1] == 2:
        metrics.update(
            {
                "x_r2": clipped_rsquared[0],
                "y_r2": clipped_rsquared[1],
            }
        )
    else:
        raise ValueError(f"Unexpected number of continuous joystick actions: {pred.shape[1]}. Expected 2 or 4 joystick actions.")

    return metrics


def get_continuous_joystick_metrics(
    joystick_logits: Tensor,
    joystick_target: Tensor,
) -> Dict[str, float]:
    """
    Compute metrics on the continuous joystick predictions (MSE, R2)
    :param joystick_logits: (batch_size, actions)
    :param joystick_target: (batch_size, actions)
    :return: metrics on the continuous joystick predictions
    """
    joystick_actions = torch.tanh(joystick_logits)
    return {f"joystick_{k}": v for k, v in get_continuous_metrics(joystick_actions, joystick_target).items()}
