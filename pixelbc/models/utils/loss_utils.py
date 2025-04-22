import torch
import torch.nn.functional as F
from torch import Tensor


def compute_joystick_loss(
    joystick_pred: Tensor,
    joystick_target: Tensor,
    discretise_joystick: bool,
) -> Tensor:
    if discretise_joystick:
        # CE loss over classification predictions (batch_size, num_classes, actions)
        joystick_loss = F.cross_entropy(joystick_pred, joystick_target.long())
    else:
        # MSE loss over regression predictions (batch_size, actions)
        joystick_loss = F.mse_loss(torch.tanh(joystick_pred), joystick_target)
    return joystick_loss


def compute_trigger_loss(
    trigger_pred: Tensor,
    trigger_target: Tensor,
) -> Tensor:
    trigger_loss = F.binary_cross_entropy_with_logits(trigger_pred, trigger_target)
    return trigger_loss


def compute_button_loss(
    button_pred: Tensor,
    button_target: Tensor,
) -> Tensor:
    button_loss = F.binary_cross_entropy_with_logits(button_pred, button_target)
    return button_loss
