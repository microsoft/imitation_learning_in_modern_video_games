import argparse
from enum import Enum
from functools import partial
from pathlib import Path

import numpy as np
import torch
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

from pixelbc.data.image_preprocessing import \
    get_preprocessing_function_and_image_shape
from pixelbc.data.images import load_image, render_image
from pixelbc.models.encoders.pretrained_encoders import unfreeze_model
from pixelbc.utils.load_checkpoint import load_checkpoint

DEVICE = "cpu"  # "cuda" if torch.cuda.is_available() else "cpu"


class GradCamTargetConcept(Enum):
    EMBEDDING = "embedding"
    ACTIONS = "actions"
    MOVEMENT = "movement"
    ATTACK = "attack"


class ScoreOutputTarget:
    def __init__(self, target_concept, index):
        self.target_concept = target_concept
        self.index = index

    def __call__(self, model_output):
        if self.target_concept == GradCamTargetConcept.EMBEDDING:
            return model_output[0, self.index]
        else:
            return model_output[self.index]


def get_grad_cam_targets(model, model_config, target_concept):
    # get targets and layers
    if target_concept == GradCamTargetConcept.EMBEDDING:
        num_targets = model.image_encoder.get_embedding_dim()
        return [ScoreOutputTarget(target_concept, i) for i in range(num_targets)]
    elif target_concept == GradCamTargetConcept.ACTIONS:
        num_targets = model_config.num_actions
        return [ScoreOutputTarget(target_concept, i) for i in range(num_targets)]
    elif target_concept == GradCamTargetConcept.MOVEMENT:
        return [ScoreOutputTarget(GradCamTargetConcept.ACTIONS, 0), ScoreOutputTarget(GradCamTargetConcept.ACTIONS, 1)]
    elif target_concept == GradCamTargetConcept.ATTACK:
        return [ScoreOutputTarget(GradCamTargetConcept.ACTIONS, 5)]
    else:
        raise ValueError(f"Unknown target concept {target_concept}")


class GradCamPreTrainedModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        unfreeze_model(self.model)

    def get_embedding_dim(self):
        return self.model.get_embedding_dim()

    def forward(self, x):
        # ensure pretrained encoders allow gradient flow
        return self.model(x, no_grad=False)


class GradCamModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        # concatenate action logits
        action_logits = list(self.model(x))
        logits = torch.concatenate(action_logits, dim=-1)
        return logits


def vit_reshape_transform(reshape_dim, tensor):
    if reshape_dim == 0:
        tensor = tensor[1:, :, :].transpose(0, 1)
    elif reshape_dim == 1:
        tensor = tensor[:, 1:, :]
    dim = int(np.sqrt(tensor.size(1)))
    result = tensor.reshape(tensor.size(0), dim, dim, tensor.size(2))
    # Bring the channels to the first dimension like in CNNs.
    result = result.permute(0, 3, 1, 2)
    return result


def get_target_layers_and_reshape_transform(encoder_model, model_config):
    if "pretrained_encoder" in model_config and model_config.pretrained_encoder is not None:
        # pretrained encoder
        if model_config.pretrained_encoder.family == "clip":
            if model_config.pretrained_encoder.name == "RN50":
                return [encoder_model.model.model.visual.layer4[-1]], None
            elif model_config.pretrained_encoder.name == "ViT-B/16":
                return [encoder_model.model.model.visual.transformer.resblocks[-1].ln_1], partial(vit_reshape_transform, 0)
            elif model_config.pretrained_encoder.name == "ViT-L/14":
                return [encoder_model.model.model.visual.transformer.resblocks[-1].ln_1], partial(vit_reshape_transform, 0)
            else:
                raise ValueError(f"Grad-Cam not supported for {model_config.pretrained_encoder.name} CLIP encoder.")
        elif model_config.pretrained_encoder.family == "dino":
            return [encoder_model.model.model.blocks[-1].norm1], partial(vit_reshape_transform, 1)
        elif model_config.pretrained_encoder.family == "focal":
            return [encoder_model.model.model.layers_3.blocks[-1].norm1], None
        elif model_config.pretrained_encoder.family == "stablediffusion":
            return [encoder_model.model.model.encoder.mid_block.resnets[-1]], None
        else:
            raise ValueError(f"Grad-Cam not supported for pretrained {model_config.pretrained_encoder.family} encoder.")
    else:
        # end-to-end encoder
        if model_config.encoder.type == "impala_resnet":
            return [encoder_model.encoder.encoder.blocks[-1]], None
        elif model_config.encoder.type == "resnet":
            # unsure which block
            return [encoder_model.encoder.model.block4], None
        elif model_config.encoder.type == "vit":
            # unsure which part
            return [encoder_model.encoder.model.transformer.layers[-1][0].norm], partial(vit_reshape_transform, 1)
        else:
            raise ValueError(f"Grad-Cam not supported for end-to-end {model_config.encoder.type} encoder.")


def get_grad_cam_image(image, model, model_config, target_concept, save_path=None):
    if model_config.pretrained_encoder is not None:
        model.image_encoder = GradCamPreTrainedModelWrapper(model.image_encoder)

    np_preprocessing, _ = get_preprocessing_function_and_image_shape(
        model_config.pretrained_encoder is not None,
        model_config.pretrained_encoder,
        model_config.image_width,
        model_config.image_height,
    )

    # resize image and convert to tensor
    image = np_preprocessing(image)
    image_tensor = torch.tensor(image, dtype=torch.uint8, device=DEVICE)
    image_tensor = image_tensor.reshape(1, 1, *image_tensor.shape)

    targets = get_grad_cam_targets(model, model_config, target_concept)
    target_layers, reshape_transform = get_target_layers_and_reshape_transform(model.image_encoder, model_config)

    if target_concept == GradCamTargetConcept.EMBEDDING:
        forward_model = model.image_encoder
    else:
        forward_model = GradCamModelWrapper(model)
    # ensure model is on right device, in training mode and at full precision if no GPU is used
    forward_model = forward_model.train()
    forward_model = forward_model.to(DEVICE)
    if DEVICE == "cpu":
        forward_model = forward_model.to(dtype=torch.float32)

    cam = GradCAM(model=forward_model, target_layers=target_layers, use_cuda=DEVICE == "cuda", reshape_transform=reshape_transform)
    grayscale_cam = cam(input_tensor=image_tensor, targets=targets)
    grayscale_cam = grayscale_cam[0]

    # get normalised image and transpose to (height, width, channels)
    image_array = (image.astype(np.float32) / 255.0).transpose(1, 2, 0)
    visualization = show_cam_on_image(image_array, grayscale_cam, use_rgb=True)
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
    render_image(visualization, reshape=False, save_path=save_path)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--checkpoint_path", type=str, required=True, help="Path to model checkpoint")
    argparser.add_argument("--image_path", type=str, required=True, help="Path to image")
    argparser.add_argument("--save_path", type=str, default=None, help="Path to the directory where the image will be saved")
    argparser.add_argument(
        "--target_concept",
        type=GradCamTargetConcept,
        choices=[
            GradCamTargetConcept.EMBEDDING,
            GradCamTargetConcept.ACTIONS,
            GradCamTargetConcept.MOVEMENT,
            GradCamTargetConcept.ATTACK,
        ],
        default=GradCamTargetConcept.ACTIONS,
        help="Target concept for grad cam",
    )
    args = argparser.parse_args()

    # load model
    checkpoint_path = Path(args.checkpoint_path)
    assert checkpoint_path.exists() and checkpoint_path.is_file(), f"Checkpoint path {checkpoint_path} does not exist"
    model, model_config = load_checkpoint(checkpoint_path, eval_mode=True, device=DEVICE)

    # load image
    image_path = Path(args.image_path)
    assert image_path.exists() and image_path.is_file(), f"Image path {image_path} does not exist"
    image = load_image(image_path)

    get_grad_cam_image(image, model, model_config, args.target_concept, args.save_path)
