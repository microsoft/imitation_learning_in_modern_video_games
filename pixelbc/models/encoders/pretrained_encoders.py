import warnings
from contextlib import nullcontext

import clip
import timm
import torch
from diffusers import AutoencoderKL

from pixelbc.models.encoders.encoders import ImageEncoder
from pixelbc.models.utils.image_augmentations import (
    clip_transform, dino_transform, focalnet_transform,
    stablediffusion_transform)

# dictionary mapping pretrained encoder names to their family and model name
PRETRAINED_ENCODERS = {
    "dinov2-vits14": ("dino", "dinov2_vits14"),
    "dinov2-vitb14": ("dino", "dinov2_vitb14"),
    "dinov2-vitl14": ("dino", "dinov2_vitl14"),
    "dinov2-vitg14": ("dino", "dinov2_vitg14"),
    "clip-rn50": ("clip", "RN50"),
    "clip-rn50x4": ("clip", "RN50x4"),
    "clip-rn50x16": ("clip", "RN50x16"),
    "clip-rn101": ("clip", "RN101"),
    "clip-vitb32": ("clip", "ViT-B/32"),
    "clip-vitb16": ("clip", "ViT-B/16"),
    "clip-vitl14": ("clip", "ViT-L/14"),
    "focalnet-large-fl3": ("focal", "focalnet_large_fl3"),
    "focalnet-large-fl4": ("focal", "focalnet_large_fl4"),
    "focalnet-xlarge-fl3": ("focal", "focalnet_xlarge_fl3"),
    "focalnet-xlarge-fl4": ("focal", "focalnet_xlarge_fl4"),
    "focalnet-huge-fl3": ("focal", "focalnet_huge_fl3"),
    "focalnet-huge-fl4": ("focal", "focalnet_huge_fl4"),
    "stablediffusion-vae2.1": ("stablediffusion", "vae2.1"),
}

IMAGE_SIZE_BY_CLIP_MODEL_NAME = {
    "RN50": 224,
    "RN101": 224,
    "RN50x4": 288,
    "RN50x16": 384,
    "RN50x64": 448,
    "ViT-B/32": 224,
    "ViT-B/16": 224,
    "ViT-L/14": 224,
    "ViT-L/14@336px": 336,
}
EMBEDDING_SIZE_BY_CLIP_MODEL_NAME = {
    "RN50": 1024,
    "RN101": 512,
    "RN50x4": 640,
    "RN50x16": 768,
    "RN50x64": 1024,
    "ViT-B/32": 512,
    "ViT-B/16": 512,
    "ViT-L/14": 768,
    "ViT-L/14@336px": 768,
}

DINO_IMAGE_SIZE = 256
DINO_CROP_SIZE = 224
EMBEDDING_SIZE_BY_DINO_MODEL_NAME = {
    "dinov2_vits14": 384,
    "dinov2_vitb14": 768,
    "dinov2_vitl14": 1024,
    "dinov2_vitg14": 1536,
}

FOCALNET_IMAGE_SIZE = 224
EMBEDDING_SIZE_BY_FOCALNET_MODEL_NAME = {
    "focalnet_large_fl3": 1536,  # 205M params
    "focalnet_large_fl4": 1536,  # 205M params
    "focalnet_xlarge_fl3": 2048,  # 364M params
    "focalnet_xlarge_fl4": 2048,  # 364M params
    "focalnet_huge_fl3": 2816,  # 683M params
    "focalnet_huge_fl4": 2816,  # 683M params
}

STABLE_DIFFUSION_IMAGE_SIZE = 256
STABLE_DIFFUSION_KWARGS_BY_MODEL_NAME = {
    "vae2.1": {
        "pretrained_model_name_or_path": "stabilityai/stable-diffusion-2-1",
        "subfolder": "vae",
    }
}
EMBEDDING_SIZE_BY_STABLE_DIFFUSION_MODEL_NAME = {
    "vae2.1": 4 * 32 * 32,
}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def get_supported_pretrained_encoders():
    return list(PRETRAINED_ENCODERS.values())


def get_embedding_size(encoder_family, encoder_name):
    if encoder_family == "clip":
        return EMBEDDING_SIZE_BY_CLIP_MODEL_NAME[encoder_name]
    elif encoder_family == "dino":
        return EMBEDDING_SIZE_BY_DINO_MODEL_NAME[encoder_name]
    elif encoder_family == "focal":
        return EMBEDDING_SIZE_BY_FOCALNET_MODEL_NAME[encoder_name]
    elif encoder_family == "stablediffusion":
        return EMBEDDING_SIZE_BY_STABLE_DIFFUSION_MODEL_NAME[encoder_name]
    else:
        raise ValueError(f"Unsupported encoder family {encoder_family}.")


def get_image_input_size(encoder_family, encoder_name):
    if encoder_family == "clip":
        return IMAGE_SIZE_BY_CLIP_MODEL_NAME[encoder_name]
    elif encoder_family == "dino":
        return DINO_CROP_SIZE
    elif encoder_family == "focal":
        return FOCALNET_IMAGE_SIZE
    elif encoder_family == "stablediffusion":
        return STABLE_DIFFUSION_IMAGE_SIZE
    else:
        raise ValueError(f"Unsupported encoder family {encoder_family}.")


def freeze_model(model):
    for param in model.parameters():
        param.requires_grad = False


def unfreeze_model(model):
    for param in model.parameters():
        param.requires_grad = True


class PretrainedEncoderWrapper(ImageEncoder):
    def __init__(self, model, process, embedding_size):
        super().__init__()
        self.model = model
        self.process = process
        self.embedding_size = embedding_size
        freeze_model(self.model)
        model.eval()

    def get_embedding_dim(self):
        return self.embedding_size

    def reshape_and_process_input(self, x):
        """
        Reshape the input to a 5D tensor of shape (batch_size, sequence_length, channels, height, width) and apply the preprocessing function.
        """
        assert len(x.shape) == 5, f"Expected input shape (batch_size, sequence_length, channels, height, width), but got {x.shape}."
        batch_size, sequence_length, *image_shape = x.shape
        x = x.reshape(batch_size * sequence_length, *image_shape)
        x = self.process(x)
        assert torch.is_tensor(x) and x.dtype == torch.float32, "ImageEncoderWrapper expects float32 tensor"
        if x.sum() == 0:
            warnings.warn("Image augmentation process returned all 0s! If the screen was not fully black, this is likely a bug!")
        return x, (batch_size, sequence_length, *image_shape)

    def reshape_output(self, x, shape):
        """
        Reshape the output to the shape of the input with (channels, height, width) replaced by (embedding_size,)
        """
        batch_size, sequence_length, *image_shape = shape
        return x.reshape(batch_size, sequence_length, -1)

    def forward(self, x, no_grad=True):
        x, shape = self.reshape_and_process_input(x)
        with torch.no_grad() if no_grad else nullcontext():
            embedding = self.model(x).float()
        embedding = self.reshape_output(embedding, shape)
        return embedding.detach() if no_grad else embedding


class PretrainedClipEncoderWrapper(PretrainedEncoderWrapper):
    def forward(self, x, no_grad=True):
        x, shape = self.reshape_and_process_input(x)
        with torch.no_grad() if no_grad else nullcontext():
            embedding = self.model.encode_image(x).float()
        embedding = self.reshape_output(embedding, shape)
        return embedding.detach() if no_grad else embedding


class PretrainedFocalNetEncoderWrapper(PretrainedEncoderWrapper):
    def forward(self, x, no_grad=True):
        x, shape = self.reshape_and_process_input(x)
        with torch.no_grad() if no_grad else nullcontext():
            # (batch_size, sequence_length, embedding_size, height, width)
            embedding = self.model(x)[-1]
        # average pooling over height and width
        embedding = embedding.mean(dim=(-1, -2))
        embedding = self.reshape_output(embedding, shape)
        return embedding.detach() if no_grad else embedding


class PretrainedStableDiffusionEncoderWrapper(PretrainedEncoderWrapper):
    def forward(self, x, no_grad=True):
        x, shape = self.reshape_and_process_input(x)
        with torch.no_grad() if no_grad else nullcontext():
            embedding_dist = self.model.encode(x).latent_dist
        # embedding = embedding_dist.sample()
        embedding = embedding_dist.mean
        embedding = self.reshape_output(embedding, shape)
        return embedding.detach() if no_grad else embedding


def is_supported_encoder(encoder_family, encoder_name):
    supported = False
    for family, name in get_supported_pretrained_encoders():
        if encoder_family == family and encoder_name == name:
            supported = True
            break
    return supported


def get_clip_encoder(model_name):
    model, _ = clip.load(model_name)
    # remove parts of the model that are not needed for encoding images
    # del model.transformer
    # del model.token_embedding
    # del model.positional_embedding
    # del model.ln_final
    # del model.text_projection
    return PretrainedClipEncoderWrapper(model, clip_transform(), EMBEDDING_SIZE_BY_CLIP_MODEL_NAME[model_name])


def get_dino_encoder(model_name):
    model = torch.hub.load("facebookresearch/dinov2", model_name, pretrained=True)
    return PretrainedEncoderWrapper(model, dino_transform(), EMBEDDING_SIZE_BY_DINO_MODEL_NAME[model_name])


def get_focalnet_encoder(model_name):
    model = timm.create_model(model_name, pretrained=True, features_only=True)
    return PretrainedFocalNetEncoderWrapper(model, focalnet_transform(), EMBEDDING_SIZE_BY_FOCALNET_MODEL_NAME[model_name])


def get_stable_diffusion_encoder(model_name):
    model = AutoencoderKL.from_pretrained(**STABLE_DIFFUSION_KWARGS_BY_MODEL_NAME[model_name]).to(DEVICE)
    # save memory by not keeping decoder
    del model.decoder
    return PretrainedStableDiffusionEncoderWrapper(model, stablediffusion_transform(), EMBEDDING_SIZE_BY_STABLE_DIFFUSION_MODEL_NAME[model_name])


def get_pretrained_encoder(pretrained_encoder_config):
    assert (
        "family" in pretrained_encoder_config and "name" in pretrained_encoder_config
    ), "Pretrained encoder config must contain the family and name of the model."
    encoder_family = pretrained_encoder_config["family"]
    encoder_name = pretrained_encoder_config["name"]

    if not is_supported_encoder(encoder_family, encoder_name):
        print(f"Error: Unsupported pretrained encoder family={encoder_family}, name={encoder_name}.")
        print("All supported pretrained encoders:")
        for family, name in get_supported_pretrained_encoders():
            print(f"\tfamily={family}, name={name}")
        raise ValueError("Unsupported pretrained encoder!")

    if encoder_family == "clip":
        return get_clip_encoder(encoder_name)
    elif encoder_family == "dino":
        return get_dino_encoder(encoder_name)
    elif encoder_family == "focal":
        return get_focalnet_encoder(encoder_name)
    elif encoder_family == "stablediffusion":
        return get_stable_diffusion_encoder(encoder_name)
    else:
        raise ValueError(f"Unsupported pretrained encoder family {encoder_family}.")
