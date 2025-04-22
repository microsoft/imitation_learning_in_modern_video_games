# Image preprocessing for np.array. This only handles resizing and cropping! Further augmentations including normalisation and image augmentations
# are handled in the encoders.
from functools import partial

import cv2
import numpy as np
from albumentations import Compose, augmentations

from pixelbc.models.encoders.pretrained_encoders import (
    DINO_CROP_SIZE, DINO_IMAGE_SIZE, FOCALNET_IMAGE_SIZE,
    IMAGE_SIZE_BY_CLIP_MODEL_NAME, STABLE_DIFFUSION_IMAGE_SIZE)


def albumentations_preprocess(transform, img):
    """
    Apply the albumentations transformation on images.
    :param transform: The albumentations transform to use.
    :param img: The image to preprocess as np.array in RGB format.
    :return: The preprocessed image as RGB np.array.
    """
    return np.transpose(transform(image=img)["image"], (2, 0, 1)).astype(np.uint8)


def default_resize(image_height, image_width):
    """
    Get image resizing pipeline for default training.
    :param image_height: The target height.
    :param image_width: The target width.
    :return: The albumentations transform.
    """
    return augmentations.Resize(image_height, image_width, interpolation=cv2.INTER_LINEAR)


def get_default_resize_and_image_shape(image_height, image_width):
    """
    Get the default resizing function and image shape.
    :param image_height: The height of the image for default preprocessing.
    :param image_width: The width of the image for default preprocessing.
    :return: The preprocessing function, and output image shape as (C, image_height, image_width).
    """
    transform = default_resize(image_height, image_width)
    return partial(albumentations_preprocess, transform), (3, image_height, image_width)


def clip_resize(image_size):
    """
    Get image resizing pipeline for OpenAI CLIP models, source:
    https://github.com/openai/CLIP/blob/main/clip/clip.py
    :param image_size: The desired size of the image (width and height).
    :return: The albumentations transform.
    """
    return Compose(
        [
            augmentations.Resize(image_size, image_size, interpolation=cv2.INTER_CUBIC),
            augmentations.CenterCrop(image_size, image_size),
        ]
    )


def get_clip_resize_and_image_shape(clip_model_name):
    """
    Get the preprocessing pipeline and image shape for a given CLIP model name.
    :param clip_model_name: The CLIP model name.
    :return: The preprocessing function, and output image shape as (C, image_height, image_width).
    """
    assert clip_model_name in IMAGE_SIZE_BY_CLIP_MODEL_NAME, "The clip_model_name is not valid."
    image_size = IMAGE_SIZE_BY_CLIP_MODEL_NAME[clip_model_name]
    transform = clip_resize(image_size)
    return partial(albumentations_preprocess, transform), (3, image_size, image_size)


def dino_resize():
    """
    Get the resizing pipeline for DINOv2 models, source:
    https://github.com/facebookresearch/dinov2/blob/main/dinov2/data/transforms.py#L78
    :return: The albumentations transform.
    """
    return Compose(
        [
            augmentations.Resize(DINO_IMAGE_SIZE, DINO_IMAGE_SIZE, interpolation=cv2.INTER_CUBIC),
            augmentations.CenterCrop(DINO_CROP_SIZE, DINO_CROP_SIZE),
        ]
    )


def get_dino_resize_and_image_shape():
    """
    Get the preprocessing function and image shape for DINOv2.
    :return: The preprocessing function, and output image shape as (C, image_height, image_width).
    """
    return partial(albumentations_preprocess, dino_resize()), (3, DINO_CROP_SIZE, DINO_CROP_SIZE)


def focalnet_resize():
    """
    Get the resizing pipeline for FocalNet models, source:
    https://github.com/microsoft/FocalNet/blob/main/data/build.py#L123
    :return: The albumentations transform.
    """
    return augmentations.Resize(FOCALNET_IMAGE_SIZE, FOCALNET_IMAGE_SIZE, interpolation=cv2.INTER_CUBIC)


def get_focalnet_resize_and_image_shape():
    """
    Get the preprocessing function and image shape for FocalNet.
    :return: The preprocessing function, and output image shape as (C, image_height, image_width).
    """
    return partial(albumentations_preprocess, focalnet_resize()), (3, FOCALNET_IMAGE_SIZE, FOCALNET_IMAGE_SIZE)


def stablediffusion_resize():
    """
    Get the resizing pipeline for stable diffusion models, source:
    :return: The albumentations transform.
    """
    return augmentations.Resize(STABLE_DIFFUSION_IMAGE_SIZE, STABLE_DIFFUSION_IMAGE_SIZE, interpolation=cv2.INTER_CUBIC)


def get_stablediffusion_resize_and_image_shape():
    """
    Get the preprocessing function and image shape for stable diffusion.
    :return: The preprocessing function, and output image shape as (C, image_height, image_width).
    """
    return partial(albumentations_preprocess, stablediffusion_resize()), (
        3,
        STABLE_DIFFUSION_IMAGE_SIZE,
        STABLE_DIFFUSION_IMAGE_SIZE,
    )


def get_preprocessing_function_and_image_shape(use_pretrained_encoder, pretrained_encoder_config, image_height, image_width):
    """
    Get the preprocessing function and image shape for a given encoder type.
    :param use_pretrained_encoder: Whether to use a pretrained encoder or not.
    :param pretrained_encoder_config: The config for pretrained encoders.
    :param image_height: The height of the image (only for default preprocessing).
    :param image_width: The width of the image (only for default preprocessing).
    :return: The preprocessing function, and output image shape as (C, image_height, image_width).
    """
    if use_pretrained_encoder:
        assert pretrained_encoder_config is not None, "Pretrained encoder config must be provided for pretrained encoders."
        assert (
            "family" in pretrained_encoder_config and "name" in pretrained_encoder_config
        ), "Pretrained encoder config must contain the family and name of the model."
        if pretrained_encoder_config["family"] == "clip":
            print("Use OpenAI CLIP style preprocessing for images ...")
            return get_clip_resize_and_image_shape(pretrained_encoder_config["name"])
        elif pretrained_encoder_config["family"] == "dino":
            print("Use DINOv2 style preprocessing for images ...")
            return get_dino_resize_and_image_shape()
        elif pretrained_encoder_config["family"] == "focal":
            print("Use FocalNet style preprocessing for images ...")
            return get_focalnet_resize_and_image_shape()
        elif pretrained_encoder_config["family"] == "stablediffusion":
            print("Use Stable Diffusion style preprocessing for images ...")
            return get_stablediffusion_resize_and_image_shape()
        else:
            raise ValueError(f"Unsupported pretrained encoder family {pretrained_encoder_config['family']}.")
    else:
        print("Use default preprocessing for images ...")
        return get_default_resize_and_image_shape(image_height, image_width)
