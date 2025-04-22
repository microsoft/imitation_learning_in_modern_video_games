# Image augmentations for torch.Tensor. This only handles normalisation and image augmentations!
# Reshaping and cropping is handled by the data processing.
import torch
from torchvision import transforms

CLIP_NORMALISATION_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_NORMALISATION_STD = (0.26862954, 0.26130258, 0.27577711)
IMAGENET_NORMALISATION_MEAN = (0.485, 0.456, 0.406)
IMAGENET_NORMALISATION_STD = (0.229, 0.224, 0.225)

# image augmentation settings of VPT and taken from human-modelling
DEFAULT_COLOR_JITTER_STRENGTH = 0.2
DEFAULT_ROTATION_DEGREE = 2.0
DEFAULT_TRANSLATION_PERCENT = 0.02
DEFAULT_SCALE_FACTOR = 0.02
DEFAULT_SHEAR_DEGREE = 2.0


def get_image_augmentation_transforms(
    color_jitter_strength=DEFAULT_COLOR_JITTER_STRENGTH,
    rotation_degree=DEFAULT_ROTATION_DEGREE,
    translation_percent=DEFAULT_TRANSLATION_PERCENT,
    scale_factor=DEFAULT_SCALE_FACTOR,
    shear_degree=DEFAULT_SHEAR_DEGREE,
):
    """
    Get image augmentation transformations. Same as for VPT paper.
    :param color_jitter_strength: The strength of the color jittering.
    :param rotation_degree: The degree of rotation.
    :param translation_percent: The percent of translation.
    :param scale_factor: The scale factor.
    :param shear_degree: The degree of shearing.
    :return: List of image augmentation transformations.
    """
    return [
        transforms.ColorJitter(
            brightness=color_jitter_strength,
            contrast=color_jitter_strength,
            saturation=color_jitter_strength,
            hue=color_jitter_strength,
        ),
        transforms.RandomAffine(
            degrees=rotation_degree,
            translate=(translation_percent, translation_percent),
            scale=(1 - scale_factor, 1 + scale_factor),
            shear=shear_degree,
        ),
    ]


class ToFloatNormalise(object):
    """Takes in uint8 torch tensor and normalises to [0, 1] float32 torch tensor"""

    def __call__(self, img):
        assert torch.is_tensor(img)
        assert img.dtype == torch.uint8
        img = img.float().div_(255)
        return img


def default_transform(use_image_augmentation):
    """
    Get image transformation pipeline for default training.
    :param image_height: The target height.
    :param image_width: The target width.
    :param use_image_augmentation: Whether to use VPT-style image augmentation or not.
    :return: The torchvision transform.
    """
    transformations = []
    if use_image_augmentation:
        print("Use image augmentation ...")
        transformations += get_image_augmentation_transforms()
    transformations.append(ToFloatNormalise())
    return transforms.Compose(transformations)


def clip_transform():
    """
    Get image transformation normalisation pipeline for OpenAI CLIP models, source:
    https://github.com/openai/CLIP/blob/main/clip/clip.py
    :return: The torchvision transform.
    """
    return transforms.Compose(
        [
            ToFloatNormalise(),
            transforms.Normalize(mean=CLIP_NORMALISATION_MEAN, std=CLIP_NORMALISATION_STD),
        ]
    )


def dino_transform():
    """
    Get the transformation pipeline for normalisation for DINOv2 models, source:
    https://github.com/facebookresearch/dinov2/blob/main/dinov2/data/transforms.py#L78
    :return: The torchvision transform.
    """
    return transforms.Compose(
        [
            ToFloatNormalise(),
            transforms.Normalize(mean=IMAGENET_NORMALISATION_MEAN, std=IMAGENET_NORMALISATION_STD),
        ]
    )


def focalnet_transform():
    """
    Get the transformation pipeline for normalisation of FocalNet models, source:
    https://github.com/microsoft/FocalNet/blob/main/data/build.py#L123
    :return: The torchvision transform.
    """
    return transforms.Compose(
        [
            ToFloatNormalise(),
            transforms.Normalize(mean=IMAGENET_NORMALISATION_MEAN, std=IMAGENET_NORMALISATION_STD),
        ]
    )


def stablediffusion_transform():
    """
    Get the transformation pipeline for normalisation of Stable Diffusion models.
    :return: The torchvision transform.
    """
    return ToFloatNormalise()
