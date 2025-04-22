import cv2
import numpy as np

DEFAULT_FRAMESTACKING = 1


def load_image(path):
    """
    Load an image from a path and return as RGB.
    :param path: The path to the image.
    :return: The loaded image as RGB.
    """
    # load image as BGR
    img = cv2.imread(str(path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def render_image(img, reshape=True, save_path=None):
    """
    Render an image.
    :param img: The image to render as (C, H, W) RGB.
    :param reshape: Whether to reshape the image to (H, W, C) RGB.
    :param save_path: The path to save the rendered image to.
    :return: The rendered image.
    """
    import matplotlib.pyplot as plt

    if reshape:
        img = np.transpose(img, (1, 2, 0))
    plt.imshow(img)
    plt.axis("off")
    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight")
    else:
        plt.show()
