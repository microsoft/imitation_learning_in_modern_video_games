from pixelbc.models.encoders.encoders import get_image_encoder
from pixelbc.models.encoders.pretrained_encoders import get_pretrained_encoder


def get_encoder(image_shape, encoder_config, pretrained_encoder_config):
    if pretrained_encoder_config is not None:
        image_encoder = get_pretrained_encoder(pretrained_encoder_config)
    else:
        assert encoder_config is not None, "Encoder config must be provided for default encoders."
        image_encoder = get_image_encoder(image_shape, encoder_config)
    return image_encoder
