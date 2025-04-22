from pixelbc.models.utils.model_utils import LSTM, MLP
from pixelbc.models.utils.pico_gpt import GPTTransformer


def get_model(input_size, sequence_length, output_size, model_config):
    """
    Get model from config.
    :param input_size: input size
    :param sequence_length: sequence length during training
    :param output_size: output size
    :param model_config: model config
    :return: model
    """
    assert "type" in model_config, "Model config is missing 'type' field"
    model_type = model_config["type"]
    if model_type == "mlp":
        return MLP(input_size, output_size=output_size, **model_config)
    elif model_type == "lstm":
        return LSTM(input_size, output_size=output_size, **model_config)
    elif model_type == "gpt":
        return GPTTransformer(
            input_size,
            output_size=output_size,
            num_layers=model_config.gpt_num_layers,
            num_heads=model_config.gpt_num_heads,
            embedding_dim=model_config.gpt_embedding_dim,
            sequence_length=sequence_length,
            use_positional_encoding=model_config.gpt_use_positional_encoding,
            bias=model_config.gpt_bias,
            is_causal=model_config.gpt_is_causal,
        )
    else:
        raise ValueError(f"Unknown model type {model_type}")
