from abc import ABC, abstractmethod

import torch
import torch.nn.functional as F
from torch import nn


class BackboneModel(ABC):
    def init_for_sequence(self, batch_size):
        pass

    @abstractmethod
    def forward(self, x, rollout=False):
        raise NotImplementedError


class MLP(nn.Module, BackboneModel):
    def __init__(self, input_size, hidden_size, num_layers, output_size=None, **kwargs):
        """
        Multi-layer perceptron with ReLU activations.
        :param input_size: input size
        :param hidden_size: hidden size
        :param num_layers: number of hidden layers
        :param output_size: output size (if None, no output layer is added)
        """
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_dim = output_size if output_size is not None else hidden_size

        layers = [nn.Linear(input_size, hidden_size), nn.ReLU()]
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ReLU())
        if output_size is not None:
            layers.append(nn.Linear(hidden_size, output_size))

        self.model = nn.Sequential(*layers)

    def forward(self, x, rollout=False):
        """Flatten input for and then feed through MLP before reshaping to original shape.

        :param x: input tensor of shape (batch_size, ..., input_size)
        :param rollout: whether to use rollout mode (not used)
        :return: output tensor of shape (batch_size, ..., output_size)
        """
        num_input_dims = 0
        input_dim = 1
        for dim in x.shape[::-1]:
            num_input_dims += 1
            input_dim *= dim
            if input_dim == self.input_size:
                break

        batch_shape = x.shape[:-num_input_dims]
        x = x.reshape(-1, self.input_size)
        x = self.model(x)
        x = x.reshape(*batch_shape, self.output_dim)
        return x


class LSTM(nn.Module, BackboneModel):
    def __init__(self, input_size, lstm_hidden_size, lstm_num_layers, num_layers, output_size=None, **kwargs):
        """
        Multi-layer LSTM followed by single layer MLP model.
        :param input_size: input size
        :param lstm_hidden_size: hidden size of LSTM and MLPs
        :param lstm_num_layers: number of hidden layers of LSTM
        :param num_layers: number of hidden layers of MLP before LSTM
        :param output_size: output size (if None, no output layer is added)
        """
        super().__init__()
        self.input_size = input_size
        self.hidden_size = lstm_hidden_size
        self.lstm_num_layers = lstm_num_layers
        self.mlp_num_layers = num_layers
        self.output_dim = output_size if output_size is not None else lstm_hidden_size

        self.mlp_in = MLP(input_size, lstm_hidden_size, num_layers)
        self.lstm = nn.LSTM(lstm_hidden_size, lstm_hidden_size, lstm_num_layers, batch_first=True)
        self.mlp_out = MLP(lstm_hidden_size, lstm_hidden_size, 1, output_size=output_size)

        self.hidden_state = None
        self.cell_state = None

    def init_for_sequence(self, batch_size):
        current_device = next(self.parameters()).device
        self.hidden_state = torch.zeros(self.lstm_num_layers, batch_size, self.hidden_size).to(current_device)
        self.cell_state = torch.zeros(self.lstm_num_layers, batch_size, self.hidden_size).to(current_device)

    def forward(self, x, rollout=False):
        verify_input_shape(self.input_size, x.shape)
        batch_size, seq_len = x.shape[:2]
        x = x.reshape(batch_size, seq_len, self.input_size)

        if self.hidden_state is None:
            self.init_for_sequence(batch_size)
        assert (
            self.cell_state.shape[1] == self.hidden_state.shape[1] == batch_size
        ), f"Hidden state and cell state batch size must match input batch size {batch_size}, but was "
        +f"{self.hidden_state.shape[1]} and {self.cell_state.shape[1]}"

        x = self.mlp_in(x)
        x, (self.hidden_state, self.cell_state) = self.lstm(x, (self.hidden_state, self.cell_state))
        x = F.relu(x)
        x = self.mlp_out(x)

        return x


def verify_input_shape(input_size, input_shape):
    assert input_shape[-1] == input_size, f"Input shape {input_shape} does not match input size {input_size}"
    assert len(input_shape) == 3, "Input shape must be (batch_size, sequence_length, input_size)"


def count_parameters(model):
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    return trainable_params, total_params
