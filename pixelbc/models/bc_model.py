import warnings

import lightning.pytorch as pl
import torch
from lightning.pytorch.utilities import grad_norm
from torch.distributions import Bernoulli, Categorical

from pixelbc.models.encoders import get_encoder
from pixelbc.models.utils import get_model
from pixelbc.models.utils.loss_utils import (compute_button_loss,
                                             compute_joystick_loss,
                                             compute_trigger_loss)
from pixelbc.models.utils.train_metrics import (
    get_button_metrics, get_continuous_joystick_metrics,
    get_discrete_joystick_metrics, get_trigger_metrics)


class BCModel(pl.LightningModule):
    def __init__(
        self,
        image_shape,
        sequence_length,
        num_actions,
        num_joystick_actions,
        num_trigger_actions,
        discretise_joystick,
        discretise_joystick_bins,
        encoder,
        pretrained_encoder,
        bc_policy,
        learning_rate,
        joystick_loss_weight,
        button_loss_weight,
        train_from_embeddings,
        lr_warmup_steps,
        **kwargs,
    ):
        super().__init__()
        self.image_shape = image_shape
        self.sequence_length = sequence_length
        self.num_actions = num_actions
        self.num_joystick_actions = num_joystick_actions
        self.num_trigger_actions = num_trigger_actions
        self.num_button_actions = num_actions - num_joystick_actions - num_trigger_actions
        self.discretise_joystick = discretise_joystick
        self.num_discretisation_bins = len(discretise_joystick_bins)
        self.discretise_joystick_bins = torch.Tensor(discretise_joystick_bins).float()
        self.encoder_config = encoder
        self.pretrained_encoder_config = pretrained_encoder
        self.bc_policy_config = bc_policy
        self.learning_rate = learning_rate
        self.joystick_loss_weight = joystick_loss_weight
        self.button_loss_weight = button_loss_weight
        self.train_from_embeddings = train_from_embeddings
        self.lr_warmup_steps = lr_warmup_steps

        if self.bc_policy_config.type == "gpt":
            assert self.lr_warmup_steps > 0, "GPT policy requires LR warmup to be enabled."

        if self.train_from_embeddings:
            assert self.pretrained_encoder_config is not None, "Training from embeddings is only supported for pretrained encoders."

        # define encoder
        self.image_encoder = get_encoder(image_shape, encoder, pretrained_encoder)
        self.image_encoding_dim = self.image_encoder.get_embedding_dim()

        # define BC models
        if self.discretise_joystick:
            self.num_joystick_logits = self.num_discretisation_bins * self.num_joystick_actions
        else:
            self.num_joystick_logits = self.num_joystick_actions
        self.num_trigger_logits = self.num_trigger_actions
        self.num_button_logits = self.num_button_actions
        self.num_logits = self.num_joystick_logits + self.num_trigger_logits + self.num_button_logits
        self.bc_model = get_model(self.image_encoding_dim, self.sequence_length, self.num_logits, self.bc_policy_config)

        # save extra hyperparameters
        self.hparams.image_encoding_dim = self.image_encoding_dim
        self.hparams.num_logits = self.num_logits
        self.save_hyperparameters()

    def _split_action_targets(self, action_targets):
        """
        Split action targets into joystick, trigger, and button targets.
        :param action_targets: action targets as tensor
        :return: flattened joystick targets, trigger targets, and button targets
        """
        assert action_targets.shape[-1] == self.num_actions, "Number of targets does not match number of actions."
        action_targets = action_targets.reshape(-1, self.num_actions)
        joystick_targets = action_targets[:, : self.num_joystick_actions]
        trigger_targets = action_targets[:, self.num_joystick_actions : self.num_joystick_actions + self.num_trigger_actions]
        button_targets = action_targets[:, self.num_joystick_actions + self.num_trigger_actions :]
        return joystick_targets, trigger_targets, button_targets

    def _split_action_logits(self, action_logits):
        """
        Split action logits into joystick, trigger, and button logits.
        :param action_logits: action logits as tensor
        :return: joystick logits, trigger logits, and button logits
        """
        assert action_logits.shape[-1] == self.num_logits, "Number of logits does not match number of actions."
        action_logits = action_logits.reshape(-1, self.num_logits)

        # extract joystick logits
        joystick_logits = action_logits[:, : self.num_joystick_logits]
        if self.discretise_joystick:
            joystick_logits = joystick_logits.reshape(-1, self.num_discretisation_bins, self.num_joystick_actions)
        else:
            joystick_logits = joystick_logits.reshape(-1, self.num_joystick_actions)

        # extract trigger logits
        if self.num_trigger_actions > 0:
            trigger_logits = action_logits[:, self.num_joystick_logits : self.num_joystick_logits + self.num_trigger_logits]
        else:
            trigger_logits = None

        # extract button logits
        button_logits = action_logits[:, self.num_joystick_logits + self.num_trigger_logits :]

        return joystick_logits, trigger_logits, button_logits

    def _get_embedding(self, x):
        """
        Reshape input to have batch size as first dimension and then input dimensions and embed if image is given.
        :param x: input image or embedding (if train_from_embeddings is True)
            (batch_size, sequence_length, channels, height, width) if image is given
            (batch_size, sequence_length, embedding_dim) if embedding is given
        :return: image embedding
        """
        return x if self.train_from_embeddings else self.image_encoder(x)

    def _step(self, batch, batch_idx, log_prefix="", log_sync_dist=False):
        x, action_targets = batch

        # split action targets into joystick, trigger, and button targets
        joystick_targets, trigger_targets, button_targets = self._split_action_targets(action_targets)

        # compute action logits
        joystick_logits, trigger_logits, button_logits = self.forward(x, rollout=False)

        # compute and log joystick loss and metrics
        joystick_loss = compute_joystick_loss(joystick_logits, joystick_targets, self.discretise_joystick)
        self.log(f"{log_prefix}/joystick_loss", joystick_loss, sync_dist=log_sync_dist)
        if not torch.isnan(joystick_logits).all():
            joystick_logits_nonan = joystick_logits[~torch.isnan(joystick_logits)]
            joystick_targets_nonan = joystick_targets[~torch.isnan(joystick_logits)]
            if self.discretise_joystick:
                joystick_metrics = get_discrete_joystick_metrics(
                    joystick_logits_nonan,
                    joystick_targets_nonan,
                    self.num_joystick_actions,
                    self.num_discretisation_bins,
                )
            else:
                joystick_metrics = get_continuous_joystick_metrics(joystick_logits, joystick_targets)
            for k, v in joystick_metrics.items():
                self.log(f"{log_prefix}/{k}", v, sync_dist=log_sync_dist)

        # compute and log trigger loss and metrics
        if self.num_trigger_actions > 0:
            trigger_loss = compute_trigger_loss(trigger_logits, trigger_targets)
            self.log(f"{log_prefix}/trigger_loss", trigger_loss, sync_dist=log_sync_dist)
            if not torch.isnan(trigger_logits).all():
                trigger_logits_nonan = trigger_logits[~torch.isnan(trigger_logits)]
                trigger_targets_nonan = trigger_targets[~torch.isnan(trigger_logits)]
                trigger_metrics = get_trigger_metrics(trigger_logits_nonan, trigger_targets_nonan)
                for k, v in trigger_metrics.items():
                    self.log(f"{log_prefix}/{k}", v, sync_dist=log_sync_dist)

        else:
            trigger_loss = 0.0

        # compute and log button loss and metrics
        button_loss = compute_button_loss(button_logits, button_targets)
        self.log(f"{log_prefix}/button_loss", button_loss, sync_dist=log_sync_dist)
        if not torch.isnan(button_logits).all():
            button_logits_nonan = button_logits[~torch.isnan(button_logits)]
            button_targets_nonan = button_targets[~torch.isnan(button_logits)]
            button_metrics = get_button_metrics(button_logits_nonan, button_targets_nonan)
            for k, v in button_metrics.items():
                self.log(f"{log_prefix}/{k}", v, sync_dist=log_sync_dist)

        loss = self.joystick_loss_weight * joystick_loss + self.button_loss_weight * (trigger_loss + button_loss)
        self.log(f"{log_prefix}/loss", loss, sync_dist=log_sync_dist)
        return loss

    def init_for_rollout(self):
        """
        Initialise the model for rollout.
        """
        self.bc_model.init_for_sequence(batch_size=1)

    def forward(self, x, rollout=False):
        """
        Forward pass through the BC model.
        :param x: input image or embedding (if train_from_embeddings is True)
            (batch_size, sequence_length, channels, height, width) if image is given
            (batch_size, sequence_length, embedding_dim) if embedding is given
        :param rollout: whether forward pass is for rollout
        :param action_selection: whether to compute no actions, sample from action distributions ('stochastic'), or take greedy actions
            ('deterministic')
        :return: joystick logits, trigger logits, and button logits
        """
        img_enc = self._get_embedding(x)

        if rollout:
            action_logits = self.bc_model(img_enc, rollout=True)
        else:
            # initialise model for sequence during training
            assert img_enc.dim() == 3, f"Input to model must be of shape (batch_size, sequence_length, embedding_dim) but is {img_enc.shape}"
            batch_size = img_enc.shape[0]
            self.bc_model.init_for_sequence(batch_size=batch_size)
            action_logits = self.bc_model(img_enc, rollout=False)

        if torch.isnan(img_enc).any():
            warnings.warn("NaNs in image embedding during BCModel forward pass.")
        if torch.isnan(action_logits).any():
            warnings.warn("NaNs in action logits during BCModel forward pass.")

        return self._split_action_logits(action_logits)

    def act(self, x, joystick_mode="stochastic", trigger_mode="stochastic", button_mode="stochastic"):
        """
        Take an action given an image.
        :param x: input image as (channels, height, width)
        :param joystick_mode: whether to compute no actions, sample from action distributions ('stochastic'), or take greedy actions ('deterministic')
        :param trigger_mode: whether to compute no actions, sample from action distributions ('stochastic'), or take greedy actions ('deterministic')
        :param button_mode: whether to compute no actions, sample from action distributions ('stochastic'), or take greedy actions ('deterministic')
        :return: joystick actions, trigger actions, and button actions
        """
        assert self.train_from_embeddings is False, "Can not act from embeddings"
        assert len(x.shape) == 3, f"Input to model must be of shape (channels, height, width) but is {x.shape}"
        # add singleton batchsize and sequence length dimensions
        x = x.reshape(1, 1, *x.shape)
        with torch.no_grad():
            joystick_logits, trigger_logits, button_logits = self.forward(x, rollout=True)
        joystick_logits = joystick_logits.squeeze(0)
        if self.discretise_joystick:
            # Softmax distribution over discrete joystick action bins (have to reshape for bins/ classes to be in last dimension)
            joystick_dist = Categorical(logits=joystick_logits.swapaxes(-1, -2))
        else:
            joystick_dist = None

        if trigger_logits is not None:
            trigger_logits = trigger_logits.squeeze(0)
            trigger_dist = Bernoulli(logits=trigger_logits)  # triggers in range [0, 1]

        button_logits = button_logits.squeeze(0)
        button_dist = Bernoulli(logits=button_logits)  # buttons in range [0, 1]

        # joystick actions
        if self.discretise_joystick:
            if joystick_mode == "stochastic":
                joystick_bins = joystick_dist.sample()
            elif joystick_mode == "deterministic":
                joystick_bins = joystick_dist.probs.argmax(dim=-1)
            else:
                raise ValueError(f"Unknown joystick action mode {joystick_mode}")

            # convert discrete joystick actions to joystick positions
            self.discretise_joystick_bins = self.discretise_joystick_bins.to(joystick_bins.device)
            joystick_actions = self.discretise_joystick_bins[joystick_bins].int()
        else:
            joystick_actions = torch.tanh(joystick_logits)  # joystick positions in range [-1, 1]

        # trigger actions
        if trigger_logits is not None:
            if trigger_mode == "stochastic":
                trigger_actions = trigger_dist.sample()
            elif trigger_mode == "deterministic":
                trigger_actions = trigger_dist.probs.round()
            else:
                raise ValueError(f"Unknown trigger action mode {trigger_mode}")
            trigger_actions = trigger_actions.int()
        else:
            trigger_actions = None

        if button_mode == "stochastic":
            button_actions = button_dist.sample()
        elif button_mode == "deterministic":
            button_actions = button_dist.probs.round()
        else:
            raise ValueError(f"Unknown button action mode {button_mode}")
        button_actions = button_actions.int()

        return joystick_actions, trigger_actions, button_actions

    def training_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, "train", False)

    def validation_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, "val", True)

    def test_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, "test", True)

    def configure_optimizers(self):
        optimiser = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        if self.lr_warmup_steps == 0:
            # no warmup
            return [optimiser], []
        else:
            # use linear LR warmup
            lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimiser, lambda steps: min((steps + 1) / self.lr_warmup_steps, 1))
            return [optimiser], [{"scheduler": lr_scheduler, "interval": "step"}]

    def on_before_optimizer_step(self, optimizer):
        # Compute the 2-norm for each layer
        norms = grad_norm(self.image_encoder, norm_type=2)
        norms.update(grad_norm(self.bc_model, norm_type=2))
        self.log_dict(norms)
