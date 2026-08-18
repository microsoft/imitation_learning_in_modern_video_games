import os
import warnings

import cv2
import lightning.pytorch as pl
import numpy as np
import torch
from lightning.pytorch.utilities import grad_norm
from torch.distributions import Bernoulli, Categorical

from pixelbc.models.encoders import get_encoder
from pixelbc.models.rgb_decoder import RGBDecoder
from pixelbc.models.utils import get_model
from pixelbc.models.utils.loss_utils import (compute_button_loss,
                                             compute_depth_loss,
                                             compute_joystick_loss,
                                             compute_trigger_loss)
from pixelbc.models.utils.train_metrics import (
    get_button_metrics, get_continuous_joystick_metrics,
    get_discrete_joystick_metrics, get_trigger_metrics)

# predict_depth target resolution -- must match PREDICT_DEPTH_SIZE in data_parsers.py
PREDICT_DEPTH_SIZE = 224


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
        predict_depth=False,
        depth_loss_weight=0.0,
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
        self.predict_depth = predict_depth
        self.depth_loss_weight = depth_loss_weight

        # Epoch-aggregated trigger TP/positive counts, for a recall that isn't
        # poisoned by per-batch 0/0 (trigger/"shoot" positives are ~4% of frames,
        # so many individual batches have zero shoot-target frames -> that batch's
        # tp_rate is nan -> Lightning's per-batch-averaged self.log() turns the
        # whole epoch nan even though other batches had real positives). Summing
        # counts across the full epoch before dividing avoids that.
        self._trigger_epoch_stats = {}

        if self.predict_depth:
            # Reconstructs the SAME-timestep depth map from the LSTM's hidden state
            # at t -- see rgb_decoder.py (reused generically via output_channels=1) and
            # data_parsers.py's CSGOEmbeddingParser(predict_depth=True) for where the
            # raw depth-pixel target comes from.
            assert self.train_from_embeddings, "predict_depth requires train_from_embeddings=True."
            assert bc_policy.type == "lstm", "predict_depth currently requires bc_policy.type == 'lstm' (needs a hidden state to branch off)."

        if self.bc_policy_config.type == "gpt":
            assert self.lr_warmup_steps > 0, "GPT policy requires LR warmup to be enabled."

        if self.train_from_embeddings:
            assert self.pretrained_encoder_config is not None, "Training from embeddings is only supported for pretrained encoders."

        # define encoder
        self.image_encoder = get_encoder(image_shape, encoder, pretrained_encoder)
        self.image_encoding_dim = self.image_encoder.get_embedding_dim()
        # RGB+D (embedding-concat): the loader concatenates a depth embedding onto
        # the RGB embedding (data.depth_embedding=True), doubling the policy input.
        # depth_embedding is preserved in hparams, so 1536 checkpoints reload correctly.
        if kwargs.get("depth_embedding"):
            self.image_encoding_dim *= 2

        # define BC models
        if self.discretise_joystick:
            self.num_joystick_logits = self.num_discretisation_bins * self.num_joystick_actions
        else:
            self.num_joystick_logits = self.num_joystick_actions
        self.num_trigger_logits = self.num_trigger_actions
        self.num_button_logits = self.num_button_actions
        self.num_logits = self.num_joystick_logits + self.num_trigger_logits + self.num_button_logits
        self.bc_model = get_model(self.image_encoding_dim, self.sequence_length, self.num_logits, self.bc_policy_config)

        # auxiliary depth-in-image-space reconstruction head. RGBDecoder is generic
        # (output_channels) -- output_channels=1 gives a depth decoder for free, no
        # new decoder class. Input is the LSTM's hidden state, not the frozen
        # embedding, so this loss's gradient is connected to the policy's weights.
        if self.predict_depth:
            self.depth_decoder = RGBDecoder(self.bc_model.hidden_size, output_size=PREDICT_DEPTH_SIZE, output_channels=1)

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
            (batch_size, sequence_length, embedding_dim [+ trailing raw-depth payload
            if predict_depth]) if embedding is given
        :return: image embedding, with any trailing raw-depth payload (predict_depth)
            sliced off -- the policy never sees it, only _get_depth_target does
        """
        if not self.train_from_embeddings:
            return self.image_encoder(x)
        if self.predict_depth:
            return x[..., : self.image_encoding_dim]
        return x

    def _get_depth_target(self, x):
        """
        Ground-truth depth map for the auxiliary reconstruction loss, in [0, 1].
        :param x: same batch input passed to _get_embedding -- the raw depth payload
            is concatenated onto the tail of the embedding's feature axis by
            CSGOEmbeddingParser(predict_depth=True).
        """
        raw_depth = x[..., self.image_encoding_dim :]
        return raw_depth.reshape(-1, 1, PREDICT_DEPTH_SIZE, PREDICT_DEPTH_SIZE).float().div(255)

    def _save_depth_sample(self, depth_pred, depth_target):
        """Save one real-vs-reconstructed depth frame pair (side by side, PNG) per
        validation epoch, for visual QA of the predict_depth auxiliary task.
        """
        if self.logger is None or not hasattr(self.logger, "log_dir") or self.logger.log_dir is None:
            return
        out_dir = os.path.join(self.logger.log_dir, "depth_samples")
        os.makedirs(out_dir, exist_ok=True)
        target_img = (depth_target[0, 0].detach().cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        pred_img = (depth_pred[0, 0].detach().cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        divider = np.full((target_img.shape[0], 4), 255, dtype=np.uint8)
        side_by_side = np.concatenate([target_img, divider, pred_img], axis=1)
        cv2.imwrite(os.path.join(out_dir, f"epoch_{self.current_epoch:03d}.png"), side_by_side)

    def _step(self, batch, batch_idx, log_prefix="", log_sync_dist=False):
        x, action_targets = batch

        # split action targets into joystick, trigger, and button targets
        joystick_targets, trigger_targets, button_targets = self._split_action_targets(action_targets)

        # compute action logits (and the policy's hidden state too, in one pass, if
        # predict_depth needs it -- must NOT call self.bc_model a second time, since
        # the LSTM is stateful and a duplicate call would double-advance it)
        if self.predict_depth:
            (joystick_logits, trigger_logits, button_logits), hidden_repr = self.forward(
                x, rollout=False, return_hidden=True
            )
        else:
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

                # epoch-aggregated trigger recall (see __init__ comment on why
                # this needs whole-epoch counts instead of per-batch averaging)
                stats = self._trigger_epoch_stats.setdefault(log_prefix, {"tp": 0.0, "pos": 0.0})
                trigger_pred = torch.sigmoid(trigger_logits_nonan) > 0.5
                trigger_target_bool = trigger_targets_nonan.bool()
                stats["tp"] += (trigger_pred & trigger_target_bool).sum().item()
                stats["pos"] += trigger_target_bool.sum().item()

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

        # auxiliary depth-in-image-space reconstruction loss: reconstruct the SAME
        # timestep's depth map from the policy's hidden state at that timestep.
        if self.predict_depth:
            hidden_flat = hidden_repr.reshape(-1, self.bc_model.hidden_size)
            depth_pred = self.depth_decoder(hidden_flat)
            depth_target = self._get_depth_target(x)
            depth_loss = compute_depth_loss(depth_pred, depth_target)
            self.log(f"{log_prefix}/depth_loss", depth_loss, sync_dist=log_sync_dist)
            loss = loss + self.depth_loss_weight * depth_loss

            # one sample pair per validation epoch, for visual QA (real depth vs.
            # reconstructed depth, side by side)
            if log_prefix == "val" and batch_idx == 0:
                self._save_depth_sample(depth_pred, depth_target)

        self.log(f"{log_prefix}/loss", loss, sync_dist=log_sync_dist)
        return loss

    def init_for_rollout(self):
        """
        Initialise the model for rollout.
        """
        self.bc_model.init_for_sequence(batch_size=1)

    def forward(self, x, rollout=False, return_hidden=False):
        """
        Forward pass through the BC model.
        :param x: input image or embedding (if train_from_embeddings is True)
            (batch_size, sequence_length, channels, height, width) if image is given
            (batch_size, sequence_length, embedding_dim) if embedding is given
        :param rollout: whether forward pass is for rollout
        :param return_hidden: if True (training only), also return the policy's
            hidden state alongside the action logits -- used by _step to feed
            predict_depth without a second, redundant (and, since the LSTM is
            stateful, incorrect) call into self.bc_model.
        :param action_selection: whether to compute no actions, sample from action distributions ('stochastic'), or take greedy actions
            ('deterministic')
        :return: joystick logits, trigger logits, and button logits (plus hidden state if return_hidden)
        """
        img_enc = self._get_embedding(x)
        hidden = None

        if rollout:
            action_logits = self.bc_model(img_enc, rollout=True)
        else:
            # initialise model for sequence during training
            assert img_enc.dim() == 3, f"Input to model must be of shape (batch_size, sequence_length, embedding_dim) but is {img_enc.shape}"
            batch_size = img_enc.shape[0]
            self.bc_model.init_for_sequence(batch_size=batch_size)
            if return_hidden:
                action_logits, hidden = self.bc_model(img_enc, rollout=False, return_hidden=True)
            else:
                action_logits = self.bc_model(img_enc, rollout=False)

        if torch.isnan(img_enc).any():
            warnings.warn("NaNs in image embedding during BCModel forward pass.")
        if torch.isnan(action_logits).any():
            warnings.warn("NaNs in action logits during BCModel forward pass.")

        split_logits = self._split_action_logits(action_logits)
        return (split_logits, hidden) if return_hidden else split_logits

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

    def _reset_trigger_epoch_stats(self, log_prefix):
        self._trigger_epoch_stats[log_prefix] = {"tp": 0.0, "pos": 0.0}

    def _log_trigger_epoch_recall(self, log_prefix):
        stats = self._trigger_epoch_stats.get(log_prefix)
        if stats is None or stats["pos"] == 0:
            return  # no trigger positives seen this epoch -- genuinely undefined, skip rather than log nan
        recall = stats["tp"] / stats["pos"]
        self.log(f"{log_prefix}/trigger_recall_epoch", recall, sync_dist=(log_prefix != "train"))

    def on_train_epoch_start(self):
        self._reset_trigger_epoch_stats("train")

    def on_train_epoch_end(self):
        self._log_trigger_epoch_recall("train")

    def on_validation_epoch_start(self):
        self._reset_trigger_epoch_stats("val")

    def on_validation_epoch_end(self):
        self._log_trigger_epoch_recall("val")

    def on_test_epoch_start(self):
        self._reset_trigger_epoch_stats("test")

    def on_test_epoch_end(self):
        self._log_trigger_epoch_recall("test")

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
