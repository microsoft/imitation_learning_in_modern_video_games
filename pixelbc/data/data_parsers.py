import json
from abc import ABC, abstractmethod
from collections import deque

import decord
import h5py
import numpy as np
import torch

from pixelbc.data.actions import get_joystick_bins
from pixelbc.data.csgo_actions import preprocess_csgo_action_line
from pixelbc.data.image_preprocessing import \
    get_preprocessing_function_and_image_shape
from pixelbc.data.minerl_actions import preprocess_minerl_action_line
from pixelbc.data.utils import (get_embedding_file,
                                get_minerl_video_file_from_action_file,
                                get_pretrained_encoder_dirname)

EMBEDDING_KEY = "embedding"


class DataParser(ABC):
    IMAGE_TYPE = torch.uint8
    ACTION_TYPE = torch.float32
    EMBEDDING_TYPE = torch.float32

    def __init__(
        self,
        preprocess_fn,
        image_shape,
        sequence_length,
        framestacking,
        discretise_joystick,
        downsample,
        decord_num_workers=1,
    ):
        self.preprocess_fn = preprocess_fn
        self.image_shape = image_shape
        self.sequence_length = sequence_length
        self.framestacking = framestacking
        self.discretise_joystick = discretise_joystick
        self.downsample = downsample
        self.decord_num_workers = decord_num_workers

    def _load_video(self, video_filename, indices=None):
        """
        Load the video file using decord

        :param video_filename: path to the video file
        :param indices: indices of the frames to load (or all frames if None)
        :return: numpy array of frames (num_frames, height, width, channels) as RGB
        """
        with open(video_filename, "rb") as f:
            vr = decord.VideoReader(f, ctx=decord.cpu(), num_threads=self.decord_num_workers)
        if indices is None:
            return vr[:].asnumpy()
        else:
            return vr.get_batch(indices).asnumpy()

    def _load_jsonl(self, jsonl_filename, indices=None):
        """
        Load the jsonl file

        :param jsonl_filename: path to the jsonl file
        :param indices: indices of the lines to load (if None, load all lines)
        :return: list of dictionaries with one entry per loaded line
        """
        with open(jsonl_filename, "r") as f:
            lines = f.readlines()
        if indices is None:
            return [json.loads(line) for line in lines]
        else:
            return [json.loads(lines[i]) for i in indices]

    def _load_embedding(self, embedding_filename, indices=None):
        """
        Load the embedding file

        :param embedding_filename: path to the embedding file
        :param indices: indices of the embeddings to load (if None, load all embeddings)
        :return: numpy array of embeddings (num_frames, embedding_dim)
        """
        data = np.load(embedding_filename, allow_pickle=True)[EMBEDDING_KEY]
        if indices is None:
            return data
        return data[indices]

    def _get_sequence_indices(self, start_index, num_steps):
        """Get a list of indices for a random sequence of length sequence_length

        :param start_index: index of the first step to include
        :param num_steps: number of steps in the data (before downsampling)
        :return: list of indices (for too short sequences, the last index is repeated)
        """
        indices = [start_index + i * self.downsample for i in range(self.sequence_length)]
        return [min(int(idx), num_steps - 1) for idx in indices]

    def _create_stacks(self, inputs):
        """Create frame stacks from the inputs

        :param inputs: torch tensor (num_inputs, ...)
        :return: torch tensor of input stacks (num_inputs, ...)
        """
        img_stack = deque(maxlen=self.framestacking)
        for _ in range(self.framestacking - 1):
            img_stack.append(torch.zeros(self.image_shape, dtype=torch.uint8))
        stacks = []
        for img in inputs:
            img_stack.append(img)
            stacks.append(torch.concat(list(img_stack), dim=0))
        return torch.tensor(stacks)

    @abstractmethod
    def get_all_data(self, action_filename):
        """Return inputs and outputs for given action file as torch tensors

        :param action_filename: path to the action file
        :return: tuple of torch tensors of processed inputs and actions
        """
        raise NotImplementedError

    @abstractmethod
    def get_start_indices_from_data(self, actions):
        """Get a list of start indices for the sequences in the action file

        :param actions: numpy array of actions
        :return: list of start indices
        """
        raise NotImplementedError

    @abstractmethod
    def get_sequence_from_data(self, inputs, actions, start_index):
        """Get the sequence of indices to access from the data and downsample

        :param inputs: torch tensor of inputs
        :param actions: torch tensor of actions
        :param start_index: index of the first step to include
        :return: tuple of torch tensors of inputs and actions
        """
        raise NotImplementedError


class MineRLImageParser(DataParser):
    # How much extra steps to include after first log has been obtained
    STEPS_TO_INCLUDE_AFTER_FIRST_LOG = 2 * 20

    def __init__(
        self,
        preprocess_fn,
        image_shape,
        sequence_length,
        framestacking,
        discretise_joystick,
        downsample,
        decord_num_workers=1,
    ):
        super().__init__(
            preprocess_fn,
            image_shape,
            sequence_length,
            framestacking,
            discretise_joystick,
            downsample,
            decord_num_workers,
        )

        # Null bin is one with associated zero continuous value
        self._null_bin = np.where(get_joystick_bins() == 0)[0]
        assert len(self._null_bin) == 1, "Expected bins to have one null bin (continuous value is 0), but found none"
        self._null_bin = self._null_bin[0].item()

    def _load_actions(self, action_filename, indices=None):
        return self._load_jsonl(action_filename, indices)

    def _preprocess_action(self, action):
        return preprocess_minerl_action_line(action, self.discretise_joystick)

    def _check_if_has_log(self, step_data):
        inventory = step_data["inventory"]
        for item in inventory:
            if "log" in item["type"]:
                return True
        return False

    def _is_noop(self, action):
        if self.discretise_joystick:
            # For discrete actions, the values are all either zero or null bin
            return torch.all(torch.isin(action, torch.tensor([0, self._null_bin])))
        else:
            return torch.all(action == 0)

    def _get_num_steps(self, action_filename):
        with open(action_filename, "r") as f:
            lines = f.readlines()

        # Find the first log step
        first_log_step = None
        for i, line in enumerate(lines):
            data = json.loads(line)
            if self._check_if_has_log(data):
                first_log_step = i
                break
        assert first_log_step is not None, f"No log found in {action_filename}"

        # Go until the end of the episode or until STEPS_TO_INCLUDE_AFTER_FIRST_LOG after the first log
        return min(first_log_step + self.STEPS_TO_INCLUDE_AFTER_FIRST_LOG, len(lines))

    def _get_sequence_indices(self, start_index, num_steps, actions):
        """Get a list of indices for a random sequence of length sequence_length

        :param start_index: index of the first step to include
        :param num_steps: number of steps in the data (before downsampling)
        :param actions: list of processed actions for all steps
        :return: list of indices (for too short sequences, the last index is repeated)
        """
        valid_steps = [start_index + i for i, a in enumerate(actions[start_index:]) if not self._is_noop(a)]
        assert len(valid_steps) > 0, f"No valid steps found in {start_index} to {num_steps}"
        indices = []
        for i in range(self.sequence_length):
            idx = i * self.downsample
            value = valid_steps[idx] if idx < len(valid_steps) else valid_steps[-1]
            indices.append(value)
        return indices

    def get_all_data(self, action_filename):
        """Return inputs and outputs for given action file as torch tensors

        :param action_filename: path to the action file
        :return: tuple of torch tensors of processed inputs and actions
        """
        actions = self._load_actions(action_filename)
        actions = np.array([self._preprocess_action(action) for action in actions])
        actions_tensor = torch.from_numpy(actions).to(dtype=DataParser.ACTION_TYPE)

        video_filename = get_minerl_video_file_from_action_file(action_filename)
        frames = self._load_video(video_filename)
        inputs = np.array([self.preprocess_fn(frame) for frame in frames])
        inputs_tensor = torch.from_numpy(inputs).to(dtype=DataParser.IMAGE_TYPE)

        return inputs_tensor, actions_tensor

    def get_start_indices_from_data(self, actions):
        """Get a list of start indices for the sequences in the action file

        :param actions: numpy array of actions
        :return: list of start indices
        """
        valid_steps = [i for i, a in enumerate(actions) if not self._is_noop(a)]
        start_indices = [i for i in valid_steps[: -(self.sequence_length * self.downsample)]]
        if not start_indices:
            # no start indices with enough steps after --> take earliest possible steps (one for each downsample to train on all possible samples)
            start_indices = [valid_steps[i] for i in range(self.downsample)]
        return start_indices

    def get_sequence_from_data(self, inputs, actions, start_index):
        """Get the sequence of indices to access from the data and downsample

        :param inputs: torch tensor of inputs
        :param actions: torch tensor of actions
        :param start_index: index of the first step to include
        :return: tuple of torch tensors of inputs and actions
        """
        indices = self._get_sequence_indices(start_index, len(actions), actions)
        inputs = inputs[indices]
        if self.framestacking > 1:
            inputs = self._create_stacks(inputs)
        actions = actions[indices]
        return inputs, actions


class MineRLEmbeddingParser(MineRLImageParser):
    def __init__(
        self,
        encoder_dirname,
        preprocess_fn,
        image_shape,
        sequence_length,
        framestacking,
        discretise_joystick,
        downsample,
        decord_num_workers=1,
    ):
        super().__init__(
            preprocess_fn,
            image_shape,
            sequence_length,
            framestacking,
            discretise_joystick,
            downsample,
            decord_num_workers,
        )
        self.encoder_dirname = encoder_dirname

    def get_all_data(self, action_filename):
        """Return inputs and outputs for given action file as torch tensors

        :param action_filename: path to the action file
        :return: tuple of torch tensors of processed inputs and actions
        """
        actions = self._load_actions(action_filename)
        actions = np.array([self._preprocess_action(action) for action in actions])
        actions_tensor = torch.from_numpy(actions).to(dtype=DataParser.ACTION_TYPE)

        embedding_filename = get_embedding_file(self.encoder_dirname, action_filename)
        embeddings = self._load_embedding(embedding_filename)
        embeddings_tensor = torch.from_numpy(embeddings).to(dtype=DataParser.EMBEDDING_TYPE)

        return embeddings_tensor, actions_tensor

    def get_sequence_from_data(self, inputs, actions, start_index):
        """Get the sequence of indices to access from the data and downsample

        :param inputs: torch tensor of inputs
        :param actions: torch tensor of actions
        :param start_index: index of the first step to include
        :return: tuple of torch tensors of inputs and actions
        """
        indices = self._get_sequence_indices(start_index, len(actions), actions)
        return inputs[indices], actions[indices]


class CSGOImageParser(DataParser):
    NUM_STEPS_PER_FILE = 1000  # fixed in dataset; indices from 0 to 999
    KEY_INITIAL = "frame_"
    IMAGE_KEY_SUFFIX = "_x"
    ACTION_KEY_SUFFIX = "_y"

    def __init__(
        self,
        preprocess_fn,
        image_shape,
        sequence_length,
        framestacking,
        discretise_joystick,
        downsample,
        decord_num_workers=1,
    ):
        super().__init__(
            preprocess_fn,
            image_shape,
            sequence_length,
            framestacking,
            discretise_joystick,
            downsample,
            decord_num_workers,
        )

    def _preprocess_action(self, action):
        return preprocess_csgo_action_line(action, self.discretise_joystick)

    def _get_img_key(self, idx):
        return self.KEY_INITIAL + str(idx) + self.IMAGE_KEY_SUFFIX

    def _get_action_key(self, idx):
        return self.KEY_INITIAL + str(idx) + self.ACTION_KEY_SUFFIX

    def get_start_indices(self, action_filename):
        """Get a list of start indices for the sequences in the action file

        :param action_filename: path to the action file
        :return: list of start indices
        """
        return [i for i in range(CSGOImageParser.NUM_STEPS_PER_FILE - self.sequence_length)]

    def get_all_data(self, action_filename):
        """Return inputs and outputs for given action file as torch tensors

        :param action_filename: path to the action file
        :return: tuple of torch tensors of processed inputs and actions
        """
        all_indices = range(CSGOImageParser.NUM_STEPS_PER_FILE)
        frames = []
        actions = []

        with h5py.File(action_filename, "r") as f:
            img_keys = [self._get_img_key(idx) for idx in all_indices]
            action_keys = [self._get_action_key(idx) for idx in all_indices]
            for img_key, action_key in zip(img_keys, action_keys):
                frame = f[img_key][:].astype(np.uint8)[:, :, ::-1]
                frame = self.preprocess_fn(frame)
                action = self._preprocess_action(f[action_key][:])
                frames.append(frame)
                actions.append(action)

        inputs = np.array(frames)
        actions = np.array(actions)

        return torch.from_numpy(inputs).to(dtype=DataParser.IMAGE_TYPE), torch.from_numpy(actions).to(dtype=DataParser.ACTION_TYPE)

    def get_start_indices_from_data(self, actions):
        """Get a list of start indices for the sequences in the action file

        :param actions: numpy array of actions
        :return: list of start indices
        """
        num_steps = len(actions)
        start_indices = [i for i in range(num_steps - self.sequence_length * self.downsample)]
        if not start_indices:
            # no start indices with enough steps after --> take earliest possible steps (one for each downsample to train on all possible samples)
            start_indices = [i for i in range(self.downsample)]
        return start_indices

    def get_sequence_from_data(self, inputs, actions, start_index):
        """Get the sequence of indices to access from the data and downsample

        :param inputs: torch tensor of inputs
        :param actions: torch tensor of actions
        :param start_index: index of the first step to include
        :return: tuple of torch tensors of inputs and actions
        """
        indices = self._get_sequence_indices(start_index, len(actions))
        inputs = inputs[indices]
        if self.framestacking > 1:
            inputs = self._create_stacks(inputs)
        actions = actions[indices]
        return inputs, actions


class CSGOEmbeddingParser(CSGOImageParser):
    def __init__(
        self,
        encoder_dirname,
        preprocess_fn,
        image_shape,
        sequence_length,
        framestacking,
        discretise_joystick,
        downsample,
        decord_num_workers=1,
    ):
        super().__init__(
            preprocess_fn,
            image_shape,
            sequence_length,
            framestacking,
            discretise_joystick,
            downsample,
            decord_num_workers,
        )
        self.encoder_dirname = encoder_dirname

    def get_all_data(self, action_filename):
        """Return inputs and outputs for given action file as torch tensors

        :param action_filename: path to the action file
        :return: tuple of torch tensors of processed inputs and actions
        """
        all_indices = range(CSGOImageParser.NUM_STEPS_PER_FILE)
        actions = []

        with h5py.File(action_filename, "r") as f:
            action_keys = [self._get_action_key(idx) for idx in all_indices]
            for action_key in action_keys:
                actions.append(self._preprocess_action(f[action_key][:]))
        actions = np.array(actions)

        embedding_filename = get_embedding_file(self.encoder_dirname, action_filename)
        embeddings = self._load_embedding(embedding_filename)

        return torch.from_numpy(embeddings).to(dtype=DataParser.EMBEDDING_TYPE), torch.from_numpy(actions).to(dtype=DataParser.ACTION_TYPE)

    def get_sequence_from_data(self, inputs, actions, start_index):
        """Get the sequence of indices to access from the data and downsample

        :param inputs: torch tensor of inputs
        :param actions: torch tensor of actions
        :param start_index: index of the first step to include
        :return: tuple of torch tensors of inputs and actions
        """
        indices = self._get_sequence_indices(start_index, len(actions))
        return inputs[indices], actions[indices]


def get_data_parser(
    model_config,
    game,
    image_height,
    image_width,
    sequence_length,
    framestacking,
    discretise_joystick,
    downsample,
    decord_num_workers=1,
    **kwargs,
):
    preprocess_fn, image_shape = get_preprocessing_function_and_image_shape(
        model_config.encoder is None, model_config.pretrained_encoder, image_height, image_width
    )
    train_from_embeddings = model_config.train_from_embeddings

    if train_from_embeddings:
        assert model_config.pretrained_encoder is not None, "Pretrained encoder must be specified for training from embeddings"
        encoder_dirname = get_pretrained_encoder_dirname(model_config.pretrained_encoder.family, model_config.pretrained_encoder.name)
    else:
        encoder_dirname = None

    if game == "minerl":
        if train_from_embeddings:
            return MineRLEmbeddingParser(
                encoder_dirname,
                preprocess_fn,
                image_shape,
                sequence_length,
                framestacking,
                discretise_joystick,
                downsample,
                decord_num_workers,
            )
        return MineRLImageParser(
            preprocess_fn,
            image_shape,
            sequence_length,
            framestacking,
            discretise_joystick,
            downsample,
            decord_num_workers,
        )
    elif game == "csgo":
        if train_from_embeddings:
            return CSGOEmbeddingParser(
                encoder_dirname,
                preprocess_fn,
                image_shape,
                sequence_length,
                framestacking,
                discretise_joystick,
                downsample,
                decord_num_workers,
            )
        return CSGOImageParser(
            preprocess_fn,
            image_shape,
            sequence_length,
            framestacking,
            discretise_joystick,
            downsample,
            decord_num_workers,
        )
    else:
        raise ValueError(f"Unsupported game {game}")
