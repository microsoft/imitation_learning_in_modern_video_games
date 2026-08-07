import lightning.pytorch as pl
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from pixelbc.data.data_parsers import get_data_parser
from pixelbc.data.data_split import get_file_paths_from_split_file


class BCDataSet(Dataset):
    """Naive dataset for BC data

    On creation, load all data into memory, and determine valid start indices to sample sequences from.
    During training, sample a random start index, and return a sequence of length `seq_len` from that index.
    """

    def __init__(self, data_parser, data_files):
        self.data_parser = data_parser
        self.data_by_file = {}
        print(f"Precomputing data for {len(data_files)} files ...")
        for data_file in tqdm(data_files):
            self.data_by_file[data_file] = self.data_parser.get_all_data(data_file)

        self.data_file_start_indices = []
        for data_file, (_, actions_data) in self.data_by_file.items():
            start_indices = self.data_parser.get_start_indices_from_data(actions_data)
            for start_index in start_indices:
                self.data_file_start_indices.append((data_file, start_index))

    def __len__(self):
        return len(self.data_file_start_indices)

    def __getitem__(self, idx):
        data_file, start_index = self.data_file_start_indices[idx]
        inputs, actions = self.data_by_file[data_file]
        inputs_seq, actions_seq = self.data_parser.get_sequence_from_data(inputs, actions, start_index)
        return inputs_seq, actions_seq


class BCDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_config,
        model_config,
        **kwargs,
    ):
        super().__init__()
        self.data_config = data_config
        self.model_config = model_config
        self.base_data_path = data_config.data_path
        self.save_hyperparameters()

    def setup(self, stage=None):
        self.data_parser = get_data_parser(self.model_config, **self.data_config)

        if self.data_config.train_split_file_path is not None:
            train_files = get_file_paths_from_split_file(self.data_config.train_split_file_path, self.base_data_path)
            if len(train_files) == 0:
                self.train_set = None
            self.train_set = BCDataSet(self.data_parser, train_files)
        else:
            self.train_set = None

        print(f"total training set length: {len(self.train_set)}")

        if self.data_config.validation_split_file_path is not None:
            val_files = get_file_paths_from_split_file(self.data_config.validation_split_file_path, self.base_data_path)
            if len(val_files) == 0:
                self.val_set = None
            self.val_set = BCDataSet(self.data_parser, val_files)
        else:
            self.val_set = None

        if self.data_config.test_split_file_path is not None:
            test_files = get_file_paths_from_split_file(self.data_config.test_split_file_path, self.base_data_path)
            if len(test_files) == 0:
                self.test_set = None
            self.test_set = BCDataSet(self.data_parser, test_files)
        else:
            self.test_set = None

    def _loader_kwargs(self, num_workers, pin_memory):
        # persistent_workers/prefetch_factor are only valid with num_workers > 0.
        # On Windows the dataset is already fully in RAM, so num_workers=0 (no
        # spawn, no per-worker dataset copy) is both valid and fastest.
        kwargs = dict(
            batch_size=self.data_config.batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        if num_workers > 0:
            kwargs["persistent_workers"] = True
            kwargs["prefetch_factor"] = self.data_config.prefetch_factor
        return kwargs

    def train_dataloader(self):
        return DataLoader(
            self.train_set,
            shuffle=True,
            **self._loader_kwargs(self.data_config.train_num_workers, pin_memory=True),
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_set,
            **self._loader_kwargs(self.data_config.other_num_workers, pin_memory=False),
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_set,
            **self._loader_kwargs(self.data_config.other_num_workers, pin_memory=False),
        )

    def teardown(self, stage: str = None):
        pass
