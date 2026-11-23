"""
@file SequenceLoader.py
@brief Pytorch Dataset / DataModule for video sequences represented by pre-computed frame embeddings.
       Each video is annotated with the frame-wise developmental stages, the transferability category,
       the morphokinetic category and the Lag Phase onset and duration.
"""

import os
from argparse import ArgumentParser

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from DataLoading.sequencedatamodule.utils import getlabel_ciray, getlabel_cda, get_LAG_phase

ENCODER_FILE_FORMAT = ".npy"
EMBEDDINGS_PATH = os.environ.get("MKLAG_EMBEDDINGS_PATH", "rn50")

# Embryo category
CATEGORY_MAPPINGS = {
    5: {"EHB": 4, "HB": 3, "SSB": 2, "ADB": 1, "NT": 0},   # A = 4 - B = 3 - C = 2 - D = 1 - NT = 0
    2: {"EHB": 1, "HB": 1, "SSB": 1, "ADB": 1, "NT": 0},   # transferability (T = 1, NT = 0)
}
CATEGORY_TO_5CLASS = CATEGORY_MAPPINGS[5]
CATEGORY_TO_TRANSFERABLE = CATEGORY_MAPPINGS[2]


class SequenceDataset(Dataset):
    """
    @brief Dataset of video represented by frame-wise pre-computed embeddings).

    @details Loads the frame embeddings, and builds the labels from the annotation CSV:
             frame-wise developmental stages (EC1 to EC4), transferability category,
             morphokinetic category and the Lag phase.
    """

    def __init__(self, data_path: str, mode: str, max_frame: int,
                 cdt_annotation: bool = True, blocked: bool = True, version: str = "cda", n_tclasses: int = 8,
                 n_actions: int = None, embeddings_path: str = None):
        """
        @param data_path (str): Path to the CSV file containing the metadata
        @param mode (str): 'train', 'val' or 'test'
        @param max_frame (int): Maximum number of frames kept per video
        @param cdt_annotation (bool): Whether to add the CDT class of the CDA annotation framework
        @param blocked (bool): Whether to add the BLOCKED class of the CDA annotation framework
        @param version (str): Developmental anomalies tolerated conditionning the annotation framework (ciray, cda)
        @param n_tclasses (int): Number of transferability classes (8, 5 or 2), see CATEGORY_MAPPINGS
        @param n_actions (int): Number of developmental stage classes (only used to check the labels)
        """
        print(f"CDT class added ? {cdt_annotation}")
        print(f"BLOCKED class added ? {blocked}")
        print(f"Max frame: {max_frame}")
        print(f"Annotation version : {version}")

        self.data = pd.read_csv(data_path).set_index('video').drop_duplicates()
        self.data.columns = self.data.columns.str.replace(r'^ECC(\d)', r'EC\1', regex=True)
        self.video_paths = self.data.index
        self.mode = mode
        self.version = version
        self.max_frame = max_frame
        self.embeddings_path = embeddings_path or EMBEDDINGS_PATH  # hard-coded default if no option

        if n_tclasses not in CATEGORY_MAPPINGS:
            raise ValueError(f"n_tclasses={n_tclasses} not supported (choices: {list(CATEGORY_MAPPINGS)})")
        self.category_to_class = CATEGORY_MAPPINGS[n_tclasses]
        self.n_tclasses = n_tclasses
        self.n_actions = n_actions

        if not cdt_annotation and not blocked and self.version == "ciray":
            print("Cell stage annotation - Ciray et al. 2014 (Bertha-D)")
            self._getlabel = getlabel_ciray
        elif cdt_annotation and not blocked and self.version == "ciray":
            print("CDT annotation (Bertha-D)")
            self._getlabel = getlabel_ciray
        elif cdt_annotation and blocked and self.version == "cda":
            print("CDA annotation (Gertrude-DT or Brunhilde-DTM)")
            self._getlabel = getlabel_cda
        else:
            raise ValueError("Wrong dataset / annotation framework combination")

    def __len__(self) -> int:
        """@brief Effective size of the dataset"""
        return len(self.video_paths)

    @staticmethod
    def _load_embedding(encoder_path: str) -> torch.Tensor:
        """
        @brief Load pre-computed embeddings.

        @param encoder_path (str): Path to the saved encoder file (.pt or .npy)
        @return torch.Tensor: Tensor of shape (T, Features)
        """
        if not os.path.exists(encoder_path):
            raise FileNotFoundError(f"Encoder file not found: {encoder_path}")

        if encoder_path.endswith(".pt"):
            encoder_data = torch.load(encoder_path, map_location='cpu')
            if not isinstance(encoder_data, torch.Tensor):
                encoder_data = torch.tensor(encoder_data)
        elif encoder_path.endswith(".npy"):
            encoder_data = torch.from_numpy(np.load(encoder_path))
        else:
            raise ValueError(f"Unsupported encoder file format: {encoder_path}")

        return encoder_data.float()

    def __getitem__(self, idx: int) -> dict:
        """
        @return dict: {
            'embedding': (T, F), 'ydev': (T,), 'ycat': (), 'ytrans': (),
            'has_segk', 'lag_sk', 'lag_ek' (k=1..3), 'n_segment', 'video_name'
        }
        """
        video_path = self.video_paths[idx]

        embedding_name = os.path.basename(video_path)[:-4] + ENCODER_FILE_FORMAT
        embedding = self._load_embedding(os.path.join(self.embeddings_path, embedding_name))[:self.max_frame]
        T = embedding.shape[0]

        category = self.data.loc[video_path, 'Category']
        ytrans = torch.tensor(CATEGORY_TO_TRANSFERABLE[category], dtype=torch.long)
        ycat = torch.tensor(self.category_to_class[category], dtype=torch.long)

        lag = get_LAG_phase(self.data, video_path)

        ydev = self._getlabel(data=self.data, version=self.version, video_path=video_path,
                              start_frame=0, end_frame=T, max_frame=T)[:self.max_frame]
        

        # Labels sanity checks
        if len(ydev) != T:
            raise ValueError(f"{video_path}: {len(ydev)} frame labels for {T} frames")
        if self.n_actions is not None and (ydev.min() < 0 or ydev.max() >= self.n_actions):
            raise ValueError(f"{video_path}: frame labels in [{int(ydev.min())}, {int(ydev.max())}] "
                             f"but n_actions={self.n_actions}")
        if not 0 <= int(ycat) < self.n_tclasses:
            raise ValueError(f"{video_path}: ycat={int(ycat)} but n_tclasses={self.n_tclasses}")
        print(embedding_name)
        return {
            'embedding': embedding,
            'ytrans': ytrans,
            'ydev': ydev,
            'ycat': ycat,
            'has_seg1': lag[0], 'lag_s1': lag[1], 'lag_e1': lag[2],
            'has_seg2': lag[3], 'lag_s2': lag[4], 'lag_e2': lag[5],
            'has_seg3': lag[6], 'lag_s3': lag[7], 'lag_e3': lag[8],
            'n_segment': lag[9],
            'video_name': video_path,
        }

    @staticmethod
    def add_specific_args(parent_parser: ArgumentParser) -> ArgumentParser:
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--version', type=str, choices=["v1", "v2", "v3", "v4", "cs"],
                            help='ciray=embryos with standard development, cda= embryos with standard development, direct cleavage, reverse cleavage and developmental arrest')
        parser.add_argument('--cdt_annotation', action="store_true", help='Whether to add the CDT class or not')
        parser.add_argument('--blocked', action="store_true", help='Whether to add the BLOCKED class or not')
        parser.add_argument('--max_frame', type=int)
        return parser


class SequenceDataModule(pl.LightningDataModule):
    """
    @brief DataModule for training, validation, and testing sequences.

    @param dir_path (str) Path to the directory containing the train, val, and test folders with CSV files.
    @param data_name (str) Name of the CSV file containing data annotations.
    @param batch_size (int) Number of samples per batch (training only, val/test use 1).
    @param num_workers (int) Number of parallel workers for data loading.

    Example directory structure:
    └── Datasplit
        └── Bovine
            ├── train
            │   └── Split.csv
            ├── val
            │   └── Split.csv
            └── test
                └── Split.csv
    """

    def __init__(self, dir_path: str, data_name: str, batch_size: int, num_workers: int,  **kwargs):
        super().__init__()
        self._DATA_NAME = data_name
        self._DIR_PATH = dir_path
        self._BATCH_SIZE = batch_size
        self._NUM_WORKERS = num_workers

        self.set_dataset_parameters(**kwargs)

    def set_dataset_parameters(self, **kwargs):
        self.version = kwargs.get("version", "v4")
        self.cdt_annotation = kwargs.get("cdt_annotation", True)
        self.blocked = kwargs.get("blocked", True)
        self.max_frame = kwargs.get("max_frame", 301)
        self.n_tclasses = kwargs["n_tclasses"]
        self.n_actions = kwargs.get("n_actions", None)
        self.embeddings_path = kwargs.get("embeddings_path", None)

    def _make_dataset(self, split: str) -> SequenceDataset:
        return SequenceDataset(
            data_path=os.path.join(self._DIR_PATH, split, self._DATA_NAME),
            mode=split,
            cdt_annotation=self.cdt_annotation,
            blocked=self.blocked,
            version=self.version,
            max_frame=self.max_frame,
            n_tclasses=self.n_tclasses,
            n_actions=self.n_actions,
            embeddings_path=self.embeddings_path,
        )

    def train_dataloader(self):
        train_dataset = self._make_dataset("train")

        # Class balanced sampling on the category
        labels = torch.tensor([train_dataset.category_to_class[train_dataset.data.loc[v, 'Category']]
                               for v in train_dataset.video_paths])
        class_counts = torch.bincount(labels)
        class_weights = 1.0 / class_counts.float().clamp(min=1)
        sample_weights = class_weights[labels]

        sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

        return DataLoader(
            train_dataset,
            batch_size=self._BATCH_SIZE,
            num_workers=self._NUM_WORKERS,
            drop_last=True,
            pin_memory=True,
            sampler=sampler,  # shuffle must stay False with a sampler
        )

    def val_dataloader(self):
        return DataLoader(self._make_dataset("val"), batch_size=1, num_workers=self._NUM_WORKERS, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self._make_dataset("test"), batch_size=1, num_workers=self._NUM_WORKERS, shuffle=False)

    def predict_dataloader(self):
        """Alias of test_dataloader for Lightning's predict."""
        return self.test_dataloader()

    @staticmethod
    def add_specific_args(parent_parser: ArgumentParser) -> ArgumentParser:
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser = SequenceDataset.add_specific_args(parser)
        parser.add_argument('--batch_size', default=16, type=int, help='Batch size for training')
        parser.add_argument('--num_workers', default=8, type=int, help='Number of workers for data loading')
        parser.add_argument('--data_name', type=str, help='Name of the CSV file for dataset')
        parser.add_argument('--dir_path', type=str, help='Path to the root directory of the dataset')
        parser.add_argument('--embeddings_path', type=str, default=None, help='Folder of the .npy embeddings')
        return parser