import re
from collections import OrderedDict
from typing import List, Optional

import pandas as pd
import torch
from omegaconf import DictConfig
from torch.utils.data import Dataset

from cross_modal_distillation.build import build_module
from cross_modal_distillation.data.metadata import Metadata
from cross_modal_distillation.utility.utils import init_logger

torch_version = torch.__version__.split("+")[0]


class MultiSessionDataset(Dataset):
    def __init__(
        self,
        config: DictConfig,
        max_cache_size: int = 50000,
    ):
        self.config = config
        self.logger = init_logger(name=self.__class__.__name__)

        dataset_metadatas = self.build_datasets()
        self.metadata = self.build_metadata(dataset_metadatas)

        self.max_cache_size = max_cache_size
        self.data_cache = OrderedDict()

    def build_datasets(self):
        dataset_metadatas = []
        for dataset_name, dataset_config in self.config.items():
            self.logger.info(
                f"Building {dataset_name} dataset with module {dataset_config._target_}."
            )
            dataset = build_module(config=dataset_config)
            dataset_metadatas.append(dataset.metadata)
        return dataset_metadatas

    def build_metadata(self, dataset_metadatas: List[pd.DataFrame]) -> Metadata:
        metadata = Metadata.merge_metadatas(dataset_metadatas, drop_duplicate=False)
        metadata.shuffle()
        return metadata

    def reduce_metadata(
        self,
        subject_sessions: List[str],
        exclude_subject_sessions: Optional[List[str]] = None,
    ):
        if not isinstance(subject_sessions, list):
            subject_sessions = [subject_sessions]

        combined_pattern = "|".join(subject_sessions)

        self.metadata.reduce_based_on_col_value(
            col_name="subject_session", value=combined_pattern, regex=True
        )

        if exclude_subject_sessions:
            if not isinstance(exclude_subject_sessions, list):
                exclude_subject_sessions = [exclude_subject_sessions]
            exclude_combined_pattern = "|".join(exclude_subject_sessions)
            self.metadata.reduce_based_on_col_value(
                col_name="subject_session",
                value=exclude_combined_pattern,
                regex=True,
                inverse=True,
            )

    def set_split(self, split: str):
        self.metadata.reduce_based_on_col_value(col_name="split", value=split)

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        meta_row = self.metadata.get_row_by_index(idx)
        segment_path = meta_row["path"]

        if segment_path not in self.data_cache:
            data = torch.load(segment_path, weights_only=True)
            if len(self.data_cache) >= self.max_cache_size:
                first_path = next(iter(self.data_cache))
                self.data_cache.pop(first_path)
            self.data_cache[segment_path] = data
        else:
            data = self.data_cache[segment_path]

        batch_dict = dict(
            input=data["lfp"],
            subject_session=meta_row.subject_session,
            segment_filename=meta_row.segment_filename,
        )
        return batch_dict
