# Copyright (c) 2023, Zikang Zhou. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Modified by TSG author, 2026.

import os
from typing import Callable, Optional

import pytorch_lightning as pl
import torch
import torch.distributed as dist
from torch_geometric.loader import DataLoader
from torch.utils.data import RandomSampler, SequentialSampler, Subset
from torch.utils.data.distributed import DistributedSampler

from datasets import ArgoverseV2Dataset
from datamodules.samplers import LargeSampleAwareBatchSampler
from transforms import TargetBuilder


class ArgoverseV2DataModule(pl.LightningDataModule):

    def __init__(self,
                 root: str,
                 train_batch_size: int,
                 val_batch_size: int,
                 test_batch_size: int,
                 shuffle: bool = True,
                 num_workers: int = 0,
                 pin_memory: bool = True,
                 persistent_workers: bool = True,
                 train_raw_dir: Optional[str] = None,
                 val_raw_dir: Optional[str] = None,
                 test_raw_dir: Optional[str] = None,
                 train_processed_dir: Optional[str] = None,
                 val_processed_dir: Optional[str] = None,
                 test_processed_dir: Optional[str] = None,
                 train_transform: Optional[Callable] = TargetBuilder(50, 60),
                 val_transform: Optional[Callable] = TargetBuilder(50, 60),
                 test_transform: Optional[Callable] = None,
                 auto_prepare_data: bool = False,
                 limit_large_samples: bool = False,
                 large_sample_threshold_kb: int = 300,
                 max_large_per_batch: int = 1,
                 sampler_seed: int = 2023,
                 sampler_drop_last: bool = False,
                 max_file_size_kb: Optional[int] = None,
                 **kwargs) -> None:
        super(ArgoverseV2DataModule, self).__init__()
        self.root = root
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.test_batch_size = test_batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers and num_workers > 0
        self.train_raw_dir = train_raw_dir
        self.val_raw_dir = val_raw_dir
        self.test_raw_dir = test_raw_dir
        self.train_processed_dir = train_processed_dir
        self.val_processed_dir = val_processed_dir
        self.test_processed_dir = test_processed_dir
        self.train_transform = train_transform
        self.val_transform = val_transform
        self.test_transform = test_transform
        self.auto_prepare_data = auto_prepare_data
        self.limit_large_samples = limit_large_samples
        self.large_sample_threshold_kb = large_sample_threshold_kb
        self.max_large_per_batch = max_large_per_batch
        self.sampler_seed = sampler_seed
        self.sampler_drop_last = sampler_drop_last
        self.max_file_size_kb = max_file_size_kb

    def prepare_data(self) -> None:
        if not self.auto_prepare_data:
            return
        ArgoverseV2Dataset(self.root, 'train', self.train_raw_dir, self.train_processed_dir, self.train_transform,
                           auto_prepare=True)
        ArgoverseV2Dataset(self.root, 'val', self.val_raw_dir, self.val_processed_dir, self.val_transform,
                           auto_prepare=True)
        ArgoverseV2Dataset(self.root, 'test', self.test_raw_dir, self.test_processed_dir, self.test_transform,
                           auto_prepare=True)

    def setup(self, stage: Optional[str] = None) -> None:
        self.train_dataset = ArgoverseV2Dataset(self.root, 'train', self.train_raw_dir, self.train_processed_dir,
                                                self.train_transform, auto_prepare=self.auto_prepare_data)
        self.val_dataset = ArgoverseV2Dataset(self.root, 'val', self.val_raw_dir, self.val_processed_dir,
                                              self.val_transform, auto_prepare=self.auto_prepare_data)
        self.test_dataset = ArgoverseV2Dataset(self.root, 'test', self.test_raw_dir, self.test_processed_dir,
                                               self.test_transform, auto_prepare=self.auto_prepare_data)
        # Apply max_file_size_kb filter to all splits (train, val, test)
        if self.max_file_size_kb is not None:
            self.train_dataset = self._filter_dataset_by_size(self.train_dataset, self.max_file_size_kb)
            self.val_dataset = self._filter_dataset_by_size(self.val_dataset, self.max_file_size_kb)
            self.test_dataset = self._filter_dataset_by_size(self.test_dataset, self.max_file_size_kb)

    def train_dataloader(self):
        if self.limit_large_samples:
            file_sizes = self._get_processed_file_sizes(self.train_dataset)
            base_sampler = self._build_base_sampler(self.train_dataset)
            batch_sampler = LargeSampleAwareBatchSampler(
                sampler=base_sampler,
                file_sizes_kb=file_sizes,
                batch_size=self.train_batch_size,
                large_threshold_kb=self.large_sample_threshold_kb,
                max_large_per_batch=min(self.max_large_per_batch, self.train_batch_size),
                drop_last=self.sampler_drop_last,
            )
            return DataLoader(self.train_dataset, batch_sampler=batch_sampler,
                              num_workers=self.num_workers, 
                              pin_memory=self.pin_memory and self.num_workers <= 4,  # Disable if too many workers
                              persistent_workers=self.persistent_workers)
        return DataLoader(self.train_dataset, batch_size=self.train_batch_size, shuffle=self.shuffle,
                          num_workers=self.num_workers, 
                          pin_memory=self.pin_memory and self.num_workers <= 4,  # Disable if too many workers
                          persistent_workers=self.persistent_workers)

    def val_dataloader(self):
        # Reduce pin_memory for validation to avoid "too many open files" error
        # Validation doesn't need pin_memory as much as training
        return DataLoader(self.val_dataset, batch_size=self.val_batch_size, shuffle=False,
                          num_workers=min(self.num_workers, 2),  # Limit workers for val
                          pin_memory=False,  # Disable pin_memory for val to avoid errors
                          persistent_workers=self.persistent_workers and min(self.num_workers, 2) > 0)

    def test_dataloader(self):
        # Reduce pin_memory for test to avoid "too many open files" error
        return DataLoader(self.test_dataset, batch_size=self.test_batch_size, shuffle=False,
                          num_workers=min(self.num_workers, 2),  # Limit workers for test
                          pin_memory=False,  # Disable pin_memory for test to avoid errors
                          persistent_workers=self.persistent_workers and min(self.num_workers, 2) > 0)
    def _build_base_sampler(self, dataset):
        world_size, rank = self._get_distributed_context()
        if world_size > 1:
            return DistributedSampler(dataset,
                                      num_replicas=world_size,
                                      rank=rank,
                                      shuffle=self.shuffle,
                                      seed=self.sampler_seed,
                                      drop_last=self.sampler_drop_last)
        if self.shuffle:
            generator = torch.Generator()
            generator.manual_seed(self.sampler_seed)
            return RandomSampler(dataset, generator=generator)
        return SequentialSampler(dataset)

    @staticmethod
    def _get_processed_file_sizes(dataset):
        indices = None
        base_dataset = dataset
        if isinstance(dataset, Subset):
            indices = dataset.indices
            base_dataset = dataset.dataset
        if not hasattr(base_dataset, '_file_sizes_kb'):
            base_dataset._file_sizes_kb = [os.path.getsize(path) / 1024 for path in base_dataset.processed_paths]
        if indices is None:
            return base_dataset._file_sizes_kb
        return [base_dataset._file_sizes_kb[i] for i in indices]

    @staticmethod
    def _filter_dataset_by_size(dataset, max_file_size_kb: int):
        file_sizes = ArgoverseV2DataModule._get_processed_file_sizes(dataset)
        keep_indices = [idx for idx, size in enumerate(file_sizes) if size <= max_file_size_kb]
        if len(keep_indices) == len(file_sizes):
            return dataset
        if len(keep_indices) == 0:
            raise ValueError('Filtering by file size removed all samples.')
        return Subset(dataset, keep_indices)

    @staticmethod
    def _get_distributed_context():
        try:
            if dist.is_available() and dist.is_initialized():
                return dist.get_world_size(), dist.get_rank()
        except (RuntimeError, ValueError):
            pass
        return 1, 0
