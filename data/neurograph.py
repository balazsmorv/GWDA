from __future__ import annotations

from typing import Callable, Optional, Sequence, Tuple, Dict, Any
import os
import random

import torch
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import NeuroGraphDataset


class NeuroGraphDataModule:
    """Convenience wrapper around torch_geometric.datasets.NeuroGraphDataset.

    Responsibilities:
    - Lazy-load dataset with optional transforms.
    - Create deterministic train/val/test splits (stratified by labels when available).
    - Provide PyG `DataLoader`s for each split.
    - Expose basic utilities to inspect domains/sites if present on data objects.
    """

    # Supported dataset variants as described by the NeuroGraph paper/release
    VALID_DATASET_NAMES = [
        "HCPTask",   # task-based fMRI (graph classification)
        "HCPGender", # graph classification
        "HCPAge",    # graph classification
        "HCPFI",     # graph regression (Fluid Intelligence)
        "HCPWM",     # graph regression (Working Memory)
    ]

    def __init__(
        self,
        root: str,
        *,
        dataset_name: str = "HCPTask",
        batch_size: int = 32,
        num_workers: int = 0,
        val_size: float = 0.1,
        test_size: float = 0.1,
        shuffle: bool = True,
        seed: int = 42,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
        pin_memory: bool = False,
        persistent_workers: bool = False,
    ) -> None:
        if not (0.0 <= val_size < 1.0) or not (0.0 <= test_size < 1.0):
            raise ValueError("val_size and test_size must be in [0,1)")
        if val_size + test_size >= 0.99:
            raise ValueError("val_size + test_size must be < 1.0")

        self.root = os.path.abspath(root)
        if dataset_name not in self.VALID_DATASET_NAMES:
            raise ValueError(
                f"dataset_name must be one of {self.VALID_DATASET_NAMES}, got: {dataset_name!r}"
            )
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_size = val_size
        self.test_size = test_size
        self.shuffle = shuffle
        self.seed = seed
        self.transform = transform
        self.pre_transform = pre_transform
        self.pre_filter = pre_filter
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers if num_workers > 0 else False

        self._dataset = None  # type: Optional[NeuroGraphDataset]
        self._train = None  # type: Optional[Subset]
        self._val = None  # type: Optional[Subset]
        self._test = None  # type: Optional[Subset]

    @property
    def dataset(self) -> NeuroGraphDataset:
        if self._dataset is None:
            self._dataset = NeuroGraphDataset(
                root=self.root,
                name=self.dataset_name,
                transform=self.transform,
                pre_transform=self.pre_transform,
                pre_filter=self.pre_filter,
            )
        return self._dataset

    @staticmethod
    def _extract_labels(dataset: Sequence[Any]) -> Optional[Sequence[int]]:
        labels: list[int] = []
        has_any_label = False
        for data in dataset:
            y = getattr(data, "y", None)
            if y is None:
                labels.append(-1)
                continue
            if isinstance(y, torch.Tensor):
                if y.numel() == 0:
                    labels.append(-1)
                else:
                    has_any_label = True
                    labels.append(int(y.view(-1)[0].item()))
            else:
                try:
                    labels.append(int(y))
                    has_any_label = True
                except Exception:
                    labels.append(-1)
        return labels if has_any_label else None

    def _split_indices(self, n: int, labels: Optional[Sequence[int]]) -> Tuple[Sequence[int], Sequence[int], Sequence[int]]:
        rng = random.Random(self.seed)
        indices = list(range(n))
        if self.shuffle:
            rng.shuffle(indices)

        n_test = int(round(self.test_size * n))
        n_val = int(round(self.val_size * n))
        n_train = max(0, n - n_val - n_test)

        # Simple split by shuffled indices; stratified split can be added if needed later.
        test_indices = indices[:n_test]
        val_indices = indices[n_test:n_test + n_val]
        train_indices = indices[n_test + n_val: n_test + n_val + n_train]
        return train_indices, val_indices, test_indices

    def setup(self, stage: Optional[str] = None) -> None:
        ds = self.dataset
        labels = self._extract_labels(ds)
        train_idx, val_idx, test_idx = self._split_indices(len(ds), labels)
        self._train = Subset(ds, train_idx)
        self._val = Subset(ds, val_idx)
        self._test = Subset(ds, test_idx)

    @property
    def train_dataset(self) -> Subset:
        if self._train is None:
            self.setup()
        assert self._train is not None
        return self._train

    @property
    def val_dataset(self) -> Subset:
        if self._val is None:
            self.setup()
        assert self._val is not None
        return self._val

    @property
    def test_dataset(self) -> Subset:
        if self._test is None:
            self.setup()
        assert self._test is not None
        return self._test

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
        )

    def domain_counts(self) -> Dict[str, int]:
        """If each data object has a `domain` or `site` attribute, count occurrences.

        Returns an empty dict if attribute is not available.
        """
        ds = self.dataset
        counts: Dict[str, int] = {}
        for data in ds:
            value = None
            for key in ("domain", "site"):
                if hasattr(data, key):
                    v = getattr(data, key)
                    if isinstance(v, torch.Tensor):
                        if v.numel() > 0:
                            value = str(int(v.view(-1)[0].item()))
                    else:
                        value = str(v)
                    if value is not None:
                        break
            if value is None:
                continue
            counts[value] = counts.get(value, 0) + 1
        return counts


__all__ = ["NeuroGraphDataModule"]
