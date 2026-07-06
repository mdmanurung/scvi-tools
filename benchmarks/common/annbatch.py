"""Optional AnnBatch datamodule for CytoANVI benchmark training."""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass
from itertools import cycle
from pathlib import Path
from typing import TYPE_CHECKING

import anndata as ad
import lightning.pytorch as pl
import numpy as np
import pandas as pd
import torch

from scvi.dataloaders import SemiSupervisedDataSplitter
from scvi.external.cytovi._constants import CYTOVI_REGISTRY_KEYS

if TYPE_CHECKING:
    from collections.abc import Iterator
    from typing import Any, Literal


@dataclass(frozen=True)
class AnnBatchConfig:
    """Configuration for the opt-in benchmark AnnBatch backend."""

    enabled: bool = False
    cache_dir: str | Path | None = None
    chunk_size: int = 512
    preload_nchunks: int = 32
    cache_mode: Literal["temporary", "reuse"] = "temporary"
    cache_key: str | None = None


class UnsupportedAnnBatchRegistry(ValueError):
    """Raised when the registered tensors cannot be represented by the benchmark adapter."""


_SUPPORTED_KEYS = {
    CYTOVI_REGISTRY_KEYS.X_KEY,
    CYTOVI_REGISTRY_KEYS.BATCH_KEY,
    CYTOVI_REGISTRY_KEYS.LABELS_KEY,
    CYTOVI_REGISTRY_KEYS.SAMPLE_KEY,
    CYTOVI_REGISTRY_KEYS.PROTEIN_NAN_MASK,
}


def _require_annbatch():
    try:
        from annbatch import DatasetCollection, Loader
    except ModuleNotFoundError as err:
        if err.name not in (None, "annbatch"):
            raise
        raise ImportError(
            "AnnBatch backend requested, but annbatch is not installed. "
            "Install the optional extra `scvi-tools[cytoanvi-annbatch]`."
        ) from err
    return DatasetCollection, Loader


def make_cytoanvi_annbatch_datamodule(
    model,
    config: AnnBatchConfig,
    *,
    batch_size: int,
    train_size: float = 0.9,
    validation_size: float | None = None,
    shuffle_set_split: bool = True,
    drop_last: bool = False,
):
    """Build a Lightning datamodule that feeds CytoANVI from AnnBatch."""
    if not config.enabled:
        raise ValueError("AnnBatchConfig.enabled must be True to build an AnnBatch datamodule.")
    dataset_collection_cls, loader_cls = _require_annbatch()
    return AnnBatchSemiSupervisedDataModule(
        model.adata_manager,
        config=config,
        batch_size=batch_size,
        train_size=train_size,
        validation_size=validation_size,
        shuffle_set_split=shuffle_set_split,
        drop_last=drop_last,
        dataset_collection_cls=dataset_collection_cls,
        loader_cls=loader_cls,
    )


class AnnBatchSemiSupervisedDataModule(pl.LightningDataModule):
    """Semi-supervised CytoANVI datamodule backed by AnnBatch loaders."""

    def __init__(
        self,
        adata_manager,
        *,
        config: AnnBatchConfig,
        batch_size: int,
        train_size: float = 0.9,
        validation_size: float | None = None,
        shuffle_set_split: bool = True,
        drop_last: bool = False,
        dataset_collection_cls,
        loader_cls,
    ):
        super().__init__()
        self.adata_manager = adata_manager
        self.config = config
        self.batch_size = batch_size
        self.train_size = train_size
        self.validation_size = validation_size
        self.shuffle_set_split = shuffle_set_split
        self.drop_last = drop_last
        self._dataset_collection_cls = dataset_collection_cls
        self._loader_cls = loader_cls
        self._loader_cache: dict[str, Any] = {}
        self._setup_complete = False

        registry_keys = set(adata_manager.data_registry.keys())
        unsupported = sorted(registry_keys - _SUPPORTED_KEYS)
        if unsupported:
            raise UnsupportedAnnBatchRegistry(
                "AnnBatch CytoANVI benchmark backend supports only registered tensors "
                f"{sorted(_SUPPORTED_KEYS)}; found unsupported entries {unsupported}."
            )

        self._has_nan_layer = CYTOVI_REGISTRY_KEYS.PROTEIN_NAN_MASK in registry_keys
        self._n_vars = int(adata_manager.adata.n_vars)
        self._obs_keys = [
            key
            for key in (
                CYTOVI_REGISTRY_KEYS.BATCH_KEY,
                CYTOVI_REGISTRY_KEYS.LABELS_KEY,
                CYTOVI_REGISTRY_KEYS.SAMPLE_KEY,
            )
            if key in registry_keys
        ]
        cache_root = Path(config.cache_dir or ".scratch/cytoanvi-benchmark/annbatch-cache")
        if config.cache_mode == "temporary":
            self._cache_root = cache_root / f"run-{uuid.uuid4().hex}"
        elif config.cache_mode == "reuse":
            if not config.cache_key:
                raise ValueError("AnnBatchConfig.cache_key is required when cache_mode='reuse'.")
            self._cache_root = cache_root / _sanitize_cache_key(config.cache_key)
        else:
            raise ValueError("AnnBatchConfig.cache_mode must be one of {'temporary', 'reuse'}.")

    def setup(self, stage: str | None = None):
        """Create train/validation splits that match scvi's semi-supervised splitter."""
        if self._setup_complete:
            return
        splitter = SemiSupervisedDataSplitter(
            adata_manager=self.adata_manager,
            train_size=self.train_size,
            validation_size=self.validation_size,
            shuffle_set_split=self.shuffle_set_split,
            n_samples_per_label=None,
            batch_size=self.batch_size,
            drop_last=self.drop_last,
        )
        splitter.setup(stage=stage)
        self.train_idx = splitter.train_idx
        self.val_idx = splitter.val_idx
        self.test_idx = splitter.test_idx
        self.n_train = splitter.n_train
        self.n_val = splitter.n_val
        self._labeled_indices = np.asarray(splitter._labeled_indices, dtype=int)
        self._setup_complete = True

    def train_dataloader(self):
        """Return AnnBatch loaders for full and labelled training cells."""
        return self._semi_loader("train", self.train_idx, shuffle=True, drop_last=self.drop_last)

    def val_dataloader(self):
        """Return AnnBatch loaders for validation cells when present."""
        if len(self.val_idx) == 0:
            return None
        return self._semi_loader("val", self.val_idx, shuffle=False, drop_last=False)

    def test_dataloader(self):
        """Return AnnBatch loaders for test cells when present."""
        if len(self.test_idx) == 0:
            return None
        return self._semi_loader("test", self.test_idx, shuffle=False, drop_last=False)

    def _semi_loader(
        self,
        name: str,
        indices: np.ndarray,
        *,
        shuffle: bool,
        drop_last: bool,
    ):
        cache_key = f"{name}:{shuffle}:{drop_last}"
        if cache_key in self._loader_cache:
            return self._loader_cache[cache_key]

        full = self._tensor_loader(
            f"{name}-full",
            np.asarray(indices, dtype=int),
            shuffle=shuffle,
            drop_last=drop_last,
        )
        labelled_idx = np.intersect1d(np.asarray(indices, dtype=int), self._labeled_indices)
        labelled = None
        if len(labelled_idx) > 0:
            labelled = self._tensor_loader(
                f"{name}-labelled",
                labelled_idx,
                shuffle=shuffle,
                drop_last=drop_last,
            )
        loader = _CyclingSemiSupervisedLoader(full, labelled)
        self._loader_cache[cache_key] = loader
        return loader

    def _tensor_loader(
        self,
        name: str,
        indices: np.ndarray,
        *,
        shuffle: bool,
        drop_last: bool,
    ):
        compact = self._compact_adata(indices)
        collection = self._write_collection(name, compact)
        loader = self._loader_cls(
            batch_size=self.batch_size,
            chunk_size=self.config.chunk_size,
            preload_nchunks=self.config.preload_nchunks,
            shuffle=shuffle,
            drop_last=drop_last,
            preload_to_gpu=False,
            to_torch=True,
        ).use_collection(collection, load_adata=self._load_compact_collection_adata)
        return _AnnBatchTensorLoader(
            loader,
            obs_keys=self._obs_keys,
            has_nan_layer=self._has_nan_layer,
            n_vars=self._n_vars,
        )

    def _compact_adata(self, indices: np.ndarray) -> ad.AnnData:
        x = _as_2d_numpy(self.adata_manager.get_from_registry(CYTOVI_REGISTRY_KEYS.X_KEY)[indices])
        x = x.astype(np.float32, copy=False)
        if self._has_nan_layer:
            nan = _as_2d_numpy(
                self.adata_manager.get_from_registry(CYTOVI_REGISTRY_KEYS.PROTEIN_NAN_MASK)[
                    indices
                ]
            ).astype(np.float32, copy=False)
            x = np.concatenate([x, nan], axis=1)

        obs = pd.DataFrame(index=self.adata_manager.adata.obs_names[indices])
        for key in self._obs_keys:
            values = np.asarray(self.adata_manager.get_from_registry(key)[indices]).reshape(-1)
            obs[key] = values.astype(np.int64, copy=False)
        var = pd.DataFrame(index=[f"feature_{i}" for i in range(x.shape[1])])
        return ad.AnnData(X=x, obs=obs, var=var)

    def _write_collection(self, name: str, compact: ad.AnnData):
        split_dir = self._cache_root / name
        split_dir.mkdir(parents=True, exist_ok=True)
        source_path = split_dir / "source.h5ad"
        store_path = split_dir / "collection.zarr"
        if self.config.cache_mode == "reuse" and store_path.exists():
            return self._dataset_collection_cls(str(store_path))
        compact.write_h5ad(source_path)
        collection = self._dataset_collection_cls(str(store_path))
        collection.add_adatas(
            [source_path],
            load_adata=self._load_compact_adata,
            n_obs_per_chunk=max(1, int(self.config.chunk_size)),
            dataset_size=max(1, int(compact.n_obs)),
            shuffle_chunk_size=max(1, min(int(compact.n_obs), int(self.config.chunk_size))),
            shuffle=False,
        )
        return collection

    def _load_compact_adata(self, path_or_group):
        loaded = ad.experimental.read_lazy(path_or_group, load_annotation_index=False)
        obs = (
            loaded.obs[self._obs_keys]
            if self._obs_keys
            else pd.DataFrame(index=loaded.obs_names)
        )
        if hasattr(obs, "to_memory"):
            obs = obs.to_memory()
        else:
            obs = pd.DataFrame(obs)
        var = loaded.var
        if hasattr(var, "to_memory"):
            var = var.to_memory()
        return ad.AnnData(X=loaded.X, obs=obs, var=var)

    def _load_compact_collection_adata(self, group):
        from annbatch.utils import load_x_and_obs_and_var

        loaded = load_x_and_obs_and_var(group)
        obs = (
            loaded.obs[self._obs_keys]
            if self._obs_keys
            else pd.DataFrame(index=loaded.obs_names)
        )
        return ad.AnnData(X=loaded.X, obs=obs, var=loaded.var)


class _CyclingSemiSupervisedLoader:
    def __init__(self, full_loader, labelled_loader=None):
        self.full_loader = full_loader
        self.labelled_loader = labelled_loader

    def __len__(self) -> int:
        return len(self.full_loader)

    def __iter__(self) -> Iterator[dict[str, torch.Tensor] | tuple[dict, dict]]:
        if self.labelled_loader is None:
            yield from self.full_loader
            return

        labelled_iter = cycle(self.labelled_loader)
        for full_batch in self.full_loader:
            yield full_batch, next(labelled_iter)


class _AnnBatchTensorLoader:
    def __init__(self, loader, *, obs_keys: list[str], has_nan_layer: bool, n_vars: int):
        self.loader = loader
        self.obs_keys = obs_keys
        self.has_nan_layer = has_nan_layer
        self.n_vars = n_vars
        self._obs_torch_dtypes = {
            CYTOVI_REGISTRY_KEYS.BATCH_KEY: torch.int64,
            CYTOVI_REGISTRY_KEYS.LABELS_KEY: torch.int64,
            CYTOVI_REGISTRY_KEYS.SAMPLE_KEY: torch.float32,
        }

    def __len__(self) -> int:
        return len(self.loader)

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        for batch in self.loader:
            yield self._format_batch(batch)

    def _format_batch(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        x = _to_dense_tensor(batch["X"]).to(dtype=torch.float32)
        tensors: dict[str, torch.Tensor] = {}
        if self.has_nan_layer:
            tensors[CYTOVI_REGISTRY_KEYS.X_KEY] = x[:, : self.n_vars].contiguous()
            tensors[CYTOVI_REGISTRY_KEYS.PROTEIN_NAN_MASK] = x[:, self.n_vars :].contiguous()
        else:
            tensors[CYTOVI_REGISTRY_KEYS.X_KEY] = x.contiguous()

        obs = batch["obs"]
        for key in self.obs_keys:
            values = np.asarray(obs[key]).reshape(-1, 1)
            tensors[key] = torch.as_tensor(values, dtype=self._obs_torch_dtypes[key])
        return tensors


def _as_2d_numpy(x) -> np.ndarray:
    if hasattr(x, "toarray"):
        x = x.toarray()
    else:
        x = np.asarray(x)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    return x


def _to_dense_tensor(x) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        if x.layout != torch.strided:
            x = x.to_dense()
        return x
    if hasattr(x, "toarray"):
        x = x.toarray()
    return torch.as_tensor(np.asarray(x))


def _sanitize_cache_key(cache_key: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(cache_key)).strip("._-")
    if not safe:
        raise ValueError("AnnBatchConfig.cache_key must contain at least one safe character.")
    return safe
