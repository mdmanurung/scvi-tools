"""Private registry-adapter prototype for MrTotalVI data collections.

Extends the generic streaming-registry pattern used by
:class:`~scvi.dataloaders.MappedCollectionDataModule` (see
``src/scvi/dataloaders/_custom_dataloaders.py``) with a protein-expression
tensor and a per-sample (donor) axis, matching what
:meth:`~scvi.external.mrtotalvi.MrTotalVI.setup_anndata` registers for the
in-memory :class:`~scvi.data.AnnDataManager` path.

**Quarantined: not a model-training surface.** This module is not exported,
accepted by :class:`MrTotalVI`, or supported for end-to-end train/infer/save/load.
It exists only to preserve and test registry-construction mechanics.

Known gaps (see class docstring for detail)
---------------------------------------------
* ``panel_key`` (TOTALVI/MrTotalVI's per-panel protein masking) is **not**
  supported -- ``setup_anndata`` swaps in a ``"panel"`` batch field for the
  protein batch-mask computation and inserts a second ``"panel"`` registry
  entry; reproducing that for a streaming, possibly multi-file collection
  was out of scope here. Passing ``panel_key`` raises ``NotImplementedError``.
* ``size_factor_key`` is **not** supported (always registered empty, which
  matches ``MrTotalVI.setup_anndata``'s *default* ``size_factor_key=None``
  behavior, but an explicit key cannot be threaded through the streaming
  backend here). Passing a non-``None`` value raises ``NotImplementedError``.
* The **lamindb** backend is refused unconditionally. This private adapter
  cannot verify both row-aligned protein values and an authoritative protein
  feature axis through that backend, so accepting a mapped collection would
  be an untested and potentially silent feature-order failure.
* The tested, parity-verified backend is the in-memory
  :class:`_InMemoryProteinMappedDataset` adapter (built directly from
  ``AnnData``/``list[AnnData]``), used automatically when ``collection`` is
  not a lamindb ``Collection``. It requires all constituent ``AnnData``
  objects to share identical ``var_names`` (no inner/outer join logic).
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import torch
from anndata import AnnData
from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader, Dataset

import scvi
from scvi import REGISTRY_KEYS
from scvi.external.mrtotalvi._contracts import (
    authoritative_protein_names,
    matrix_row,
)
from scvi.model._utils import parse_device_args
from scvi.utils import dependencies

if TYPE_CHECKING:
    from typing import Any

    import lamindb as ln


def _is_lamindb_collection(obj: Any) -> bool:
    """Best-effort duck-typed check for a ``lamindb.Collection``.

    Anything that is not an in-memory ``AnnData``/``list``/``tuple`` and that
    exposes a ``.mapped`` method is treated as a lamindb-style collection.
    """
    if isinstance(obj, AnnData) or isinstance(obj, list | tuple):
        return False
    return hasattr(obj, "mapped")


class _InMemoryProteinMappedDataset(Dataset):
    """Minimal in-memory stand-in for a lamindb ``MappedCollection``.

    Wraps one or more in-memory :class:`~anndata.AnnData` objects (a CITE-seq
    "collection") and exposes just enough of the lamindb ``MappedCollection``
    surface (``encoders``, ``n_obs``, ``n_vars``, ``var_joint``,
    ``torch_worker_init_fn``, integer indexing) for
    :class:`_MrTotalVIRegistryAdapter` to build tensors from it -- including
    the protein-expression ``.obsm`` array, which real lamindb streaming may
    or may not support (see module docstring "Known gaps").

    This is the backend exercised by this repository's parity tests, since
    no lamindb instance is available without network access in CI/dev
    sandboxes. It is not a substitute for genuine multi-file streaming I/O
    (all constituent AnnData objects are held in memory).
    """

    def __init__(
        self,
        adatas: list[AnnData],
        obs_keys: list[str],
        protein_expression_obsm_key: str,
        protein_names_uns_key: str | None = None,
        continuous_keys: list[str] | None = None,
        require_protein_names: bool = False,
    ):
        if len(adatas) == 0:
            raise ValueError("`adatas` must contain at least one AnnData object.")
        self._adatas = list(adatas)
        self._obs_keys = list(dict.fromkeys(obs_keys))  # de-dup, preserve order
        # Continuous covariate keys are passed through as raw floats, never
        # run through a categorical encoder (unlike every other obs_key).
        self._continuous_keys = set(continuous_keys or [])
        self._categorical_obs_keys = [k for k in self._obs_keys if k not in self._continuous_keys]
        self._protein_expression_obsm_key = protein_expression_obsm_key

        var_names = np.asarray(self._adatas[0].var_names)
        for a in self._adatas[1:]:
            if not np.array_equal(np.asarray(a.var_names), var_names):
                raise ValueError(
                    "All AnnData objects in `adatas` must share identical var_names; "
                    "inner/outer join across files is not implemented by this backend."
                )
        self.var_joint = var_names
        self.n_vars = int(len(var_names))

        protein_width = self._adatas[0].obsm[protein_expression_obsm_key].shape[1]
        for a in self._adatas[1:]:
            if a.obsm[protein_expression_obsm_key].shape[1] != protein_width:
                raise ValueError(
                    "All AnnData objects in `adatas` must have the same number of "
                    f"protein columns in obsm['{protein_expression_obsm_key}']."
                )
        self.n_proteins = int(protein_width)
        axes = [
            authoritative_protein_names(
                adata,
                protein_expression_obsm_key=protein_expression_obsm_key,
                protein_names_uns_key=protein_names_uns_key,
                required=require_protein_names or len(self._adatas) > 1,
            )
            for adata in self._adatas
        ]
        available_axes = [axis for axis in axes if axis is not None]
        if available_axes:
            reference_axis = available_axes[0]
            for axis in axes:
                if axis is None or not np.array_equal(axis, reference_axis):
                    raise ValueError(
                        "All files must provide identical authoritative protein names "
                        "in exact order; equal width is insufficient."
                    )
            self.protein_names = reference_axis
        else:
            self.protein_names = np.arange(self.n_proteins)

        sizes = [a.n_obs for a in self._adatas]
        self._offsets = np.concatenate([[0], np.cumsum(sizes)]).astype(int)
        self.n_obs = int(self._offsets[-1])

        # Categories are derived the same way `_make_column_categorical`
        # derives them (`pd.Series(raw).astype("category").cat.categories`),
        # not via a bare `np.unique`. This matters: for an object/string obs
        # column, pandas categories come back `dtype=object`; for a plain
        # int64 obs column, they come back `dtype=int64`. A bare
        # `np.unique(raw)` on a string ndarray instead yields a fixed-width
        # `<U..` dtype, which would silently fail exact registry parity for
        # any int- or `<U..`-vs-`object`-typed batch/label/sample/covariate
        # key -- not just the string-valued keys this backend happens to be
        # tested with.
        self.categories: dict[str, np.ndarray] = {}
        self.encoders: dict[str, dict] = {}
        for key in self._categorical_obs_keys:
            raw = np.concatenate([np.asarray(a.obs[key].to_numpy()) for a in self._adatas])
            categories = pd.Series(raw).astype("category").cat.categories.to_numpy(copy=True)
            self.categories[key] = categories
            self.encoders[key] = {cls: i for i, cls in enumerate(categories)}

    def full_protein_matrix(self) -> np.ndarray:
        """Full (n_obs, n_proteins) protein matrix, concatenated across files."""
        mats = []
        for a in self._adatas:
            m = a.obsm[self._protein_expression_obsm_key]
            if isinstance(m, pd.DataFrame):
                m = m.to_numpy()
            else:
                m = m.toarray() if hasattr(m, "toarray") else np.asarray(m)
            mats.append(m)
        return np.concatenate(mats, axis=0).astype(np.float32)

    def full_encoded_column(self, key: str) -> np.ndarray:
        """Full (n_obs,) array of integer codes for obs column ``key``."""
        raw = np.concatenate([np.asarray(a.obs[key].to_numpy()) for a in self._adatas])
        encoder = self.encoders[key]
        return np.array([encoder[v] for v in raw], dtype=np.int64)

    def get_uns(self, key: str) -> Any:
        """``.uns[key]`` of the first constituent AnnData."""
        return self._adatas[0].uns[key]

    def torch_worker_init_fn(self, _worker_id: int) -> None:
        return None

    def __len__(self) -> int:
        return self.n_obs

    def _locate(self, idx: int) -> tuple[int, int]:
        file_idx = int(np.searchsorted(self._offsets, idx, side="right") - 1)
        return file_idx, int(idx - self._offsets[file_idx])

    def __getitem__(self, idx: int) -> dict[str, Any]:
        file_idx, local_idx = self._locate(int(idx))
        adata = self._adatas[file_idx]

        x_row = adata.X[local_idx]
        x_row = x_row.toarray().ravel() if hasattr(x_row, "toarray") else np.asarray(x_row).ravel()

        p_row = matrix_row(adata.obsm[self._protein_expression_obsm_key], local_idx)

        item: dict[str, Any] = {
            "X": x_row.astype(np.float32, copy=False),
            "protein_expression": p_row.astype(np.float32, copy=False),
            "ind_x": int(idx),
        }
        for key in self._categorical_obs_keys:
            raw_value = adata.obs[key].to_numpy()[local_idx]
            item[key] = int(self.encoders[key][raw_value])
        for key in self._continuous_keys:
            item[key] = float(adata.obs[key].to_numpy()[local_idx])
        return item


class _MrTotalVIRegistryAdapter(LightningDataModule):
    """Private, non-training registry adapter for MrTotalVI-shaped collections.

    Follows the constructor/``registry``/dataloader shape of
    :class:`~scvi.dataloaders.MappedCollectionDataModule`, extended with:

    * a ``proteins`` field (:data:`~scvi.REGISTRY_KEYS.PROTEIN_EXP_KEY`) --
      a :class:`~anndata.AnnData` ``.obsm`` array, mirroring
      ``fields.ProteinObsmField`` including the optional
      ``protein_batch_mask`` computed when some batch has all-zero protein
      counts;
    * MrTotalVI's per-sample (donor) axis via ``sample_key`` (required, not
      optional as in the generic reference);
    * an emitted ``ind_x`` tensor (:data:`~scvi.REGISTRY_KEYS.INDICES_KEY`),
      which MrTotalVI's counterfactual/batched-query code paths read
      (``MrTotalVI`` reads ``tensors[REGISTRY_KEYS.INDICES_KEY]``) but which
      the generic reference registers without ever emitting.

    Unlike the generic reference (built for scANVI-style semi-supervised
    models), MrTotalVI's ``labels_key`` has **no** ``unlabeled_category``
    concept: :meth:`MrTotalVI.setup_anndata` registers labels with a plain
    ``CategoricalObsField``. When ``labels_key`` (or ``batch_key``) is
    ``None``, a single dummy category (code ``0``) is registered, matching
    ``CategoricalObsField``'s default-attribute behavior exactly (verified
    against the in-memory registry, not assumed).

    Parameters
    ----------
    collection
        Data source. Either a lamindb ``Collection`` (streamed via
        ``.mapped()``, gated behind ``@dependencies("lamindb")`` -- see
        module docstring "Known gaps"), or an in-memory
        :class:`~anndata.AnnData` / ``list[AnnData]``, in which case the
        tested :class:`_InMemoryProteinMappedDataset` backend is used.
    protein_expression_obsm_key
        Key in ``adata.obsm`` for protein expression data.
    sample_key
        Key in ``adata.obs`` identifying the donor/sample for each cell.
        Required (MrTotalVI has no "no sample axis" mode).
    labels_key
        Optional key in ``adata.obs`` for cell-type/annotation labels.
    protein_names_uns_key
        Optional key in ``adata.uns`` for protein names. Only supported for
        the in-memory backend (see "Known gaps").
    batch_key
        Optional key in ``adata.obs`` for batch/panel identity. Also used
        (when set) to compute the protein batch mask, matching
        ``fields.ProteinObsmField(use_batch_mask=True)``.
    batch_size
        Default dataloader batch size.
    collection_val
        Optional held-out validation collection/AnnData, same shape rules
        as ``collection``.
    shuffle
        Whether ``train_dataloader()`` shuffles.
    model_name
        Recorded in ``registry["model_name"]``. Defaults to ``"MrTotalVI"``
        (the reference module defaults to ``"SCVI"``, which would be wrong
        here: :meth:`~scvi.model.base.BaseModelClass.load` validates this
        against the loading model class's ``__name__``).
    categorical_covariate_keys, continuous_covariate_keys
        Optional extra covariate keys, as in the generic reference.

    Notes
    -----
    This class does not accept ``size_factor_key`` or ``panel_key`` -- see
    the module docstring "Known gaps". ``MrTotalVI.__init__`` (inherited
    from :class:`~scvi.model.TOTALVI`) does not accept a ``registry=``
    keyword argument (unlike :class:`~scvi.model.SCVI`), so end-to-end
    ``MrTotalVI(registry=dm.registry)`` training/save/load is out of reach
    without also changing ``MrTotalVI``'s constructor -- which is
    explicitly out of scope for this class.
    """

    def __init__(
        self,
        collection: ln.Collection | AnnData | list[AnnData],
        protein_expression_obsm_key: str,
        sample_key: str,
        labels_key: str | None = None,
        protein_names_uns_key: str | None = None,
        batch_key: str | None = None,
        size_factor_key: str | None = None,
        panel_key: str | None = None,
        batch_size: int = 128,
        collection_val: ln.Collection | AnnData | list[AnnData] | None = None,
        accelerator: str = "auto",
        device: int | str = "auto",
        shuffle: bool = True,
        model_name: str = "MrTotalVI",
        categorical_covariate_keys: list[str] | None = None,
        continuous_covariate_keys: list[str] | None = None,
        **kwargs,
    ):
        super().__init__()
        if size_factor_key is not None:
            raise NotImplementedError(
                "`size_factor_key` is not supported by _MrTotalVIRegistryAdapter "
                "(the `size_factor` field is always registered empty). See the "
                "module docstring 'Known gaps'."
            )
        if panel_key is not None:
            raise NotImplementedError(
                "`panel_key` (per-panel protein batch masking) is not supported by "
                "_MrTotalVIRegistryAdapter. See the module docstring 'Known gaps'."
            )

        self._batch_size = batch_size
        self._protein_expression_obsm_key = protein_expression_obsm_key
        self._protein_names_uns_key = protein_names_uns_key
        self._sample_key = sample_key
        self._batch_key = batch_key
        self._label_key = labels_key
        self.model_name = model_name
        self.shuffle = shuffle
        self._parallel = kwargs.pop("parallel", True)
        self._categorical_covariate_keys = categorical_covariate_keys
        self._continuous_covariate_keys = continuous_covariate_keys

        obs_keys = [self._sample_key]
        if self._batch_key is not None:
            obs_keys.append(self._batch_key)
        if self._label_key is not None:
            obs_keys.append(self._label_key)
        if self._categorical_covariate_keys is not None:
            obs_keys.extend(self._categorical_covariate_keys)
        if self._continuous_covariate_keys is not None:
            obs_keys.extend(self._continuous_covariate_keys)
        self._obs_keys = list(dict.fromkeys(obs_keys))

        compare_validation_axis = collection_val is not None
        self._dataset = self._build_dataset(
            collection,
            self._obs_keys,
            require_protein_names=compare_validation_axis,
            **kwargs,
        )
        self._validset = (
            self._build_dataset(
                collection_val,
                self._obs_keys,
                require_protein_names=True,
                **kwargs,
            )
            if collection_val is not None
            else None
        )
        if self._validset is not None and not np.array_equal(
            self._dataset.protein_names,
            self._validset.protein_names,
        ):
            raise ValueError(
                "Training and validation collections must provide identical "
                "authoritative protein names in exact order."
            )

        # n_obs_per_sample: exposed as a convenience attribute (matching the
        # generic reference's convention), but deliberately *not* stored
        # inside the `sample` registry entry -- the real in-memory MrTotalVI
        # registry has no such key there, and this class targets exact
        # registry parity.
        counts = np.zeros(self.n_samples, dtype=np.float32)
        sample_codes = self._dataset.full_encoded_column(self._sample_key)
        for code in sample_codes:
            counts[code] += 1
        self.n_obs_per_sample = torch.tensor(counts, dtype=torch.float32)

        self._log_hyperparams = False
        self.allow_zero_length_dataloader_with_multiple_devices = False
        _, _, self.device = parse_device_args(
            accelerator=accelerator, devices=device, return_device="torch"
        )

    # ------------------------------------------------------------------
    # Dataset construction
    # ------------------------------------------------------------------

    def _build_dataset(
        self,
        collection,
        obs_keys: list[str],
        *,
        require_protein_names: bool = False,
        **kwargs,
    ):
        if collection is None:
            return None
        if isinstance(collection, AnnData):
            adatas = [collection]
        elif isinstance(collection, list | tuple):
            adatas = list(collection)
        elif _is_lamindb_collection(collection):
            return self._build_lamindb_dataset(collection, obs_keys, **kwargs)
        else:
            raise TypeError(
                "`collection` must be an AnnData, a list/tuple of AnnData, or a "
                f"lamindb Collection-like object exposing `.mapped()`; got {type(collection)}."
            )
        return _InMemoryProteinMappedDataset(
            adatas,
            obs_keys=obs_keys,
            protein_expression_obsm_key=self._protein_expression_obsm_key,
            protein_names_uns_key=self._protein_names_uns_key,
            continuous_keys=self._continuous_covariate_keys,
            require_protein_names=require_protein_names,
        )

    @dependencies("lamindb")
    def _build_lamindb_dataset(self, collection: ln.Collection, obs_keys: list[str], **kwargs):
        raise NotImplementedError(
            "_MrTotalVIRegistryAdapter refuses the lamindb backend because it cannot "
            "fail-closed verify both row-aligned protein values and an authoritative "
            "protein feature axis. Pass an in-memory AnnData/list[AnnData] to use "
            "the tested private registry adapter."
        )

    def close(self):
        if self._dataset is not None and hasattr(self._dataset, "close"):
            self._dataset.close()
        if self._validset is not None and hasattr(self._validset, "close"):
            self._validset.close()

    # ------------------------------------------------------------------
    # Dataloaders
    # ------------------------------------------------------------------

    def train_dataloader(self) -> DataLoader:
        return self._create_dataloader(self._dataset, shuffle=self.shuffle)

    def val_dataloader(self) -> DataLoader | None:
        return self._create_dataloader(self._validset, shuffle=False)

    def inference_dataloader(
        self,
        shuffle: bool | None = False,
        batch_size: int = 4096,
        indices=None,
        parallel_cpu_count: int | None = None,
    ) -> _InferenceDataloader:
        """Dataloader for inference with `on_before_batch_transfer` applied."""
        if shuffle is None:
            shuffle = self.shuffle
        dataloader = self._create_dataloader(
            self._dataset, shuffle, batch_size, indices, parallel_cpu_count
        )
        return self._InferenceDataloader(dataloader, self.on_before_batch_transfer)

    def _create_dataloader(
        self, dataset, shuffle, batch_size=None, indices=None, parallel_cpu_count=None
    ):
        if dataset is None:
            return None
        if self._parallel:
            num_workers = (
                max(0, (os.cpu_count() or 1) - 1)
                if parallel_cpu_count is None
                else parallel_cpu_count
            )
            worker_init_fn = dataset.torch_worker_init_fn
        else:
            num_workers = 0
            worker_init_fn = None
        if batch_size is None:
            batch_size = self._batch_size
        if indices is not None:
            if isinstance(dataset, _InMemoryProteinMappedDataset):
                from torch.utils.data import Subset

                dataset = Subset(dataset, list(indices))
            else:
                dataset = dataset[indices]
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            worker_init_fn=worker_init_fn,
        )

    class _InferenceDataloader:
        """Wrapper to apply `on_before_batch_transfer` during iteration."""

        def __init__(self, dataloader, transform_fn):
            self.dataloader = dataloader
            self.transform_fn = transform_fn

        def __iter__(self):
            for batch in self.dataloader:
                yield self.transform_fn(batch, dataloader_idx=None)

        def __len__(self):
            return len(self.dataloader)

    # ------------------------------------------------------------------
    # Tensor emission
    # ------------------------------------------------------------------

    def on_before_batch_transfer(self, batch, dataloader_idx):
        n = batch["X"].shape[0]
        out = {
            REGISTRY_KEYS.X_KEY: batch["X"].float(),
            REGISTRY_KEYS.PROTEIN_EXP_KEY: batch["protein_expression"].float(),
            REGISTRY_KEYS.SAMPLE_KEY: batch[self._sample_key][:, None].float(),
            REGISTRY_KEYS.INDICES_KEY: batch["ind_x"][:, None].long(),
            REGISTRY_KEYS.BATCH_KEY: (
                batch[self._batch_key][:, None].long()
                if self._batch_key is not None
                else torch.zeros((n, 1), dtype=torch.long)
            ),
            REGISTRY_KEYS.LABELS_KEY: (
                batch[self._label_key][:, None].long()
                if self._label_key is not None
                else torch.zeros((n, 1), dtype=torch.long)
            ),
        }
        # Matches the real `AnnDataLoader`: empty (unregistered) covariate
        # fields are omitted from the tensor dict entirely, not included
        # with a `None` value.
        if self._categorical_covariate_keys:
            out[REGISTRY_KEYS.CAT_COVS_KEY] = torch.stack(
                [batch[k] for k in self._categorical_covariate_keys], dim=1
            ).long()
        if self._continuous_covariate_keys:
            out[REGISTRY_KEYS.CONT_COVS_KEY] = torch.stack(
                [batch[k] for k in self._continuous_covariate_keys], dim=1
            ).float()
        return out

    # ------------------------------------------------------------------
    # Shape / encoding properties
    # ------------------------------------------------------------------

    @property
    def n_obs(self) -> int:
        return self._dataset.n_obs

    @property
    def var_names(self) -> np.ndarray:
        return self._dataset.var_joint

    @property
    def n_vars(self) -> int:
        return self._dataset.n_vars

    @property
    def n_proteins(self) -> int:
        return self._dataset.n_proteins

    @property
    def protein_names(self) -> np.ndarray:
        return np.asarray(self._dataset.protein_names)

    @property
    def n_batch(self) -> int:
        return len(self.batch_labels)

    @property
    def batch_labels(self) -> np.ndarray:
        if self._batch_key is None:
            return np.array([0])
        return np.array(list(self._dataset.encoders[self._batch_key].keys()), dtype=object)

    @property
    def n_labels(self) -> int:
        return len(self.labels)

    @property
    def labels(self) -> np.ndarray:
        if self._label_key is None:
            return np.array([0])
        return np.array(list(self._dataset.encoders[self._label_key].keys()), dtype=object)

    @property
    def n_samples(self) -> int:
        return len(self.samples)

    @property
    def samples(self) -> np.ndarray:
        return np.array(list(self._dataset.encoders[self._sample_key].keys()), dtype=object)

    @property
    def sample_keys(self) -> dict:
        return self._dataset.encoders[self._sample_key]

    @property
    def label_keys(self) -> dict | None:
        if self._label_key is None:
            return None
        return self._dataset.encoders[self._label_key]

    @property
    def _protein_batch_mask(self) -> dict | None:
        """Mirrors `fields.ProteinFieldMixin._get_batch_mask_protein_data`.

        Only computed when `batch_key` is set; keys are the *encoded integer
        batch codes*, stringified (not the original batch labels) -- this
        matches the real field exactly because `setup_anndata` registers the
        main "batch" field (which overwrites `adata.obs["_scvi_batch"]` with
        integer codes) *before* the protein field runs its batch-mask check
        against a same-named, separately-constructed field instance.
        """
        if self._batch_key is None:
            return None
        pro = self._dataset.full_protein_matrix()
        codes = self._dataset.full_encoded_column(self._batch_key)
        mask = {}
        for b in np.unique(codes):
            b_inds = np.where(codes == b)[0]
            batch_sum = pro[b_inds, :].sum(axis=0)
            all_zero = batch_sum == 0
            mask[str(b)] = ~all_zero
        if np.sum([~v for v in mask.values()]) > 0:
            return mask
        return None

    @property
    def extra_categorical_covs(self) -> dict:
        if self._categorical_covariate_keys is None:
            return {
                "data_registry": {},
                "state_registry": {},
                "summary_stats": {"n_extra_categorical_covs": 0},
            }
        mapping = {
            key: np.array(list(self._dataset.encoders[key].keys()))
            for key in self._categorical_covariate_keys
        }
        return {
            "data_registry": {"attr_key": "_scvi_extra_categorical_covs", "attr_name": "obsm"},
            "state_registry": {
                "field_keys": list(self._categorical_covariate_keys),
                "mappings": mapping,
                "n_cats_per_key": [len(mapping[key]) for key in mapping],
            },
            "summary_stats": {"n_extra_categorical_covs": len(self._categorical_covariate_keys)},
        }

    @property
    def extra_continuous_covs(self) -> dict:
        if self._continuous_covariate_keys is None:
            return {
                "data_registry": {},
                "state_registry": {},
                "summary_stats": {"n_extra_continuous_covs": 0},
            }
        return {
            "data_registry": {"attr_key": "_scvi_extra_continuous_covs", "attr_name": "obsm"},
            "state_registry": {"columns": np.array(self._continuous_covariate_keys, dtype=object)},
            "summary_stats": {"n_extra_continuous_covs": len(self._continuous_covariate_keys)},
        }

    # ------------------------------------------------------------------
    # Registry
    # ------------------------------------------------------------------

    @property
    def registry(self) -> dict:
        protein_state_registry = {"column_names": self.protein_names}
        batch_mask = self._protein_batch_mask
        if batch_mask is not None:
            protein_state_registry["protein_batch_mask"] = batch_mask

        return {
            "scvi_version": scvi.__version__,
            "model_name": self.model_name,
            "setup_args": {
                "layer": None,
                "protein_expression_obsm_key": self._protein_expression_obsm_key,
                "sample_key": self._sample_key,
                "labels_key": self._label_key,
                "protein_names_uns_key": self._protein_names_uns_key,
                "batch_key": self._batch_key,
                "panel_key": None,
                "size_factor_key": None,
                "categorical_covariate_keys": self._categorical_covariate_keys,
                "continuous_covariate_keys": self._continuous_covariate_keys,
            },
            "field_registries": {
                "X": {
                    "data_registry": {"attr_name": "X", "attr_key": None},
                    "state_registry": {
                        "n_obs": self.n_obs,
                        "n_vars": self.n_vars,
                        "column_names": self.var_names,
                    },
                    "summary_stats": {"n_vars": self.n_vars, "n_cells": self.n_obs},
                },
                "batch": {
                    "data_registry": {"attr_name": "obs", "attr_key": "_scvi_batch"},
                    "state_registry": {
                        "categorical_mapping": self.batch_labels,
                        "original_key": (
                            self._batch_key if self._batch_key is not None else "_scvi_batch"
                        ),
                    },
                    "summary_stats": {"n_batch": self.n_batch},
                },
                "labels": {
                    "data_registry": {"attr_name": "obs", "attr_key": "_scvi_labels"},
                    "state_registry": {
                        "categorical_mapping": self.labels,
                        "original_key": (
                            self._label_key if self._label_key is not None else "_scvi_labels"
                        ),
                    },
                    "summary_stats": {"n_labels": self.n_labels},
                },
                "proteins": {
                    "data_registry": {
                        "attr_name": "obsm",
                        "attr_key": self._protein_expression_obsm_key,
                    },
                    "state_registry": protein_state_registry,
                    "summary_stats": {"n_proteins": self.n_proteins},
                },
                "sample": {
                    "data_registry": {"attr_name": "obs", "attr_key": "_scvi_sample"},
                    "state_registry": {
                        "categorical_mapping": self.samples,
                        "original_key": self._sample_key,
                    },
                    "summary_stats": {"n_sample": self.n_samples},
                },
                "ind_x": {
                    "data_registry": {"attr_name": "obs", "attr_key": "_indices"},
                    "state_registry": {},
                    "summary_stats": {},
                },
                "size_factor": {
                    "data_registry": {},
                    "state_registry": {},
                    "summary_stats": {},
                },
                "extra_categorical_covs": self.extra_categorical_covs,
                "extra_continuous_covs": self.extra_continuous_covs,
            },
            "setup_method_name": "setup_datamodule",
        }
