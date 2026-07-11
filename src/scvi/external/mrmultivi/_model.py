"""MrMultiVI — MultiVI with per-donor hierarchical latent space."""

from __future__ import annotations

import logging
import warnings
from typing import TYPE_CHECKING

import numpy as np
import torch
import xarray as xr
from mudata import MuData
from tqdm import tqdm

from scvi import REGISTRY_KEYS, settings
from scvi.data import AnnDataManager, fields
from scvi.data._utils import _get_adata_minify_type, _validate_adata_dataloader_input
from scvi.model._multivi import MULTIVI
from scvi.utils import setup_anndata_dsp

from ..mrtotalvi._stats import (
    _differential_expression,
    differential_abundance as _differential_abundance,
    get_aggregated_posterior as _get_aggregated_posterior,
    get_outlier_cell_sample_pairs as _get_outlier_cell_sample_pairs,
)
from ._module import MrMultiVAE

if TYPE_CHECKING:
    from typing import Iterator, Literal, Sequence

    import numpy.typing as npt
    from torch import Tensor

logger = logging.getLogger(__name__)


class MrMultiVI(MULTIVI):
    """MultiVI with an MrVI-style hierarchical donor latent space.

    Grafts MrVI's per-sample attention residual onto MultiVI, enabling cell-
    and donor-level variability to be **jointly modelled** rather than treating
    donor identity as a nuisance covariate.

    * MULTIVAE's mixed encoder output ``u0`` is fed into a sample-conditioned
      u-encoder (:class:`~._module.MrMultiVAE`) which produces ``u ~ q_u(u0, d)``.
    * A donor-specific residual ``eps`` is then computed via attention over a
      per-donor embedding table, giving ``z = z_base + eps``.
    * The decoder is **unchanged**; ``z`` drops in with the same shape as
      MULTIVAE's ``z``.

    Parameters
    ----------
    mdata
        MuData registered via :meth:`~MrMultiVI.setup_mudata`.
    sample_key
        Key in global ``mdata.obs`` identifying the donor/sample for each cell.
    n_latent_sample
        Dimension of the per-donor embedding in the attention block.
    z_u_prior_scale
        Log-scale of the prior on the donor residual ``eps``.
        ``0.0`` → ``p(eps) = N(0, 1)``.
    learn_z_u_prior_scale
        Whether ``pz_scale`` is a learnable parameter.
    **model_kwargs
        Additional keyword arguments forwarded to :class:`~._module.MrMultiVAE`
        (and transitively to :class:`~scvi.module._multivae.MULTIVAE`).

    See Also
    --------
    :class:`~._module.MrMultiVAE`
    """

    _module_cls = MrMultiVAE

    def __init__(
        self,
        mdata: MuData,
        sample_key: str,
        n_latent_sample: int = 16,
        n_latent_u: int | None = None,
        z_u_prior: bool = True,
        z_u_prior_scale: float = 0.0,
        u_prior_scale: float = 0.0,
        u_prior_mixture: bool = True,
        u_prior_mixture_k: int = 20,
        u_prior_label_weight: float = 10.0,
        u_prior: str = "mog",
        learn_z_u_prior_scale: bool = False,
        qz_kwargs: dict | None = None,
        qu_kwargs: dict | None = None,
        use_map: bool = True,
        scale_observations: bool = False,
        **model_kwargs,
    ) -> None:
        if model_kwargs.get("latent_distribution", "normal") != "normal":
            raise ValueError(
                "MrMultiVI requires latent_distribution='normal'. "
                "Under 'ln' (softmax normalisation), the additive u→z "
                "hierarchy is mathematically invalid."
            )
        model_kwargs["latent_distribution"] = "normal"

        # MULTIVI.__init__ → creates MrMultiVAE(n_sample=0) internally.
        # Hierarchy params (n_latent_sample etc.) are NOT injected into model_kwargs here:
        # they are named args in MrMultiVI.__init__ and would cause duplicate-kwargs TypeError
        # at load time if also present in model_kwargs (_get_init_params captures both).
        # _setup_hierarchy below wires all hierarchy params correctly after super().__init__.
        super().__init__(
            mdata,
            n_latent_u=n_latent_u,
            z_u_prior=z_u_prior,
            u_prior_scale=u_prior_scale,
            u_prior_mixture=u_prior_mixture,
            u_prior_mixture_k=u_prior_mixture_k,
            u_prior_label_weight=u_prior_label_weight,
            u_prior=u_prior,
            qz_kwargs=qz_kwargs,
            qu_kwargs=qu_kwargs,
            **model_kwargs,
        )

        # summary_stats.n_sample is populated from the SAMPLE_KEY registry field
        n_sample = self.summary_stats.n_sample

        counts = (
            mdata.obs["_scvi_sample"]
            .value_counts()
            .reindex(range(n_sample), fill_value=0)
            .sort_index()
        )
        if (counts == 0).any():
            missing = counts[counts == 0].index.tolist()
            raise ValueError(
                f"Donors at index positions {missing} have no cells in the registered "
                "MuData. Remove them from sample_key or filter them from the data."
            )
        n_obs_per_sample = torch.tensor(counts.values, dtype=torch.float32)

        self.module._setup_hierarchy(
            n_sample=n_sample,
            n_latent_sample=n_latent_sample,
            z_u_prior_scale=z_u_prior_scale,
            learn_z_u_prior_scale=learn_z_u_prior_scale,
            use_map=use_map,
            scale_observations=scale_observations,
            n_obs_per_sample=n_obs_per_sample,
            n_labels=self.summary_stats.get("n_labels", 0),
        )

        self._sample_key = sample_key
        self.sample_key = sample_key
        self.sample_order = (
            self.adata_manager.get_state_registry(REGISTRY_KEYS.SAMPLE_KEY).categorical_mapping
        )
        self.label_order = (
            self.adata_manager.get_state_registry(REGISTRY_KEYS.LABELS_KEY).categorical_mapping
        )
        self.sample_info = self.adata.obs[[sample_key]].drop_duplicates().reset_index(drop=True)

        self._model_summary_string = (
            f"MrMultiVI Model\n"
            f"  n_latent: {self.module.n_latent}, n_latent_u: {self.module.n_latent_u}, "
            f"n_latent_sample: {n_latent_sample}\n"
            f"  n_sample: {n_sample}, z_u_prior: {z_u_prior}, "
            f"z_u_prior_scale: {z_u_prior_scale}\n"
            f"  u_prior_mixture: {u_prior_mixture}, "
            f"u_prior_mixture_k: {self.module.resolved_u_prior_mixture_k}"
        )
        self.init_params_ = self._get_init_params(locals())

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self, *args, accelerator: str = "auto", **kwargs) -> None:
        """Train MrMultiVI with explicit GPU/CPU handling.

        Identical to :meth:`~scvi.model.MULTIVI.train` but warns loudly when
        no CUDA device is detected instead of silently falling back to CPU.
        Pass ``accelerator='cpu'`` to suppress the warning and run on CPU.
        """
        if accelerator == "auto":
            if torch.cuda.is_available():
                accelerator = "gpu"
            else:
                warnings.warn(
                    "No CUDA device found — MrMultiVI will train on CPU. "
                    "Pass accelerator='cpu' to suppress this warning, or "
                    "run on a GPU node.",
                    UserWarning,
                    stacklevel=2,
                )
        super().train(*args, accelerator=accelerator, **kwargs)

    # ------------------------------------------------------------------
    # Data registration
    # ------------------------------------------------------------------

    @classmethod
    @setup_anndata_dsp.dedent
    def setup_mudata(
        cls,
        mdata: MuData,
        sample_key: str,
        rna_layer: str | None = None,
        atac_layer: str | None = None,
        protein_layer: str | None = None,
        batch_key: str | None = None,
        labels_key: str | None = None,
        size_factor_key: str | None = None,
        categorical_covariate_keys: list[str] | None = None,
        continuous_covariate_keys: list[str] | None = None,
        idx_layer: str | None = None,
        modalities: dict[str, str] | None = None,
        **kwargs,
    ):
        """%(summary_mdata)s.

        Parameters
        ----------
        %(param_mdata)s
        sample_key
            Key in global ``mdata.obs`` identifying the donor/sample for each cell.
            Each unique value becomes one row in the per-donor embedding table.
        rna_layer
            RNA layer key. If ``None``, uses ``.X`` of the specified modality.
        atac_layer
            ATAC layer key. If ``None``, uses ``.X`` of the specified modality.
        protein_layer
            Protein layer key. If ``None``, uses ``.X`` of the specified modality.
        %(param_batch_key)s
        labels_key
            Optional key in ``mdata.obs`` identifying labels used to condition
            the mixture-of-Gaussians prior over ``u``.
        %(param_cat_cov_keys)s
        %(param_cont_cov_keys)s
        %(param_modalities)s
        """
        setup_method_args = cls._get_setup_method_args(**locals())

        if modalities is None:
            raise ValueError("Modalities cannot be None.")
        modalities = cls._create_modalities_attr_dict(modalities, setup_method_args)

        # Canonical modality order: rna → atac → protein
        desired_order = []
        if modalities.rna_layer is not None:
            desired_order.append(modalities.rna_layer)
        if modalities.atac_layer is not None:
            desired_order.append(modalities.atac_layer)
        if modalities.protein_layer is not None:
            desired_order.append(modalities.protein_layer)

        current_order = list(mdata.mod.keys())
        needs_reorder = current_order[: len(desired_order)] != desired_order
        if needs_reorder:
            reordered_keys = desired_order + [k for k in current_order if k not in desired_order]
            backing_dict = mdata._mod
            snapshot = {k: backing_dict[k] for k in reordered_keys}
            backing_dict.clear()
            backing_dict.update(snapshot)
            mdata.update()

        mdata.obs["_indices"] = np.arange(mdata.n_obs)

        batch_field = fields.MuDataCategoricalObsField(
            REGISTRY_KEYS.BATCH_KEY,
            batch_key,
            mod_key=modalities.batch_key,
        )
        mudata_fields = [
            batch_field,
            fields.MuDataCategoricalObsField(
                REGISTRY_KEYS.LABELS_KEY,
                labels_key,
                mod_key=modalities.labels_key,
            ),
            fields.MuDataNumericalJointObsField(
                REGISTRY_KEYS.SIZE_FACTOR_KEY,
                size_factor_key,
                mod_key=None,
                required=False,
            ),
            fields.MuDataCategoricalJointObsField(
                REGISTRY_KEYS.CAT_COVS_KEY,
                categorical_covariate_keys,
                mod_key=modalities.categorical_covariate_keys,
            ),
            fields.MuDataNumericalJointObsField(
                REGISTRY_KEYS.CONT_COVS_KEY,
                continuous_covariate_keys,
                mod_key=modalities.continuous_covariate_keys,
            ),
            fields.MuDataNumericalObsField(
                REGISTRY_KEYS.INDICES_KEY,
                "_indices",
                mod_key=modalities.idx_layer,
                required=False,
            ),
            # MrMultiVI: donor/sample axis in global mdata.obs
            fields.MuDataCategoricalObsField(
                REGISTRY_KEYS.SAMPLE_KEY,
                sample_key,
                mod_key=None,
            ),
        ]
        if modalities.rna_layer is not None:
            mudata_fields.append(
                fields.MuDataLayerField(
                    REGISTRY_KEYS.X_KEY,
                    rna_layer,
                    mod_key=modalities.rna_layer,
                    is_count_data=True,
                    mod_required=True,
                )
            )
        if modalities.atac_layer is not None:
            mudata_fields.append(
                fields.MuDataLayerField(
                    REGISTRY_KEYS.ATAC_X_KEY,
                    atac_layer,
                    mod_key=modalities.atac_layer,
                    is_count_data=True,
                    mod_required=True,
                )
            )
        if modalities.protein_layer is not None:
            mudata_fields.append(
                fields.MuDataProteinLayerField(
                    REGISTRY_KEYS.PROTEIN_EXP_KEY,
                    protein_layer,
                    mod_key=modalities.protein_layer,
                    use_batch_mask=True,
                    batch_field=batch_field,
                    is_count_data=True,
                    mod_required=True,
                )
            )
        mdata_minify_type = _get_adata_minify_type(mdata)
        if mdata_minify_type is not None:
            mudata_fields += cls._get_fields_for_mudata_minification(mdata_minify_type)

        adata_manager = AnnDataManager(fields=mudata_fields, setup_method_args=setup_method_args)
        adata_manager.register_fields(mdata, **kwargs)
        cls.register_manager(adata_manager)

    # ------------------------------------------------------------------
    # u-space statistical APIs
    # ------------------------------------------------------------------

    def get_aggregated_posterior(
        self,
        adata=None,
        sample: str | int | None = None,
        indices: Sequence[int] | None = None,
        batch_size: int = 256,
    ):
        """Compute the aggregated posterior mixture over ``u``."""
        return _get_aggregated_posterior(
            self,
            adata=adata,
            sample_key=self.sample_key,
            sample=sample,
            indices=indices,
            batch_size=batch_size,
        )

    def differential_abundance(
        self,
        adata=None,
        sample_cov_keys: list[str] | None = None,
        sample_subset: list[str] | None = None,
        compute_log_enrichment: bool = False,
        omit_original_sample: bool = True,
        donor_key: str | None = None,
        batch_size: int = 128,
    ) -> xr.Dataset:
        """Compute MrVI-style differential abundance log probabilities over ``u``.

        Parameters
        ----------
        adata
            MuData to compute DA on.  Defaults to the training data.
        sample_cov_keys
            Sample-level covariate column names for grouping.
        sample_subset
            Restrict to these sample names only.
        compute_log_enrichment
            If ``True``, also compute log enrichment vs. the complementary group.
        omit_original_sample
            Exclude a cell's own sample when computing the aggregated log-prob.
        donor_key
            Column in ``mdata.obs`` identifying the donor.  When set, per-donor
            log_probs are mean-centred before covariate aggregation, blocking the
            donor effect for repeated-measures designs.
        batch_size
            Dataloader batch size.
        """
        return _differential_abundance(
            self,
            adata=adata,
            sample_key=self.sample_key,
            sample_cov_keys=sample_cov_keys,
            sample_subset=sample_subset,
            compute_log_enrichment=compute_log_enrichment,
            omit_original_sample=omit_original_sample,
            donor_key=donor_key,
            batch_size=batch_size,
        )

    def get_outlier_cell_sample_pairs(
        self,
        adata=None,
        subsample_size: int | None = 5_000,
        quantile_threshold: float = 0.05,
        admissibility_threshold: float = 0.0,
        batch_size: int = 256,
    ) -> xr.Dataset:
        """Compute admissible cell-sample pairs using aggregated posteriors over ``u``."""
        return _get_outlier_cell_sample_pairs(
            self,
            adata=adata,
            sample_key=self.sample_key,
            subsample_size=subsample_size,
            quantile_threshold=quantile_threshold,
            admissibility_threshold=admissibility_threshold,
            batch_size=batch_size,
        )

    def differential_expression(
        self,
        adata=None,
        sample_cov_keys: list[str] | None = None,
        sample_subset: list[str] | None = None,
        indices=None,
        batch_size: int = 128,
        normalize_design_matrix: bool = True,
        mc_samples: int = 50,
        filter_inadmissible_samples: bool = False,
        store_lfc: bool = False,
        donor_key: str | None = None,
        delta: float | None = 0.3,
        lambd: float = 0.0,
        store_baseline: bool = False,
        eps_lfc: float = 1e-6,
        use_vmap: bool = False,
        **filter_samples_kwargs,
    ) -> xr.Dataset:
        """MrVI-style latent-space differential expression.

        Fits a per-cell weighted least-squares linear model on the sample-
        specific residual ``eps_d = z_d - z_base`` (the donor latent shift),
        following Boyeau et al. (2023).

        Parameters
        ----------
        adata
            MuData to compute DE on.  Defaults to the training data.
        sample_cov_keys
            Sample-level covariate column names in ``mdata.obs``.
        sample_subset
            Restrict DE to these sample names only.
        indices
            Cell indices to process (default: all cells).
        batch_size
            Dataloader batch size.
        normalize_design_matrix
            Min-max normalize each covariate column to [0, 1].
        mc_samples
            Monte-Carlo draws from ``q(u)``; ``1`` uses the posterior mean.
        filter_inadmissible_samples
            Weight out outlier cell-sample pairs via aggregated-posterior
            admissibility scores.
        store_lfc
            If ``True``, compute gene/protein log2-fold-changes (LFC) and store
            ``lfc``, ``lfc_std``, and optionally ``pde`` / ``baseline_expression``
            in the returned dataset.  Feature axis is split into ``gene`` and
            ``protein`` coordinates by this wrapper (RNA-only models get only
            ``gene`` coords).
        donor_key
            Column in ``mdata.obs`` identifying the donor.  Adds donor dummy
            columns as nuisance covariates (repeated-measures approximation).
        delta
            Effect-size threshold for PDE (``P(|lfc| >= delta)``).  Used only
            when ``store_lfc=True``.  Pass ``None`` to skip PDE.
        lambd
            L2 regularisation for ``X^T M X`` inversion.
        store_baseline
            When ``store_lfc=True``, also store the batch-marginalized baseline
            expression (null decode) as ``baseline_expression``.
        eps_lfc
            Small offset added before log2 to avoid log(0).  Default ``1e-6``.
        use_vmap
            Reserved; currently ignored (loop path is always used).
        **filter_samples_kwargs
            Forwarded to :meth:`get_outlier_cell_sample_pairs`.
        """
        if getattr(self, "n_regions", 0) > 0:
            raise NotImplementedError(
                "MrMultiVI differential_expression is only defined for RNA/protein "
                "outputs. ATAC-containing models should use a future "
                "differential_accessibility API instead."
            )
        ds = _differential_expression(
            self,
            adata=adata,
            sample_cov_keys=list(sample_cov_keys) if sample_cov_keys else [],
            sample_subset=sample_subset,
            indices=indices,
            batch_size=batch_size,
            normalize_design_matrix=normalize_design_matrix,
            mc_samples=mc_samples,
            filter_inadmissible_samples=filter_inadmissible_samples,
            store_lfc=store_lfc,
            donor_key=donor_key,
            delta=delta,
            lambd=lambd,
            store_baseline=store_baseline,
            eps_lfc=eps_lfc,
            use_vmap=use_vmap,
            **filter_samples_kwargs,
        )
        if store_lfc and "feature" in ds.coords:
            n_genes = self.summary_stats.n_vars
            n_proteins = self.module.n_input_proteins
            if n_proteins > 0:
                ds = ds.assign_coords(
                    feature=(
                        ["gene"] * n_genes + ["protein"] * n_proteins
                    )
                )
            else:
                ds = ds.assign_coords(feature=["gene"] * n_genes)
        return ds

    def differential_accessibility(self, *args, **kwargs) -> xr.Dataset:
        """ATAC differential accessibility is intentionally separate from DE."""
        raise NotImplementedError(
            "MrMultiVI differential_accessibility is not implemented yet; ATAC effects "
            "are intentionally not mixed into differential_expression."
        )

    # ------------------------------------------------------------------
    # Latent representation
    # ------------------------------------------------------------------

    def get_latent_representation(
        self,
        adata=None,
        modality: Literal["joint", "expression", "accessibility"] = "joint",
        indices: Sequence[int] | None = None,
        give_mean: bool = True,
        batch_size: int | None = None,
        return_dist: bool = False,
        dataloader: Iterator[dict[str, Tensor | None]] | None = None,
        give_z: bool = True,
    ) -> npt.NDArray:
        """Return the latent representation for each cell.

        Parameters
        ----------
        adata
            MuData object. Defaults to the object used to initialise the model.
        modality
            ``"joint"`` (default), ``"expression"``, or ``"accessibility"``.
            Non-joint modalities return per-modality latents (no hierarchy).
        indices
            Indices of cells in adata to use.
        give_mean
            If ``True``, uses the posterior mean of ``u`` as the query to
            :class:`~._module.MrMultiVAE`.  When ``give_z=True`` and
            ``give_mean=True``, returns ``z_mean = z_base(u_mean) + eps(u_mean, d)``.
        batch_size
            Minibatch size.
        return_dist
            If ``True``, returns ``(qz_m, qz_v)`` for the joint modality
            (distribution over ``u``, not ``z``). Delegates to super.
        dataloader
            Custom dataloader; mutually exclusive with ``adata``.
        give_z
            If ``True`` (default), returns the sample-aware hierarchical
            ``z = z_base + eps`` using each cell's actual donor index.
            If ``False``, returns the sample-unaware base ``u``.

        Returns
        -------
        Array of shape ``(n_obs, n_latent)``.

        Notes
        -----
        For ``modality != "joint"`` or ``return_dist=True``, delegates to
        :meth:`~scvi.model.MULTIVI.get_latent_representation` (no ``give_z``).

        Do **not** delegate the ``give_z=True, give_mean=True`` joint path to
        super — MULTIVI returns ``qz_m`` (mean of ``u``), silently discarding
        the donor residual.  Both branches are implemented manually.
        """
        if modality != "joint" or return_dist:
            return super().get_latent_representation(
                adata=adata,
                modality=modality,
                indices=indices,
                give_mean=give_mean,
                batch_size=batch_size,
                return_dist=return_dist,
                dataloader=dataloader,
            )

        if not self.is_trained_:
            raise RuntimeError("Please train the model first.")
        _validate_adata_dataloader_input(self, adata, dataloader)
        self._check_adata_modality_weights(adata)

        if dataloader is None:
            adata = self._validate_anndata(adata)
            scdl = self._make_data_loader(adata=adata, indices=indices, batch_size=batch_size)
        else:
            scdl = dataloader

        results = []
        with torch.inference_mode():
            for tensors in scdl:
                inf_inputs = self.module._get_inference_input(tensors)
                out = self.module.inference(**inf_inputs)

                if not give_z:
                    # Return u (sample-conditioned base: qu.loc or qu.rsample())
                    rep = out["qz_m"] if give_mean else out["u"]
                else:
                    # give_z=True: return z = z_base + eps using the real donor index.
                    # give_mean=True: re-run qz with u_mean so the returned z uses
                    #   the posterior mean of u (not a reparameterised sample).
                    # give_mean=False: out["z"] is already z_base + eps from inference.
                    if give_mean:
                        sample_index = inf_inputs["sample_index"]
                        u_mean = out["qz_m"]
                        z_base, eps, _ = self.module.qz(u_mean, sample_index)
                        rep = z_base + eps
                    else:
                        rep = out["z"]

                results.append(rep.cpu().numpy())

        return np.concatenate(results, axis=0)

    # ------------------------------------------------------------------
    # Counterfactual representations (internal)
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def _compute_cf_representations(
        self,
        adata,
        indices,
        batch_size: int,
        use_mean: bool = True,
    ) -> tuple[npt.NDArray, list[str]]:
        """Compute counterfactual z for every cell × donor combination.

        For each cell, runs the encoder once to get ``u``, then loops over all
        ``n_sample`` donors.  For each donor ``d``:

        .. code-block::

            cf_sample = d * ones(n_cells)
            z_base, eps = module.qz(u, cf_sample)
            z[cell, d, :] = z_base + eps

        Returns
        -------
        reps : np.ndarray
            Shape ``(n_cells, n_sample, n_latent)``.
        cell_names : list[str]
        """
        scdl = self._make_data_loader(adata=adata, indices=indices, batch_size=batch_size)
        n_sample = self.summary_stats.n_sample

        all_reps: list[npt.NDArray] = []
        all_cell_names: list[str] = []

        for tensors in tqdm(scdl, desc="Counterfactual representations"):
            cell_idx = tensors[REGISTRY_KEYS.INDICES_KEY].long().flatten()
            all_cell_names.extend(adata.obs_names[cell_idx.numpy()].tolist())

            inf_inputs = self.module._get_inference_input(tensors)
            base_out = self.module.inference(**inf_inputs)

            # u: sample-conditioned base posterior mean (qu.loc) or reparameterized sample
            u = base_out["qz_m"] if use_mean else base_out["u"]

            n_cells = u.shape[0]
            dev = u.device

            cf_zs = []
            for d in range(n_sample):
                cf_sample = torch.full((n_cells, 1), d, dtype=torch.long, device=dev)
                z_base, eps, _ = self.module.qz(u, cf_sample)
                cf_zs.append((z_base + eps).detach().cpu())

            # (n_sample, n_cells, n_latent) → (n_cells, n_sample, n_latent)
            batch_reps = torch.stack(cf_zs, dim=0).permute(1, 0, 2).numpy()
            all_reps.append(batch_reps)

        return np.concatenate(all_reps, axis=0), all_cell_names

    # ------------------------------------------------------------------
    # Public counterfactual API
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def get_local_sample_representation(
        self,
        adata=None,
        indices=None,
        batch_size: int = 256,
        use_mean: bool = True,
    ) -> xr.DataArray:
        """Compute the local per-donor latent representation.

        For each cell, returns a ``(n_sample, n_latent)`` matrix of
        counterfactual ``z`` values — one per registered donor.

        Returns
        -------
        :class:`~xarray.DataArray` of shape ``(n_cell, n_sample, n_latent)``.
        """
        self._check_if_trained(warn=False)
        adata = self._validate_anndata(adata)

        reps, cell_names = self._compute_cf_representations(
            adata, indices, batch_size, use_mean=use_mean
        )
        return xr.DataArray(
            reps,
            dims=["cell_name", "sample", "latent_dim"],
            coords={"cell_name": cell_names, "sample": self.sample_order},
            name="sample_representations",
        )

    @torch.inference_mode()
    def get_local_sample_distances(
        self,
        adata=None,
        indices=None,
        batch_size: int = 256,
        use_mean: bool = True,
        norm: Literal["l2", "l1"] = "l2",
    ) -> xr.DataArray:
        """Compute cell-specific pairwise donor distance matrices.

        For each cell, computes a symmetric ``(n_sample, n_sample)`` distance
        matrix over the counterfactual donor representations.

        Returns
        -------
        :class:`~xarray.DataArray` of shape ``(n_cell, n_sample, n_sample)``.
        """
        self._check_if_trained(warn=False)
        adata = self._validate_anndata(adata)

        reps, cell_names = self._compute_cf_representations(
            adata, indices, batch_size, use_mean=use_mean
        )
        reps_t = torch.tensor(reps)  # (n_cells, n_sample, n_latent)

        def _pairwise(rep: torch.Tensor) -> torch.Tensor:
            delta = rep.unsqueeze(0) - rep.unsqueeze(1)  # (n_s, n_s, n_latent)
            if norm == "l2":
                return torch.sqrt((delta**2).sum(-1))
            elif norm == "l1":
                return delta.abs().sum(-1)
            else:
                raise ValueError(f"Unsupported norm '{norm}'. Choose 'l2' or 'l1'.")

        dists = torch.vmap(_pairwise)(reps_t).numpy()

        return xr.DataArray(
            dists,
            dims=["cell_name", "sample_x", "sample_y"],
            coords={
                "cell_name": cell_names,
                "sample_x": self.sample_order,
                "sample_y": self.sample_order,
            },
            name="sample_distances",
        )
