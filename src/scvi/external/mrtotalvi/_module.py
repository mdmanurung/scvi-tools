"""MrTotalVAE — TotalVI with MrVI-style hierarchical donor latent space."""

from __future__ import annotations

import inspect

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, kl_divergence

from scvi import REGISTRY_KEYS
from scvi.module._constants import MODULE_KEYS
from scvi.module._totalvae import TOTALVAE
from scvi.module.base import EmbeddingModuleMixin, LossOutput, auto_move_data
from scvi.nn import DecoderTOTALVI

from ._components import (
    BatchEmbeddingDecoderAdapter,
    EncoderUZ,
    EncoderXU_TotalVI,
    init_u_prior,
)
from ._components import (
    build_u_prior as _build_u_prior,
)
from ._components import (
    kl_u as _kl_u,
)


class MrTotalVAE(EmbeddingModuleMixin, TOTALVAE):
    r"""TotalVI VAE with an MrVI-style u→z hierarchical latent space.

    Grafts MrVI's hierarchical design onto TotalVI as faithfully as the
    multimodal input allows. A mode-selectable u-encoder
    (:class:`~.EncoderXU_TotalVI`) replaces TotalVI's stock encoder for the
    base representation ``u``; a donor-specific residual ``eps`` is computed
    by attending over a per-sample embedding table via :class:`~.EncoderUZ`.
    The legacy default gives:

    .. math::
        u \\sim q_u(x_{\\text{rna}},\\, x_{\\text{prot}},\\, d)
        \\quad\\text{(sample-conditioned)}

    .. math::
        z = z_{\\text{base}}(u) + \\varepsilon,\\quad
        \\varepsilon \\sim \\text{AttentionBlock}(u,\\; e_d)

    Because ``n_latent_u`` is left at its default (isomorphic), ``z_base = u``
    and the decoder input dimension is identical to stock TotalVI, so the hierarchy
    itself needs no decoder changes. Setting ``batch_representation="embedding"`` is
    separate and does change the decoder: it is rebuilt without the batch category,
    widened by the embedding dimension, and wrapped in
    :class:`~._components.BatchEmbeddingDecoderAdapter`. See that parameter.

    Two-level KL loss:

    * ``kl_u = KL(q_u \\| p_u)`` — from the sample-conditioned u-encoder, where ``p_u``
      is by default a learned mixture-of-Gaussians prior (``u_prior_mixture=True``).
    * ``kl_z = -\\log p(\\varepsilon) = -\\log N(0, \\exp(\\text{pz\\_scale}))`` — eps residual.

    Parameters
    ----------
    n_input_genes
        Number of input genes.
    n_input_proteins
        Number of input proteins.
    n_sample
        Number of donors/samples.  Pass ``0`` as a placeholder; call
        :meth:`_setup_hierarchy` before training.
    n_latent_sample
        Dimension of the per-sample embedding in :class:`~.EncoderUZ`.
    z_u_prior_scale
        Log-scale of the prior on ``eps``.  ``0.0`` → ``p(eps) = N(0,1)``.
    learn_z_u_prior_scale
        If ``True``, ``pz_scale`` is a learnable :class:`~torch.nn.Parameter`;
        otherwise it is a fixed :func:`~torch.Tensor.register_buffer`.
    hierarchy_mode
        ``"legacy"`` preserves historical execution. ``"centered_v2"``
        centers raw residuals over the full registered sample universe.
    u_encoder_mode
        ``"sample_conditioned"`` preserves historical execution;
        ``"sample_blind"`` bypasses biological sample-conditioning parameters.
    batch_representation
        ``"one-hot"`` (default) preserves historical execution: batch is one-hot encoded
        into the u-encoder input and into every decoder :class:`~scvi.nn.FCLayers` block.
        ``"embedding"`` replaces both with a single learned embedding table, so the input
        width grows by ``batch_embedding_kwargs["embedding_dim"]`` instead of by
        ``n_batch``. Useful when the number of batches is large. Per-batch *parameter
        tables* — gene/protein dispersion, ``log_per_batch_efficiency``, the protein
        background prior and the library-size priors — stay one-hot indexed either way,
        matching :class:`~scvi.module.VAE`.
    batch_embedding_kwargs
        Keyword arguments passed to :class:`~scvi.nn.Embedding` when
        ``batch_representation="embedding"``, e.g. ``{"embedding_dim": 5}``.
    **kwargs
        All remaining keyword arguments forwarded verbatim to
        :class:`~scvi.module._totalvae.TOTALVAE`.

    Notes
    -----
    ``latent_distribution`` is **forced** to ``"normal"``.  Under ``"ln"``
    (softmax normalisation), ``u`` is simplex-constrained and the additive
    residual hierarchy is mathematically invalid.
    """

    def __init__(
        self,
        n_input_genes: int,
        n_input_proteins: int,
        n_sample: int = 0,
        n_latent_sample: int = 16,
        z_u_prior_scale: float = 0.0,
        learn_z_u_prior_scale: bool = False,
        n_latent_u: int | None = None,
        z_u_prior: bool = True,
        u_prior_scale: float = 0.0,
        u_prior_mixture: bool = True,
        u_prior_mixture_k: int = 20,
        u_prior_label_weight: float = 10.0,
        u_prior: str = "mog",
        qz_kwargs: dict | None = None,
        qu_kwargs: dict | None = None,
        use_map: bool = True,
        hierarchy_mode: str = "legacy",
        u_encoder_mode: str = "sample_conditioned",
        scale_observations: bool = False,
        kl_u_weight: float = 1.0,
        kl_z_weight: float = 1.0,
        batch_representation: str = "one-hot",
        batch_embedding_kwargs: dict | None = None,
        **kwargs,
    ) -> None:
        if kwargs.get("latent_distribution", "normal") != "normal":
            raise ValueError(
                "MrTotalVAE requires latent_distribution='normal'. "
                "Under 'ln' (softmax), u is simplex-constrained and the "
                "additive residual hierarchy is invalid."
            )
        kwargs["latent_distribution"] = "normal"

        n_continuous_cov = kwargs.get("n_continuous_cov", 0)
        n_cats_per_cov = kwargs.get("n_cats_per_cov", None)
        encode_covariates = kwargs.get("encode_covariates", False)

        super().__init__(n_input_genes, n_input_proteins, **kwargs)

        # Store hyperparams so _setup_hierarchy can use them if called deferred
        self._n_sample = n_sample
        self._n_latent_sample = n_latent_sample
        self._z_u_prior_scale = z_u_prior_scale
        self._learn_z_u_prior_scale = learn_z_u_prior_scale
        self._n_latent_u_requested = n_latent_u
        self.z_u_prior = bool(z_u_prior)
        self.u_prior_scale_value = float(u_prior_scale)
        self.u_prior_mixture = bool(u_prior_mixture)
        self.u_prior_mixture_k = int(u_prior_mixture_k)
        self.u_prior_label_weight = float(u_prior_label_weight)
        self.u_prior_type = u_prior
        self.qz_kwargs = qz_kwargs or {}
        self.qu_kwargs = qu_kwargs or {}
        self._use_map = use_map
        self.hierarchy_mode = hierarchy_mode
        self.u_encoder_mode = u_encoder_mode
        self._scale_observations = scale_observations
        # Per-term KL weights: static scalars applied as kl_u_weight*kl_u + kl_z_weight*kl_z
        # before the global kl_weight annealing. Defaults (1.0, 1.0) reproduce prior behavior.
        # A separate per-term annealing schedule would require training-plan changes.
        self.kl_u_weight = float(kl_u_weight)
        self.kl_z_weight = float(kl_z_weight)
        self.n_continuous_cov = int(n_continuous_cov)
        self.n_cats_per_cov = list(n_cats_per_cov or [])
        self.encode_covariates = bool(encode_covariates)

        self.batch_representation = batch_representation
        self._batch_dim: int | None = None
        if batch_representation == "embedding":
            self.init_embedding(
                REGISTRY_KEYS.BATCH_KEY, self.n_batch, **(batch_embedding_kwargs or {})
            )
            self._batch_dim = self.get_embedding(REGISTRY_KEYS.BATCH_KEY).embedding_dim
            self._rebuild_decoder_for_batch_embedding(self._batch_dim, kwargs)
        elif batch_representation != "one-hot":
            raise ValueError("`batch_representation` must be one of 'one-hot', 'embedding'.")

        if n_sample > 0:
            self._setup_hierarchy(
                n_sample,
                n_latent_sample,
                z_u_prior_scale,
                learn_z_u_prior_scale,
            )

    # ------------------------------------------------------------------
    # Batch representation
    # ------------------------------------------------------------------

    def _rebuild_decoder_for_batch_embedding(self, batch_dim: int, kwargs: dict) -> None:
        """Replace the inherited decoder with one that takes a batch embedding.

        :class:`~scvi.module.TOTALVAE` builds its decoder with the batch as a one-hot
        category. Under ``batch_representation="embedding"`` the batch category is dropped
        and the decoder input is widened by ``batch_dim`` instead, then wrapped in
        :class:`~._components.BatchEmbeddingDecoderAdapter` so the call site in
        :meth:`~scvi.module.TOTALVAE.generative` is unchanged.

        Only the decoder's *conditioning* moves to the embedding. The per-batch parameter
        tables — gene/protein dispersion, ``log_per_batch_efficiency``, the protein
        background prior and the library-size priors — remain indexed by one-hot batch,
        matching :class:`~scvi.module.VAE`, since those are lookup tables rather than
        network inputs.

        Defaults are read from :meth:`~scvi.module.TOTALVAE.__init__`'s own signature
        rather than restated here, so this stays correct if the parent's defaults change.
        """
        signature = inspect.signature(TOTALVAE.__init__)

        def _arg(name: str):
            if name in kwargs:
                return kwargs[name]
            return signature.parameters[name].default

        n_cats_per_cov = _arg("n_cats_per_cov")
        use_batch_norm = _arg("use_batch_norm")
        use_layer_norm = _arg("use_layer_norm")

        decoder = DecoderTOTALVI(
            _arg("n_latent") + _arg("n_continuous_cov") + batch_dim,
            self.n_input_genes,
            self.n_input_proteins,
            n_layers=_arg("n_layers_decoder"),
            n_cat_list=list([] if n_cats_per_cov is None else n_cats_per_cov),
            n_hidden=_arg("n_hidden"),
            dropout_rate=_arg("dropout_rate_decoder"),
            use_batch_norm=use_batch_norm in ("decoder", "both"),
            use_layer_norm=use_layer_norm in ("decoder", "both"),
            scale_activation="softplus" if _arg("use_size_factor_key") else "softmax",
            **(_arg("extra_decoder_kwargs") or {}),
        )
        self.decoder = BatchEmbeddingDecoderAdapter(
            decoder,
            lambda batch_index: self.compute_embedding(REGISTRY_KEYS.BATCH_KEY, batch_index),
        )

    def _batch_representation_for(self, batch_index: torch.Tensor | None) -> torch.Tensor | None:
        """Embedding for ``batch_index``, or ``None`` when batch is one-hot encoded."""
        if getattr(self, "_batch_dim", None) is None or batch_index is None:
            return None
        return self.compute_embedding(REGISTRY_KEYS.BATCH_KEY, batch_index)

    # ------------------------------------------------------------------
    # Hierarchy setup
    # ------------------------------------------------------------------

    def _setup_hierarchy(
        self,
        n_sample: int,
        n_latent_sample: int | None = None,
        z_u_prior_scale: float | None = None,
        learn_z_u_prior_scale: bool | None = None,
        use_map: bool | None = None,
        hierarchy_mode: str | None = None,
        u_encoder_mode: str | None = None,
        scale_observations: bool | None = None,
        n_obs_per_sample: torch.Tensor | None = None,
        n_labels: int | None = None,
        prior_centroids: torch.Tensor | None = None,
        freeze_prior_after_init: bool = False,
    ) -> None:
        """Build / replace the u→z hierarchy after the base TOTALVAE is initialised.

        Safe to call from :meth:`~MrTotalVI.__init__` after ``super().__init__``,
        e.g. once the registry summary stats are available.

        Parameters
        ----------
        n_sample
            Total number of donors registered in the AnnData.
        n_latent_sample, z_u_prior_scale, learn_z_u_prior_scale
            Override the values stored in ``__init__``; ``None`` → use stored value.
        use_map
            If ``False``, treat eps as stochastic (split attention output into mean and
            log-scale; reparameterise).  ``None`` → use value from ``__init__``.
        scale_observations
            If ``True``, weight each cell's ELBO contribution by
            ``1 / n_cells_in_that_sample`` so high-cell-count donors do not dominate.
            ``None`` → use value from ``__init__``.
        n_obs_per_sample
            Integer tensor of shape ``(n_sample,)`` with per-donor cell counts.
            Required (and used only) when ``scale_observations=True``.
        prior_centroids
            Optional ``(K, dim)`` tensor of cluster centroids for data-driven prior
            initialization. See :func:`init_u_prior` for semantics.
        freeze_prior_after_init
            Passed through to :func:`init_u_prior`. See its docstring for semantics.
        """
        if n_latent_sample is None:
            n_latent_sample = self._n_latent_sample
        if z_u_prior_scale is None:
            z_u_prior_scale = self._z_u_prior_scale
        if learn_z_u_prior_scale is None:
            learn_z_u_prior_scale = self._learn_z_u_prior_scale
        if use_map is None:
            use_map = self._use_map
        if hierarchy_mode is None:
            hierarchy_mode = self.hierarchy_mode
        if u_encoder_mode is None:
            u_encoder_mode = self.u_encoder_mode
        if scale_observations is None:
            scale_observations = self._scale_observations

        if hierarchy_mode not in {"legacy", "centered_v2"}:
            raise ValueError("hierarchy_mode must be one of {'legacy', 'centered_v2'}.")
        if u_encoder_mode not in {"sample_conditioned", "sample_blind"}:
            raise ValueError(
                "u_encoder_mode must be one of {'sample_conditioned', 'sample_blind'}."
            )
        if hierarchy_mode == "centered_v2" and not use_map:
            raise ValueError("hierarchy_mode='centered_v2' requires use_map=True.")
        if hierarchy_mode == "centered_v2" and not self.z_u_prior:
            raise ValueError("hierarchy_mode='centered_v2' requires z_u_prior=True.")

        self._n_sample = n_sample
        self._use_map = use_map
        self.hierarchy_mode = hierarchy_mode
        self.u_encoder_mode = u_encoder_mode
        self._scale_observations = scale_observations
        if n_labels is not None:
            self.n_labels = int(n_labels)

        n_latent_u = (
            self.n_latent
            if self._n_latent_u_requested is None
            else int(self._n_latent_u_requested)
        )

        # Mode-selectable u-encoder: the legacy default is sample-conditioned.
        # Explicitly registered technical covariates remain available when
        # encode_covariates=True, including in sample-blind mode.
        self.qu = EncoderXU_TotalVI(
            n_input_genes=self.n_input_genes,
            n_input_proteins=self.n_input_proteins,
            n_latent=n_latent_u,
            n_sample=n_sample,
            u_encoder_mode=u_encoder_mode,
            n_batch=self.n_batch,
            n_continuous_cov=self.n_continuous_cov,
            n_cats_per_cov=self.n_cats_per_cov,
            encode_covariates=self.encode_covariates,
            batch_dim=getattr(self, "_batch_dim", None),
            **self.qu_kwargs,
        )

        # Isomorphic dims (n_latent_u=None) → z_base = u, decoder unchanged
        self.qz = EncoderUZ(
            n_latent=self.n_latent,
            n_sample=n_sample,
            n_latent_u=None if n_latent_u == self.n_latent else n_latent_u,
            n_latent_sample=n_latent_sample,
            use_map=use_map,
            **self.qz_kwargs,
        )
        init_u_prior(
            self,
            n_latent_u=n_latent_u,
            n_labels=getattr(self, "n_labels", 0),
            u_prior_scale=self.u_prior_scale_value,
            u_prior_mixture=self.u_prior_mixture,
            u_prior_mixture_k=self.u_prior_mixture_k,
            u_prior_label_weight=self.u_prior_label_weight,
            u_prior_type=getattr(self, "u_prior_type", "mog"),
            u_vamp_pseudo_dim=self.n_input_genes + self.n_input_proteins,
            prior_centroids=prior_centroids,
            freeze_prior_after_init=freeze_prior_after_init,
        )

        if learn_z_u_prior_scale:
            self.pz_scale = nn.Parameter(torch.zeros(self.n_latent))
        else:
            self.register_buffer(
                "pz_scale",
                torch.full((self.n_latent,), float(z_u_prior_scale)),
            )

        # Persistent per-sample cell counts restore observation reweighting.
        if scale_observations and n_obs_per_sample is not None:
            self.register_buffer("n_obs_per_sample", n_obs_per_sample.float(), persistent=True)
        else:
            self.n_obs_per_sample = None

    def _vamp_component_dist(self) -> Normal:
        """Run VampPrior pseudoinputs through qu to get K component distributions.

        Pseudoinputs are in raw (rna+protein) input space; Softplus constrains
        them to ≥ 0 so that log1p inside EncoderXU_TotalVI stays well-defined.
        Reference donor (index 0) is used — the learnable pseudoinputs absorb
        the manifold variation; sample conditioning is deliberately excluded
        from the prior.
        """
        K = self.resolved_u_prior_mixture_k
        pseudo = F.softplus(self.u_vamp_pseudo)  # (K, n_input_genes + n_input_proteins)
        x_p = pseudo[:, : self.n_input_genes]
        y_p = pseudo[:, self.n_input_genes :]
        sample_idx = torch.zeros(K, 1, device=pseudo.device, dtype=torch.long)
        batch_idx = (
            torch.zeros(K, 1, device=pseudo.device, dtype=torch.long)
            if self.encode_covariates
            else None
        )
        return self.qu(
            x_p,
            y_p,
            sample_idx,
            batch_index=batch_idx,
            batch_rep=self._batch_representation_for(batch_idx),
        )

    # ------------------------------------------------------------------
    # DataLoader → inference plumbing
    # ------------------------------------------------------------------

    def _get_inference_input(
        self, tensors: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Extend TotalVI's inference input with the sample index.

        The ``SAMPLE_KEY`` tensor is added unconditionally; it is present in
        every DataLoader batch once :meth:`MrTotalVI.setup_anndata` has
        registered it.  Minified-mode inference is not supported for
        MrTotalVAE (the hierarchy requires the full encoder path).
        """
        base = super()._get_inference_input(tensors)
        base["sample_index"] = tensors[REGISTRY_KEYS.SAMPLE_KEY]
        return base

    def _all_sample_residuals(
        self,
        u: torch.Tensor,
        target_chunk_size: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate and center raw residuals over every registered sample."""
        n_sample = int(self._n_sample)
        if target_chunk_size is None:
            target_chunk_size = n_sample
        if (
            isinstance(target_chunk_size, bool)
            or not isinstance(target_chunk_size, int)
            or target_chunk_size < 1
        ):
            raise ValueError("target_chunk_size must be a positive integer or None.")

        sample_dim = 1 if u.ndim == 2 else 2
        raw_chunks: list[torch.Tensor] = []
        z_base: torch.Tensor | None = None
        for start in range(0, n_sample, target_chunk_size):
            stop = min(start + target_chunk_size, n_sample)
            targets = torch.arange(start, stop, device=u.device, dtype=torch.long)
            n_targets = targets.numel()
            if u.ndim == 2:
                n_cells, n_latent_u = u.shape
                expanded_u = (
                    u.unsqueeze(1)
                    .expand(n_cells, n_targets, n_latent_u)
                    .reshape(n_cells * n_targets, n_latent_u)
                )
                expanded_targets = (
                    targets.unsqueeze(0)
                    .expand(n_cells, n_targets)
                    .reshape(n_cells * n_targets, 1)
                )
                chunk_z_base, chunk_raw, _ = self.qz(expanded_u, expanded_targets)
                chunk_z_base = chunk_z_base.reshape(n_cells, n_targets, self.n_latent)
                chunk_raw = chunk_raw.reshape(n_cells, n_targets, self.n_latent)
            elif u.ndim == 3:
                n_draws, n_cells, n_latent_u = u.shape
                expanded_u = (
                    u.unsqueeze(2)
                    .expand(n_draws, n_cells, n_targets, n_latent_u)
                    .reshape(n_draws, n_cells * n_targets, n_latent_u)
                )
                expanded_targets = (
                    targets.unsqueeze(0)
                    .expand(n_cells, n_targets)
                    .reshape(n_cells * n_targets, 1)
                )
                chunk_z_base, chunk_raw, _ = self.qz(expanded_u, expanded_targets)
                chunk_z_base = chunk_z_base.reshape(
                    n_draws, n_cells, n_targets, self.n_latent
                )
                chunk_raw = chunk_raw.reshape(
                    n_draws, n_cells, n_targets, self.n_latent
                )
            else:
                raise ValueError("u must have shape (cell, latent) or (draw, cell, latent).")

            if z_base is None:
                z_base = chunk_z_base.select(sample_dim, 0)
            raw_chunks.append(chunk_raw)

        if z_base is None:  # pragma: no cover - construction rejects an empty registry
            raise RuntimeError("The registered sample universe is empty.")
        eps_raw = torch.cat(raw_chunks, dim=sample_dim)
        eps_centered = eps_raw - eps_raw.mean(dim=sample_dim, keepdim=True)
        z_all = z_base.unsqueeze(sample_dim) + eps_centered
        return z_base, eps_raw, eps_centered, z_all

    @staticmethod
    def _gather_sample(
        values: torch.Tensor,
        sample_index: torch.Tensor,
    ) -> torch.Tensor:
        """Gather one registered sample per cell from an all-sample tensor."""
        sample_index = sample_index.to(torch.int64).flatten()
        if values.ndim == 3:
            gather_index = sample_index[:, None, None].expand(
                -1, 1, values.shape[-1]
            )
            return values.gather(1, gather_index).squeeze(1)
        gather_index = sample_index[None, :, None, None].expand(
            values.shape[0], -1, 1, values.shape[-1]
        )
        return values.gather(2, gather_index).squeeze(2)

    # ------------------------------------------------------------------
    # Inference: u → (z_base, eps) → z
    # ------------------------------------------------------------------

    @auto_move_data
    def _regular_inference(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        batch_index: torch.Tensor | None = None,
        panel_index: torch.Tensor | None = None,
        label: torch.Tensor | None = None,
        n_samples: int = 1,
        cont_covs: torch.Tensor | None = None,
        cat_covs: torch.Tensor | None = None,
        sample_index: torch.Tensor | None = None,
        cf_sample: torch.Tensor | None = None,
        target_chunk_size: int | None = None,
    ) -> dict[str, torch.Tensor | dict]:
        """Compute inference quantities with the hierarchical u→z decomposition.

        Calls TotalVI's ``_regular_inference`` to obtain library size, protein
        background priors, and dispersion parameters (all of which are parallel
        to ``z`` and require no ``z`` dependency), then runs the configured
        u-encoder to replace TotalVI's ``qz`` and ``z``:

        .. code-block::

            qu = EncoderXU_TotalVI(x, y, real_sample)  # sample-conditioned Normal
            u  = qu.rsample()                           # batch or MC by batch
            z_base, eps = qz(u, cf_sample)              # attention over per-sample embed
            z  = z_base + eps                           # → decoder input

        ``QZ_KEY`` in the returned dict is replaced with ``qu`` so that
        TotalVI's ``loss()`` computes ``kl_u = KL(qu, N(0,1))`` automatically.
        Our ``loss()`` override then adds ``kl_z = -log p(eps)``.

        Parameters
        ----------
        sample_index
            Integer donor index, shape ``(batch, 1)`` or ``(batch,)``.
            Used by ``qu``; always the **real** donor (never counterfactual).
        cf_sample
            Counterfactual donor override for ``eps`` only.  If not ``None``,
            replaces ``sample_index`` in the :class:`~.EncoderUZ` forward pass.
            ``qu`` always uses ``sample_index`` (real donor).
        """
        # TotalVI encoder pass — we keep ql, library_gene, back_mean_prior, dispersions.
        # Its qz and z are discarded and replaced below.
        out = super()._regular_inference(
            x, y,
            batch_index=batch_index,
            panel_index=panel_index,
            label=label,
            n_samples=n_samples,
            cont_covs=cont_covs,
            cat_covs=cat_covs,
        )

        if not hasattr(self, "qz"):
            # Hierarchy not yet built (n_sample=0 placeholder). Return TotalVI outputs.
            return out

        # Pass technical covariates only when encode_covariates=True. Biological
        # sample conditioning is controlled separately by u_encoder_mode.
        qu_batch_index = batch_index if self.encode_covariates else None
        qu = self.qu(
            x,
            y,
            sample_index,
            batch_index=qu_batch_index,
            cont_covs=cont_covs if self.encode_covariates else None,
            cat_covs=cat_covs if self.encode_covariates else None,
            batch_rep=self._batch_representation_for(qu_batch_index),
        )  # Normal: params (batch, n_latent_u)
        if n_samples > 1:
            u = qu.rsample((n_samples,))  # (n_samples, batch, n_latent)
        else:
            u = qu.rsample()  # (batch, n_latent)

        # Replace TotalVI's qz with qu so base loss computes KL(qu, N(0,1)) as kl_u
        out[MODULE_KEYS.QZ_KEY] = qu
        out["qu"] = qu

        # eps residual: cf_sample enables counterfactual donor substitution
        sample_index_cf = sample_index if cf_sample is None else cf_sample
        if self.hierarchy_mode == "legacy":
            z_base, eps, eps_dist = self.qz(u, sample_index_cf)
            z = z_base + eps
        else:
            z_base, eps_raw_all, eps_centered_all, z_all = self._all_sample_residuals(
                u,
                target_chunk_size=target_chunk_size,
            )
            eps = self._gather_sample(eps_centered_all, sample_index_cf)
            z = self._gather_sample(z_all, sample_index_cf)
            eps_dist = None
            out["eps_raw_all"] = eps_raw_all
            out["eps_centered_all"] = eps_centered_all
            out["z_all"] = z_all

        out[MODULE_KEYS.Z_KEY] = z
        out["u"] = u
        out["z_base"] = z_base
        out["eps"] = eps
        out["eps_dist"] = eps_dist  # None when use_map=True
        return out

    def build_u_prior(
        self,
        u: torch.Tensor,
        label_index: torch.Tensor | None = None,
    ):
        """Build the configured prior over ``u``."""
        return _build_u_prior(self, u, label_index)

    def kl_u(
        self,
        qu: Normal,
        sampled_u: torch.Tensor,
        label_index: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute ``KL(q(u|x,s) || p(u))`` under the configured prior."""
        return _kl_u(self, qu, sampled_u, label_index)

    def _loss_with_mc_samples(
        self,
        tensors: dict[str, torch.Tensor],
        inference_outputs: dict[str, torch.Tensor | dict],
        generative_outputs: dict[str, torch.Tensor | dict],
        pro_recons_weight: float,
        kl_weight: float,
    ) -> LossOutput:
        """Compute the TOTALVAE loss when hierarchy outputs carry MC samples."""
        qz = inference_outputs[MODULE_KEYS.QZ_KEY]
        ql = inference_outputs[MODULE_KEYS.QL_KEY]
        px_ = generative_outputs["px_"]
        py_ = generative_outputs["py_"]
        per_batch_efficiency = generative_outputs["per_batch_efficiency"]

        x = tensors[REGISTRY_KEYS.X_KEY]
        batch_index = tensors[REGISTRY_KEYS.BATCH_KEY]
        panel_index = tensors[self.panel_key]
        y = tensors[REGISTRY_KEYS.PROTEIN_EXP_KEY]

        if self.protein_batch_mask is not None:
            pro_batch_mask_minibatch = torch.zeros_like(y)
            for b in torch.unique(panel_index):
                b_indices = (panel_index == b).reshape(-1)
                pro_batch_mask_minibatch[b_indices] = torch.tensor(
                    self.protein_batch_mask[str(int(b.item()))].astype("float32"),
                    device=y.device,
                )
        else:
            pro_batch_mask_minibatch = None

        reconst_loss_gene, reconst_loss_protein = self.get_reconstruction_loss(
            x,
            y,
            px_,
            py_,
            pro_batch_mask_minibatch,
            per_batch_efficiency,
        )

        # NOTE: This kl_div_z is a *preliminary* placeholder using the pre-hierarchical
        # KL(q_z, N(0,1)).  Both this value and the `loss` computed below are discarded
        # by the calling loss() method, which recomputes them using kl_u + kl_z after
        # the two-level KL is available.  It is kept here to satisfy LossOutput's
        # interface contract (kl_local["kl_div_z"] must be present).
        kl_div_z = kl_divergence(qz, Normal(0, 1)).sum(dim=1)
        if not self.use_observed_lib_size:
            n_batch = self.library_log_means.shape[1]
            local_library_log_means = F.linear(
                F.one_hot(batch_index.squeeze(-1), n_batch).float(),
                self.library_log_means,
            )
            local_library_log_vars = F.linear(
                F.one_hot(batch_index.squeeze(-1), n_batch).float(),
                self.library_log_vars,
            )
            kl_div_l_gene = kl_divergence(
                ql,
                Normal(local_library_log_means, torch.sqrt(local_library_log_vars)),
            ).sum(dim=1)
        else:
            kl_div_l_gene = torch.zeros_like(kl_div_z)

        kl_div_back_pro_full = kl_divergence(
            Normal(py_["back_alpha"], py_["back_beta"]),
            inference_outputs["back_mean_prior"],
        )
        lkl_back_pro_full = -torch.distributions.LogNormal(
            torch.tensor([0.0], device=x.device),
            torch.tensor([1.0], device=x.device),
        ).log_prob(per_batch_efficiency)
        lkl_protein_expressed = -1e-3 * torch.distributions.Bernoulli(
            logits=py_["mixing"]
        ).log_prob(torch.ones_like(py_["mixing"]))
        if pro_batch_mask_minibatch is not None:
            while pro_batch_mask_minibatch.ndim < kl_div_back_pro_full.ndim:
                pro_batch_mask_minibatch = pro_batch_mask_minibatch.unsqueeze(0)
            kl_div_back_pro_full = pro_batch_mask_minibatch.bool() * kl_div_back_pro_full
        kl_div_back_pro = (
            kl_div_back_pro_full.sum(dim=-1)
            + lkl_back_pro_full.sum(dim=-1)
            + lkl_protein_expressed.sum(dim=-1)
        )

        loss = torch.mean(
            reconst_loss_gene
            + kl_weight * pro_recons_weight * reconst_loss_protein
            + kl_weight * kl_div_z
            + kl_div_l_gene
            + kl_weight * kl_div_back_pro
        )
        reconst_losses = {
            "reconst_loss_gene": reconst_loss_gene,
            "reconst_loss_protein": reconst_loss_protein,
        }
        kl_local = {
            "kl_div_z": kl_div_z,
            "kl_div_l_gene": kl_div_l_gene,
            "kl_div_back_pro": kl_div_back_pro,
        }
        extra_metrics = (
            {
                "z": inference_outputs[MODULE_KEYS.Z_KEY],
                "batch": tensors[REGISTRY_KEYS.BATCH_KEY],
                "labels": tensors[REGISTRY_KEYS.LABELS_KEY],
            }
            if self.extra_payload_autotune
            else {}
        )
        return LossOutput(
            loss=loss,
            reconstruction_loss=reconst_losses,
            kl_local=kl_local,
            extra_metrics=extra_metrics,
        )

    # ------------------------------------------------------------------
    # Loss: two-level KL (kl_u + kl_z)
    # ------------------------------------------------------------------

    def loss(
        self,
        tensors: dict[str, torch.Tensor],
        inference_outputs: dict[str, torch.Tensor | dict],
        generative_outputs: dict[str, torch.Tensor | dict],
        pro_recons_weight: float = 1.0,
        kl_weight: float = 1.0,
    ) -> LossOutput:
        r"""Extend TotalVI's loss with the second-level KL for the sample residual.

        TotalVI's existing ``kl_div_z`` term is repurposed as ``kl_u =
        KL(q_u \\| p_u)`` where ``p_u`` is the configured prior (default: a
        learned mixture-of-Gaussians, see :attr:`u_prior_mixture`).  This
        method adds:

        .. math::
            kl_z = -\\log p(\\varepsilon) = -\\log N(0,\\; \\exp(\\text{pz\\_scale}))(\\varepsilon)

        and folds it into both ``loss`` and ``kl_local["kl_div_z"]``.
        """
        if "eps" in inference_outputs and inference_outputs[MODULE_KEYS.Z_KEY].ndim > 2:
            loss_out = self._loss_with_mc_samples(
                tensors,
                inference_outputs,
                generative_outputs,
                pro_recons_weight,
                kl_weight,
            )
        else:
            loss_out = super().loss(
                tensors, inference_outputs, generative_outputs,
                pro_recons_weight, kl_weight,
            )

        if "eps" not in inference_outputs:
            # Hierarchy not built (n_sample=0 placeholder).
            return loss_out

        label_index = tensors[REGISTRY_KEYS.LABELS_KEY]
        kl_u = self.kl_u(
            inference_outputs.get("qu", inference_outputs[MODULE_KEYS.QZ_KEY]),
            inference_outputs["u"],
            label_index,
        )

        if self.z_u_prior:
            peps = Normal(0.0, torch.exp(self.pz_scale.clamp(min=-4.0)))
            if self.hierarchy_mode == "centered_v2":
                eps_raw_all = inference_outputs["eps_raw_all"]
                sample_dim = 1 if eps_raw_all.ndim == 3 else 2
                kl_z = -peps.log_prob(eps_raw_all).sum(dim=-1).mean(dim=sample_dim)
                if kl_z.ndim > 1:
                    kl_z = kl_z.mean(dim=0)
            else:
                eps = inference_outputs["eps"]
                eps_dist = inference_outputs.get("eps_dist")
                if eps_dist is not None:
                    # use_map=False: analytic KL(q(eps) || p(eps)) includes entropy
                    kl_z = kl_divergence(eps_dist, peps).sum(dim=-1)
                else:
                    # use_map=True: deterministic eps uses cross-entropy -log p(eps)
                    kl_z = -peps.log_prob(eps).sum(dim=-1)
                # kl_z shape: (batch,) or (n_samples, batch) → reduce mc dim
                if kl_z.ndim > 1:
                    kl_z = kl_z.mean(dim=0)
        else:
            kl_z = torch.zeros_like(kl_u)

        # Update kl_local: kl_div_z now = kl_u_weight*kl_u + kl_z_weight*kl_z
        kl_local = dict(loss_out.kl_local)
        kl_local["kl_div_z"] = self.kl_u_weight * kl_u + self.kl_z_weight * kl_z

        if self._scale_observations and self.n_obs_per_sample is not None:
            # Weight each cell's ELBO by 1/n_cells_in_that_sample so high-cell-count
            # donors do not dominate training.  Reconstruct the full per-cell ELBO from
            # individual component tensors (parent loss() already called torch.mean()).
            sample_index = tensors[REGISTRY_KEYS.SAMPLE_KEY].to(torch.int64).flatten()
            prefactors = self.n_obs_per_sample[sample_index]  # (batch,)
            per_cell = (
                loss_out.reconstruction_loss["reconst_loss_gene"]
                + kl_weight
                * pro_recons_weight
                * loss_out.reconstruction_loss["reconst_loss_protein"]
                + kl_weight * kl_local["kl_div_z"]
                + kl_local["kl_div_l_gene"]
                + kl_weight * kl_local["kl_div_back_pro"]
            )
            if self.hierarchy_mode == "centered_v2":
                weights = self.n_obs_per_sample.sum() / (self._n_sample * prefactors)
                loss = (per_cell * weights).mean()
            else:
                loss = (per_cell / prefactors).mean()
        else:
            if self._scale_observations:
                import warnings
                warnings.warn(
                    "scale_observations=True but n_obs_per_sample is None "
                    "(buffer was not restored after load). "
                    "Call _setup_hierarchy(n_obs_per_sample=...) to restore weighting.",
                    UserWarning,
                    stacklevel=2,
                )
            loss = (
                loss_out.reconstruction_loss["reconst_loss_gene"]
                + kl_weight * pro_recons_weight * loss_out.reconstruction_loss[
                    "reconst_loss_protein"
                ]
                + kl_weight * kl_local["kl_div_z"]
                + kl_local["kl_div_l_gene"]
                + kl_weight * kl_local["kl_div_back_pro"]
            ).mean()

        return LossOutput(
            loss=loss,
            reconstruction_loss=loss_out.reconstruction_loss,
            kl_local=kl_local,
            extra_metrics=loss_out.extra_metrics,
        )

    @torch.inference_mode()
    def _deterministic_decoder_parameters(
        self,
        z: torch.Tensor,
        library_size: torch.Tensor,
        batch_index: torch.Tensor,
        cont_covs: torch.Tensor | None = None,
        cat_covs: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Decode RNA and protein means without using stochastic ``rate_back``."""
        if cont_covs is None:
            decoder_input = z
        else:
            decoder_input = torch.cat([z, cont_covs], dim=-1)
        categorical_input = (
            tuple(torch.split(cat_covs, 1, dim=1))
            if cat_covs is not None
            else ()
        )
        decoder = self.decoder

        # Under `batch_representation="embedding"` the wrapped decoder was built without
        # the batch category, so the embedding is folded into `decoder_input` here and
        # `batch_index` is dropped from the categorical arguments.
        if isinstance(decoder, BatchEmbeddingDecoderAdapter):
            decoder_input = decoder.batch_input(decoder_input, batch_index)
            conditioning = categorical_input
        else:
            conditioning = (batch_index, *categorical_input)

        px = decoder.px_decoder(decoder_input, *conditioning)
        px_cat_z = torch.cat([px, decoder_input], dim=-1)
        rna_scale = decoder.px_scale_activation(
            decoder.px_scale_decoder(
                px_cat_z,
                *conditioning,
            )
        )

        py_back = decoder.py_back_decoder(
            decoder_input,
            *conditioning,
        )
        py_back_cat_z = torch.cat([py_back, decoder_input], dim=-1)
        back_alpha = decoder.py_back_mean_log_alpha(
            py_back_cat_z,
            *conditioning,
        )
        back_beta = (
            decoder.activation_function_bg(
                decoder.py_back_mean_log_beta(
                    py_back_cat_z,
                    *conditioning,
                )
            )
            + 1e-8
        )

        py_fore = decoder.py_fore_decoder(
            decoder_input,
            *conditioning,
        )
        py_fore_cat_z = torch.cat([py_fore, decoder_input], dim=-1)
        fore_scale = (
            decoder.py_fore_scale_decoder(
                py_fore_cat_z,
                *conditioning,
            )
            + 1
            + 1e-8
        )

        mixing_hidden = decoder.sigmoid_decoder(
            decoder_input,
            *conditioning,
        )
        mixing_cat_z = torch.cat([mixing_hidden, decoder_input], dim=-1)
        mixing = decoder.py_background_decoder(
            mixing_cat_z,
            *conditioning,
        )
        efficiency = torch.exp(
            F.linear(
                F.one_hot(
                    batch_index.squeeze(-1).to(torch.int64),
                    self.n_batch,
                ).float(),
                self.log_per_batch_efficiency,
            )
        )
        background = efficiency * torch.exp(back_alpha + 0.5 * back_beta.square())
        foreground = background * fore_scale
        foreground_probability = 1.0 - torch.sigmoid(mixing)
        background_contribution = (1.0 - foreground_probability) * background
        foreground_contribution = foreground_probability * foreground
        return {
            "rna_scale": rna_scale,
            "rna_rate": library_size * rna_scale,
            "protein_background_component_mean": background,
            "protein_foreground_component_mean": foreground,
            "protein_foreground_probability": foreground_probability,
            "protein_background_contribution": background_contribution,
            "protein_foreground_contribution": foreground_contribution,
            "protein_total_mean": background_contribution + foreground_contribution,
            "protein_batch_efficiency": efficiency,
        }

    @torch.inference_mode()
    def _infer_lfc_aux(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        batch_index: torch.Tensor,
        panel_index: torch.Tensor | None = None,
        label: torch.Tensor | None = None,
        cont_covs: torch.Tensor | None = None,
        cat_covs: torch.Tensor | None = None,
        sample_index: torch.Tensor | None = None,
        cf_sample: torch.Tensor | None = None,
        **_,
    ) -> dict:
        """Return inference outputs that are constant across CRN MC draws.

        ``library_gene`` depends only on ``x`` and ``batch_index``, not on
        ``u_anchor``, so it is safe to compute once per batch value and cache.
        Called by ``_stats.py`` before the MC loop to avoid running the full
        encoder ``mc_samples*(1+n_fixed)`` times per batch value.
        """
        n_cells = x.shape[0]
        if label is None:
            label = torch.zeros(n_cells, 1, dtype=torch.long, device=x.device)
        if panel_index is None:
            panel_index = batch_index
        out = self._regular_inference(
            x,
            y,
            batch_index=batch_index,
            panel_index=panel_index,
            label=label,
            n_samples=1,
            cont_covs=cont_covs,
            cat_covs=cat_covs,
            sample_index=sample_index,
            cf_sample=cf_sample,
        )
        return {"library_gene": out["library_gene"]}

    @torch.inference_mode()
    def compute_h_from_x_eps(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        sample_index: torch.Tensor,
        batch_index: torch.Tensor,
        extra_eps: torch.Tensor,
        label: torch.Tensor | None = None,
        panel_index: torch.Tensor | None = None,
        cont_covs: torch.Tensor | None = None,
        cat_covs: torch.Tensor | None = None,
        cf_sample: torch.Tensor | None = None,
        u_anchor: torch.Tensor | None = None,
        _lfc_aux: dict | None = None,
    ) -> torch.Tensor:
        """Return ``concat([px_scale, py_scale_det])`` at ``z = z_base + extra_eps``.

        This is the decoder hook consumed by the LFC block in
        :func:`~scvi.external.mrtotalvi._stats.differential_expression` when
        ``store_lfc=True``.  It mirrors MRVI's ``compute_h_from_x_eps`` but
        handles the TotalVI multimodal output (RNA + protein) and uses a
        deterministic protein background reconstruction (D-021) so that the
        x_1/x_0 contrast differs only via ``extra_eps``, not via independent
        background draws.

        Parameters
        ----------
        x
            RNA count matrix, shape ``(batch, n_genes)``.
        y
            Protein count matrix, shape ``(batch, n_proteins)``.
        sample_index
            Integer donor index, shape ``(batch, 1)`` or ``(batch,)``.
        batch_index
            Batch integer index for the counterfactual decode, shape
            ``(batch, 1)``.  Can differ from the cell's own batch to obtain
            batch-averaged LFC.
        extra_eps
            Counterfactual eps shift, shape ``(batch, n_latent)``.
            Set to the regression beta vector for the active covariate when
            computing x_1; set to the null (zeros or ``eps_mean``) for x_0.
        label
            Cell-type label tensor.  Defaults to zeros when ``None``.
        panel_index
            Protein panel index.  Defaults to ``batch_index`` when ``None``.
        cont_covs
            Continuous covariate tensor (passed through to inference).
        cat_covs
            Categorical covariate tensor (passed through to inference).
        cf_sample
            Counterfactual sample override for the eps encoder.  ``None``
            uses ``sample_index`` (real donor).

        Returns
        -------
        :class:`torch.Tensor`
            Shape ``(batch, n_genes + n_proteins)``.  RNA columns first,
            protein columns second.  The ``gene``/``protein`` coordinate
            split is applied at the model level (B3) when assembling the
            xarray output.
        """
        n_cells = x.shape[0]
        if label is None:
            label = torch.zeros(n_cells, 1, dtype=torch.long, device=x.device)
        if panel_index is None:
            panel_index = batch_index

        sample_index_cf = sample_index if cf_sample is None else cf_sample

        if u_anchor is not None and _lfc_aux is not None:
            # CRN fast path: skip inference — library_gene is constant per batch
            # value and was pre-computed by _infer_lfc_aux in _stats.py.
            u = u_anchor
            library_gene = _lfc_aux["library_gene"]
        else:
            # Full inference path: needed when u_anchor is None (legacy biased
            # path) or when no aux cache is available.
            if u_anchor is None:
                import warnings
                warnings.warn(
                    "compute_h_from_x_eps called without u_anchor: falling back to "
                    "qu.mean, which produces Jensen-biased LFC estimates. Pass "
                    "u_anchor (a sample from q(u|x)) for unbiased CRN estimation.",
                    UserWarning,
                    stacklevel=2,
                )
            out = self._regular_inference(
                x,
                y,
                batch_index=batch_index,
                panel_index=panel_index,
                label=label,
                n_samples=1,
                cont_covs=cont_covs,
                cat_covs=cat_covs,
                sample_index=sample_index,
                cf_sample=cf_sample,
            )
            qu = out["qu"]
            u = u_anchor if u_anchor is not None else qu.mean
            library_gene = out["library_gene"]

        z_base, _, _ = self.qz(u, sample_index_cf)
        z = z_base + extra_eps      # counterfactual latent

        gen_out = self.generative(
            z=z,
            library_gene=library_gene,
            batch_index=batch_index,
            label=label,
            cont_covs=cont_covs,
            cat_covs=cat_covs,
        )

        px_ = gen_out["px_"]
        py_ = gen_out["py_"]

        # RNA: softmax scale is deterministic given z.
        px_scale = px_["scale"]  # (batch, n_genes)

        # Protein: deterministic background reconstruction (D-021).
        # py_["scale"] is stochastic because DecoderTOTALVI calls
        # Normal(back_alpha, back_beta).rsample() internally.  Using that
        # stochastic draw in both x_1 and x_0 introduces independent
        # background noise that does NOT cancel in the LFC.  Fix: use the
        # log-mean back_alpha directly (exp of the Gaussian mean) as a
        # deterministic background stand-in.
        rate_back_det = torch.exp(py_["back_alpha"])          # (batch, n_proteins)
        rate_fore_det = rate_back_det * py_["fore_scale"]     # fore_scale is deterministic
        # py_["mixing"] is a raw logit; sigmoid → mixing probability.
        protein_mixing_det = torch.sigmoid(py_["mixing"])
        py_scale_det = F.normalize(
            (1.0 - protein_mixing_det) * rate_fore_det, p=1, dim=-1
        )  # (batch, n_proteins)

        return torch.cat([px_scale, py_scale_det], dim=-1)   # (batch, n_genes + n_proteins)
