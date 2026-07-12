"""MrTotalVAE — TotalVI with MrVI-style hierarchical donor latent space."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, kl_divergence

from scvi import REGISTRY_KEYS
from scvi.module._constants import MODULE_KEYS
from scvi.module._totalvae import TOTALVAE
from scvi.module.base import LossOutput, auto_move_data

from ._components import (
    EncoderUZ,
    EncoderXU_TotalVI,
    build_u_prior as _build_u_prior,
    init_u_prior,
    kl_u as _kl_u,
)


class MrTotalVAE(TOTALVAE):
    """TotalVI VAE with an MrVI-style u→z hierarchical latent space.

    Grafts MrVI's hierarchical design onto TotalVI as faithfully as the
    multimodal input allows.  A sample-conditioned u-encoder
    (:class:`~.EncoderXU_TotalVI`) replaces TotalVI's stock encoder for the
    base representation ``u``; a donor-specific residual ``eps`` is computed
    by attending over a per-sample embedding table via :class:`~.EncoderUZ`,
    giving:

    .. math::
        u \\sim q_u(x_{\\text{rna}},\\, x_{\\text{prot}},\\, d)
        \\quad\\text{(sample-conditioned)}

    .. math::
        z = z_{\\text{base}}(u) + \\varepsilon,\\quad
        \\varepsilon \\sim \\text{AttentionBlock}(u,\\; e_d)

    Because ``n_latent_u`` is left at its default (isomorphic), ``z_base = u``
    and the decoder input dimension is identical to stock TotalVI — no decoder
    changes are needed.

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
        scale_observations: bool = False,
        kl_u_weight: float = 1.0,
        kl_z_weight: float = 1.0,
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
        self._scale_observations = scale_observations
        # Per-term KL weights: static scalars applied as kl_u_weight*kl_u + kl_z_weight*kl_z
        # before the global kl_weight annealing. Defaults (1.0, 1.0) reproduce prior behavior.
        # A separate per-term annealing schedule would require training-plan changes.
        self.kl_u_weight = float(kl_u_weight)
        self.kl_z_weight = float(kl_z_weight)
        self.n_continuous_cov = int(n_continuous_cov)
        self.n_cats_per_cov = list(n_cats_per_cov or [])
        self.encode_covariates = bool(encode_covariates)

        if n_sample > 0:
            self._setup_hierarchy(n_sample, n_latent_sample, z_u_prior_scale, learn_z_u_prior_scale)

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
        scale_observations: bool | None = None,
        n_obs_per_sample: torch.Tensor | None = None,
        n_labels: int | None = None,
        prior_centroids: torch.Tensor | None = None,
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
        """
        if n_latent_sample is None:
            n_latent_sample = self._n_latent_sample
        if z_u_prior_scale is None:
            z_u_prior_scale = self._z_u_prior_scale
        if learn_z_u_prior_scale is None:
            learn_z_u_prior_scale = self._learn_z_u_prior_scale
        if use_map is None:
            use_map = self._use_map
        if scale_observations is None:
            scale_observations = self._scale_observations

        self._n_sample = n_sample
        self._use_map = use_map
        self._scale_observations = scale_observations
        if n_labels is not None:
            self.n_labels = int(n_labels)

        n_latent_u = (
            self.n_latent if self._n_latent_u_requested is None else int(self._n_latent_u_requested)
        )

        # Sample-conditioned u-encoder: mirrors MrVI's EncoderXU, multimodal input
        # u is the sample-uninformed cell-state representation: never condition on batch.
        # Batch conditioning belongs in the parent TotalVAE z-encoder and decoder only.
        self.qu = EncoderXU_TotalVI(
            n_input_genes=self.n_input_genes,
            n_input_proteins=self.n_input_proteins,
            n_latent=n_latent_u,
            n_sample=n_sample,
            n_batch=self.n_batch,
            n_continuous_cov=self.n_continuous_cov,
            n_cats_per_cov=self.n_cats_per_cov,
            encode_covariates=self.encode_covariates,
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
        )

        if learn_z_u_prior_scale:
            self.pz_scale = nn.Parameter(torch.zeros(self.n_latent))
        else:
            self.register_buffer(
                "pz_scale",
                torch.full((self.n_latent,), float(z_u_prior_scale)),
            )

        # Per-sample cell counts for observation reweighting (persistent so load_state_dict restores it)
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
        batch_idx = torch.zeros(K, 1, device=pseudo.device, dtype=torch.long) if self.encode_covariates else None
        return self.qu(x_p, y_p, sample_idx, batch_index=batch_idx)

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
    ) -> dict[str, torch.Tensor | dict]:
        """Compute inference quantities with the hierarchical u→z decomposition.

        Calls TotalVI's ``_regular_inference`` to obtain library size, protein
        background priors, and dispersion parameters (all of which are parallel
        to ``z`` and require no ``z`` dependency), then runs the sample-conditioned
        u-encoder to replace TotalVI's ``qz`` and ``z``:

        .. code-block::

            qu = EncoderXU_TotalVI(x, y, real_sample)  # sample-conditioned Normal
            u  = qu.rsample()                           # (batch, n_latent) or (mc, batch, n_latent)
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

        # u-encoder: sample-conditioned; pass batch/covariate info only when
        # encode_covariates=True.  Default is False: u stays batch-uninformed,
        # matching MRVI's design and the pre-trained checkpoint behavior.
        qu = self.qu(
            x,
            y,
            sample_index,
            batch_index=batch_index if self.encode_covariates else None,
            cont_covs=cont_covs if self.encode_covariates else None,
            cat_covs=cat_covs if self.encode_covariates else None,
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
        z_base, eps, eps_dist = self.qz(u, sample_index_cf)
        z = z_base + eps

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
        """Extend TotalVI's loss with the second-level KL for the sample residual.

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
            eps = inference_outputs["eps"]
            eps_dist = inference_outputs.get("eps_dist")
            peps = Normal(0.0, torch.exp(self.pz_scale.clamp(min=-4.0)))
            if eps_dist is not None:
                # use_map=False: analytic KL(q(eps) || p(eps)) — correct ELBO includes entropy
                kl_z = kl_divergence(eps_dist, peps).sum(dim=-1)
            else:
                # use_map=True: deterministic eps — cross-entropy -log p(eps) is the correct term
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
                + kl_weight * pro_recons_weight * loss_out.reconstruction_loss["reconst_loss_protein"]
                + kl_weight * kl_local["kl_div_z"]
                + kl_local["kl_div_l_gene"]
                + kl_weight * kl_local["kl_div_back_pro"]
            )
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
