"""MrMultiVAE — MULTIVAE with MrVI-style hierarchical donor latent space."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, kl_divergence

from scvi import REGISTRY_KEYS
from scvi.module._multivae import MULTIVAE, get_reconstruction_loss_protein
from scvi.module.base import LossOutput, auto_move_data

from ..mrtotalvi._components import EncoderUZ, EncoderXU_MultiVI
from ..mrtotalvi._components import (
    build_u_prior as _build_u_prior,
    init_u_prior,
    kl_u as _kl_u,
)


def _expand_to_match_mc(target: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    """Expand a batch-shaped target across MC samples when needed."""
    while target.ndim < reference.ndim:
        target = target.unsqueeze(0)
    return target.expand_as(reference)


class MrMultiVAE(MULTIVAE):
    """MULTIVAE with an MrVI-style u→z hierarchical latent space.

    Grafts MrVI's hierarchical design onto MULTIVAE.  A sample-conditioned
    u-encoder (:class:`~scvi.external.mrtotalvi._components.EncoderXU_MultiVI`)
    maps MULTIVAE's mixed latent through donor-conditioned normalization layers,
    then a donor-specific residual ``eps`` is computed by attending over a
    per-donor embedding table via
    :class:`~scvi.external.mrtotalvi._components.EncoderUZ`:

    .. math::
        u \\sim q_u(u_0,\\, d)
        \\quad\\text{(sample-conditioned, where } u_0 \\text{ is MULTIVAE's mixed latent)}

    .. math::
        z = z_{\\text{base}}(u) + \\varepsilon,\\quad
        \\varepsilon \\sim \\text{AttentionBlock}(u,\\; e_{\\text{sample}})

    The decoder receives ``z`` with the same shape as MULTIVAE's ``z`` —
    no decoder changes are needed.

    Two-level KL loss:

    * ``kl_u = KL(q_u \\| p_u)`` — from the sample-conditioned u-encoder, where
      ``p_u`` is by default a learned mixture-of-Gaussians prior
      (``u_prior_mixture=True``).  Replaces MULTIVAE's ``kl_divergence_z`` via
      the ``qz_m``/``qz_v`` override.
    * ``kl_z = -\\log p(\\varepsilon)`` — new term added here.

    Parameters
    ----------
    n_input_regions
        Number of input ATAC regions.
    n_input_genes
        Number of input RNA genes.
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
        If ``True``, ``pz_scale`` is a learnable :class:`~torch.nn.Parameter`.
    **kwargs
        Forwarded verbatim to :class:`~scvi.module._multivae.MULTIVAE`.
    """

    def __init__(
        self,
        n_input_regions: int = 0,
        n_input_genes: int = 0,
        n_input_proteins: int = 0,
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
        protein_in_encoder: bool = False,
        qz_kwargs: dict | None = None,
        qu_kwargs: dict | None = None,
        use_map: bool = True,
        scale_observations: bool = False,
        **kwargs,
    ) -> None:
        if kwargs.get("latent_distribution", "normal") != "normal":
            raise ValueError(
                "MrMultiVAE requires latent_distribution='normal'. "
                "Under 'ln' (softmax normalisation), the additive u→z "
                "hierarchy is mathematically invalid."
            )
        kwargs["latent_distribution"] = "normal"

        n_continuous_cov = kwargs.get("n_continuous_cov", 0)
        n_cats_per_cov = kwargs.get("n_cats_per_cov", None)
        encode_covariates = kwargs.get("encode_covariates", False)

        super().__init__(
            n_input_regions=n_input_regions,
            n_input_genes=n_input_genes,
            n_input_proteins=n_input_proteins,
            **kwargs,
        )

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
        self.n_continuous_cov = int(n_continuous_cov)
        self.n_cats_per_cov = list(n_cats_per_cov or [])
        self.encode_covariates = bool(encode_covariates)
        self.protein_in_encoder = bool(protein_in_encoder)

        if protein_in_encoder and n_input_proteins == 0:
            import warnings
            warnings.warn(
                "protein_in_encoder=True has no effect when n_input_proteins=0. "
                "Register protein data or set protein_in_encoder=False.",
                UserWarning,
                stacklevel=2,
            )

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
    ) -> None:
        """Build or replace the u→z hierarchy after MULTIVAE is initialised.

        Safe to call after ``super().__init__()`` — e.g. once the registry
        summary stats are available in :meth:`~MrMultiVI.__init__`.

        Parameters
        ----------
        n_sample
            Total number of donors registered in the MuData.
        n_latent_sample, z_u_prior_scale, learn_z_u_prior_scale
            Override stored values; ``None`` → use stored value.
        use_map
            If ``False``, treat eps as stochastic.  ``None`` → use stored value.
        scale_observations
            If ``True``, weight each cell's ELBO by ``1 / n_cells_in_that_sample``.
            ``None`` → use stored value.
        n_obs_per_sample
            Integer tensor of shape ``(n_sample,)`` with per-donor cell counts.
            Required (and used only) when ``scale_observations=True``.
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

        # Sample-conditioned u-encoder: mirrors MrVI's EncoderXU, takes MULTIVAE mixed latent
        # u is the sample-uninformed cell-state representation: never condition on batch.
        # Batch conditioning belongs in the parent MultiVAE z-encoder and decoder only.
        n_protein_in = self.n_input_proteins if getattr(self, "protein_in_encoder", False) else 0
        self.qu = EncoderXU_MultiVI(
            n_input=self.n_latent,
            n_latent=n_latent_u,
            n_sample=n_sample,
            n_batch=self.n_batch,
            n_continuous_cov=self.n_continuous_cov,
            n_cats_per_cov=self.n_cats_per_cov,
            encode_covariates=self.encode_covariates,
            n_input_proteins=n_protein_in,
            **self.qu_kwargs,
        )

        # Isomorphic dims (n_latent_u=None) → z_base = u, decoder input unchanged
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
            u_vamp_pseudo_dim=self.n_latent + n_protein_in,
        )

        if learn_z_u_prior_scale:
            self.pz_scale = nn.Parameter(torch.zeros(self.n_latent))
        else:
            self.register_buffer(
                "pz_scale",
                torch.full((self.n_latent,), float(z_u_prior_scale)),
            )

        # Per-sample cell counts for observation reweighting (non-persistent: recomputed at load)
        if scale_observations and n_obs_per_sample is not None:
            self.register_buffer("n_obs_per_sample", n_obs_per_sample.float(), persistent=False)
        else:
            self.n_obs_per_sample = None

    def _vamp_component_dist(self) -> Normal:
        """Run VampPrior pseudoinputs through qu to get K component distributions.

        Pseudoinputs are in MULTIVAE's continuous latent space (the u0 = qz_m
        space), so no positivity constraint is needed for the u0 portion.
        When ``protein_in_encoder=True``, the last ``n_input_proteins`` columns
        of the pseudoinput are Softplus-constrained before being passed as
        ``log1p``-transformed protein pseudo-counts.  Reference donor (index 0)
        is used — sample conditioning is excluded from the prior by design.
        """
        K = self.resolved_u_prior_mixture_k
        sample_idx = torch.zeros(K, 1, device=self.u_vamp_pseudo.device, dtype=torch.long)
        if getattr(self, "protein_in_encoder", False) and self.n_input_proteins > 0:
            pseudo = self.u_vamp_pseudo  # (K, n_latent + n_input_proteins)
            u0_pseudo = pseudo[:, : self.n_latent]
            y_pseudo = F.softplus(pseudo[:, self.n_latent :])
            return self.qu(u0_pseudo, sample_idx, y_protein=y_pseudo)
        return self.qu(self.u_vamp_pseudo, sample_idx)

    # ------------------------------------------------------------------
    # DataLoader → inference plumbing
    # ------------------------------------------------------------------

    def _get_inference_input(
        self, tensors: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Extend MULTIVAE's inference input with the sample index."""
        base = super()._get_inference_input(tensors)
        base["sample_index"] = tensors[REGISTRY_KEYS.SAMPLE_KEY]
        return base

    def _get_generative_input(self, tensors, inference_outputs, transform_batch=None):
        """Pass the hierarchical ``z = z_base + eps`` to the decoder.

        Overrides MULTIVAE's parent method to make it explicit that ``z``
        here is the hierarchical donor-aware representation, not MULTIVAE's
        plain posterior sample.

        Critical: always passes ``use_z_mean=False`` so that downstream
        counterfactual calls (e.g. :meth:`compute_h_from_x_eps`) correctly
        route through ``z`` rather than the ``qz_m`` shortcut — using
        ``qz_m`` would ignore the counterfactual ``eps`` and produce
        identically-zero log-fold-changes.
        """
        d = super()._get_generative_input(tensors, inference_outputs, transform_batch)
        # Ensure z is the full hierarchical representation (set by inference()).
        # The parent already reads inference_outputs["z"], but we make this
        # dependency explicit and guard against accidental use_z_mean=True.
        if "z" in inference_outputs:
            d["z"] = inference_outputs["z"]
        return d

    # ------------------------------------------------------------------
    # Inference: u → (z_base, eps) → z
    # ------------------------------------------------------------------

    @auto_move_data
    def inference(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        batch_index: torch.Tensor,
        cont_covs: torch.Tensor | None,
        cat_covs: torch.Tensor | None,
        label: torch.Tensor,
        cell_idx: torch.Tensor,
        size_factor: torch.Tensor | None,
        n_samples: int = 1,
        sample_index: torch.Tensor | None = None,
        cf_sample: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Compute inference quantities with the hierarchical u→z decomposition.

        Calls MULTIVAE's ``inference`` to obtain the mixed base ``u0``, then
        runs the sample-conditioned u-encoder and the attention residual block:

        .. code-block::

            u0  = mix_modalities(rna_enc(x), atac_enc(x), ...)  # MULTIVAE, unchanged
            qu  = EncoderXU_MultiVI(u0, real_sample)             # sample-conditioned Normal
            u   = qu.rsample()                                   # (batch, n_latent)
            z_base, eps = qz(u, cf_sample)                       # attention over per-sample embed
            z   = z_base + eps                                   # → decoder input

        ``qz_m`` and ``qz_v`` are replaced with ``qu.loc`` and ``qu.scale**2``
        so that MULTIVAE's ``loss()`` computes ``kl_u = KL(qu, N(0,1))``
        automatically.  Our ``loss()`` override then adds
        ``kl_z = -log p(eps)``.

        Parameters
        ----------
        sample_index
            Integer donor index, shape ``(batch, 1)`` or ``(batch,)``.
            Always the **real** donor (used by ``qu``).
        cf_sample
            Counterfactual donor override for ``eps`` only.  If not ``None``,
            replaces ``sample_index`` in the :class:`~.EncoderUZ` forward pass.
            ``qu`` always conditions on the real donor.
        """
        outputs = super().inference(
            x, y, batch_index, cont_covs, cat_covs, label, cell_idx, size_factor, n_samples
        )

        if not hasattr(self, "qz"):
            # Hierarchy not yet built (n_sample=0 placeholder).
            return outputs

        # u0: deterministic MULTIVAE mixed posterior mean.  The hierarchy supplies
        # the stochastic latent level; feeding a sampled base z here would leave the
        # sampled MULTIVAE posterior unregularized after qz_m/qz_v are replaced below.
        u0 = outputs["qz_m"]  # (batch, n_latent)

        # u-encoder: sample-conditioned; pass batch/covariate info only when
        # encode_covariates=True (opt-in; default False for MrMultiVI).  When False
        # the encoder stays fully batch-uninformed, making u comparable across
        # donors and batches.
        y_protein = y if (getattr(self, "protein_in_encoder", False) and self.n_input_proteins > 0) else None
        qu = self.qu(
            u0,
            sample_index,
            batch_index=batch_index if self.encode_covariates else None,
            cont_covs=cont_covs if self.encode_covariates else None,
            cat_covs=cat_covs if self.encode_covariates else None,
            y_protein=y_protein,
        )
        if n_samples > 1:
            u = qu.rsample((n_samples,))
        else:
            u = qu.rsample()

        # Replace qz_m / qz_v so MULTIVAE's loss computes KL(qu, N(0,1)) as kl_u
        outputs["qz_m"] = qu.loc
        outputs["qz_v"] = qu.scale ** 2

        # eps residual: cf_sample enables counterfactual donor substitution
        sample_index_cf = sample_index if cf_sample is None else cf_sample
        if sample_index_cf is None:
            raise ValueError(
                "sample_index must be provided when the MrMultiVI hierarchy is active."
            )
        z_base, eps, eps_dist = self.qz(u, sample_index_cf)
        z = z_base + eps

        outputs["u"] = u
        outputs["qu"] = qu
        outputs["z_base"] = z_base
        outputs["eps"] = eps
        outputs["eps_dist"] = eps_dist  # None when use_map=True
        outputs["z"] = z
        return outputs

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
        inference_outputs: dict[str, torch.Tensor],
        generative_outputs: dict[str, torch.Tensor],
        kl_weight: float,
    ) -> LossOutput:
        """Compute the MULTIVAE loss when hierarchy outputs carry MC samples."""
        x = inference_outputs["x"]
        x_rna = x[:, : self.n_input_genes]
        x_atac = x[:, self.n_input_genes : (self.n_input_genes + self.n_input_regions)]
        if self.n_input_proteins == 0:
            y = torch.zeros(x.shape[0], 1, device=x.device, requires_grad=False)
        else:
            y = tensors[REGISTRY_KEYS.PROTEIN_EXP_KEY]

        mask_expr = x_rna.sum(dim=1) > 0
        mask_acc = x_atac.sum(dim=1) > 0
        mask_pro = y.sum(dim=1) > 0

        p = generative_outputs["p"]
        libsize_acc = inference_outputs["libsize_acc"]
        reg_factor = torch.sigmoid(self.region_factors) if self.region_factors is not None else 1
        acc_target = _expand_to_match_mc((x_atac > 0).float(), p)
        rl_accessibility = torch.nn.BCELoss(reduction="none")(
            p * libsize_acc * reg_factor,
            acc_target,
        ).sum(dim=-1)

        px_rate = generative_outputs["px_rate"]
        px_r = generative_outputs["px_r"]
        px_dropout = generative_outputs["px_dropout"]
        expr_target = _expand_to_match_mc(x_rna, px_rate)
        rl_expression = self.get_reconstruction_loss_expression(
            expr_target,
            px_rate,
            px_r,
            px_dropout,
        )

        if mask_pro.sum().gt(0):
            py_ = generative_outputs["py_"]
            protein_target = _expand_to_match_mc(y, py_["rate_back"])
            rl_protein = get_reconstruction_loss_protein(protein_target, py_, None)
        else:
            rl_protein = torch.zeros_like(rl_expression)

        recon_loss_expression = rl_expression * mask_expr
        recon_loss_accessibility = rl_accessibility * mask_acc
        recon_loss_protein = rl_protein * mask_pro
        recon_loss = recon_loss_expression + recon_loss_accessibility + recon_loss_protein

        qz_m = inference_outputs["qz_m"]
        qz_v = inference_outputs["qz_v"]
        # NOTE: This kl_div_z is a *preliminary* placeholder using the pre-hierarchical
        # KL(q_u, N(0,1)) via the replaced qz_m/qz_v.  Both this value and the `loss`
        # computed below are discarded by the calling loss() method, which recomputes
        # them using kl_u + kl_z after the two-level KL is available.  It is kept here
        # to satisfy LossOutput's interface contract.
        kl_div_z = kl_divergence(Normal(qz_m, torch.sqrt(qz_v)), Normal(0, 1)).sum(dim=1)
        kl_div_paired = self._compute_mod_penalty(
            (inference_outputs["qzm_expr"], inference_outputs["qzv_expr"]),
            (inference_outputs["qzm_acc"], inference_outputs["qzv_acc"]),
            (inference_outputs["qzm_pro"], inference_outputs["qzv_pro"]),
            mask_expr,
            mask_acc,
            mask_pro,
        )

        loss = torch.mean(recon_loss + kl_weight * kl_div_z + kl_div_paired)
        recon_losses = {
            "reconstruction_loss_expression": recon_loss_expression,
            "reconstruction_loss_accessibility": recon_loss_accessibility,
            "reconstruction_loss_protein": recon_loss_protein,
        }
        kl_local = {
            "kl_divergence_z": kl_div_z,
            "kl_divergence_paired": kl_div_paired,
        }
        extra_metrics = (
            {
                "z": inference_outputs["z"],
                "batch": tensors[REGISTRY_KEYS.BATCH_KEY],
                "labels": tensors[REGISTRY_KEYS.LABELS_KEY],
            }
            if self.extra_payload_autotune
            else {}
        )
        return LossOutput(
            loss=loss,
            reconstruction_loss=recon_losses,
            kl_local=kl_local,
            extra_metrics=extra_metrics,
        )

    # ------------------------------------------------------------------
    # Loss: two-level KL (kl_u + kl_z)
    # ------------------------------------------------------------------

    def loss(
        self,
        tensors: dict[str, torch.Tensor],
        inference_outputs: dict[str, torch.Tensor],
        generative_outputs: dict[str, torch.Tensor],
        kl_weight: float = 1.0,
    ) -> LossOutput:
        """Extend MULTIVAE's loss with the second-level KL for the sample residual.

        MULTIVAE's ``kl_divergence_z`` term is repurposed as ``kl_u =
        KL(q_u \\| p_u)`` where ``p_u`` is the configured prior (default: a
        learned mixture-of-Gaussians, see :attr:`u_prior_mixture`).  This
        method adds:

        .. math::
            kl_z = -\\log p(\\varepsilon) = -\\log N(0,\\; \\exp(\\text{pz\\_scale}))(\\varepsilon)

        and folds it into both ``loss`` and ``kl_local["kl_divergence_z"]``.

        Note
        ----
        The MULTIVAE key is ``"kl_divergence_z"`` (not ``"kl_div_z"`` which is
        the TOTALVAE key used in :class:`~.MrTotalVAE`).
        """
        if "eps" in inference_outputs and inference_outputs["z"].ndim > 2:
            loss_out = self._loss_with_mc_samples(
                tensors,
                inference_outputs,
                generative_outputs,
                kl_weight,
            )
        else:
            loss_out = super().loss(tensors, inference_outputs, generative_outputs, kl_weight)

        if "eps" not in inference_outputs:
            # Hierarchy not built (n_sample=0 placeholder).
            return loss_out

        label_index = tensors[REGISTRY_KEYS.LABELS_KEY]
        kl_u = self.kl_u(inference_outputs["qu"], inference_outputs["u"], label_index)

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
            if kl_z.ndim > 1:
                kl_z = kl_z.mean(dim=0)
        else:
            kl_z = torch.zeros_like(kl_u)

        # kl_divergence_z now = kl_u + kl_z
        kl_local = dict(loss_out.kl_local)
        kl_local["kl_divergence_z"] = kl_u + kl_z

        if self._scale_observations and self.n_obs_per_sample is not None:
            # Weight each cell's ELBO by 1/n_cells_in_that_sample.
            # Reconstruct per-cell ELBO from component tensors (parent already called mean()).
            sample_index = tensors[REGISTRY_KEYS.SAMPLE_KEY].to(torch.int64).flatten()
            prefactors = self.n_obs_per_sample[sample_index]  # (batch,)
            recon = (
                loss_out.reconstruction_loss["reconstruction_loss_expression"]
                + loss_out.reconstruction_loss["reconstruction_loss_accessibility"]
                + loss_out.reconstruction_loss["reconstruction_loss_protein"]
            )
            per_cell = recon + kl_weight * kl_local["kl_divergence_z"] + kl_local["kl_divergence_paired"]
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
            recon = (
                loss_out.reconstruction_loss["reconstruction_loss_expression"]
                + loss_out.reconstruction_loss["reconstruction_loss_accessibility"]
                + loss_out.reconstruction_loss["reconstruction_loss_protein"]
            )
            loss = (recon + kl_weight * kl_local["kl_divergence_z"] + kl_local[
                "kl_divergence_paired"
            ]).mean()

        return LossOutput(
            loss=loss,
            reconstruction_loss=loss_out.reconstruction_loss,
            kl_local=kl_local,
            extra_metrics=loss_out.extra_metrics,
        )

    @torch.inference_mode()
    def compute_h_from_x_eps(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        sample_index: torch.Tensor,
        batch_index: torch.Tensor,
        extra_eps: torch.Tensor,
        cont_covs: torch.Tensor | None = None,
        cat_covs: torch.Tensor | None = None,
        label: torch.Tensor | None = None,
        cell_idx: torch.Tensor | None = None,
        size_factor: torch.Tensor | None = None,
        cf_sample: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return ``concat([px_scale, py_scale_det])`` at ``z = z_base + extra_eps``.

        This is the decoder hook consumed by the LFC block in
        :func:`~scvi.external.mrtotalvi._stats.differential_expression` when
        ``store_lfc=True``.  It mirrors the analogous hook in
        :class:`~scvi.external.mrtotalvi.MrTotalVAE` but handles the
        MULTIVAE generative signature (``z``, ``qz_m``, ``libsize_expr``) and
        skips the protein path when ``n_input_proteins == 0``.

        Parameters
        ----------
        x
            Expression count matrix, shape ``(batch, n_genes)``.
        y
            Protein count matrix, shape ``(batch, n_proteins)``.  Pass a
            zero-filled tensor when no proteins are present.
        sample_index
            Integer donor index, shape ``(batch, 1)`` or ``(batch,)``.
        batch_index
            Batch integer index for the counterfactual decode, shape
            ``(batch, 1)``.
        extra_eps
            Counterfactual eps shift, shape ``(batch, n_latent)``.
        cont_covs
            Continuous covariate tensor.
        cat_covs
            Categorical covariate tensor.
        label
            Cell-type label tensor.  Defaults to zeros when ``None``.
        cell_idx
            Cell index tensor (passed through to MULTIVAE inference).
            Defaults to ``arange(n_cells)`` when ``None``.
        size_factor
            Optional size factor tensor.
        cf_sample
            Counterfactual sample override for the eps encoder.

        Returns
        -------
        :class:`torch.Tensor`
            Shape ``(batch, n_genes)`` when no proteins, or
            ``(batch, n_genes + n_proteins)`` when proteins are present.
            The ``gene``/``protein`` coordinate split is applied at the
            model level (B3) when assembling the xarray output.
        """
        n_cells = x.shape[0]
        if label is None:
            label = torch.zeros(n_cells, 1, dtype=torch.long, device=x.device)
        if cell_idx is None:
            cell_idx = torch.arange(n_cells, device=x.device)

        # Run inference (single MC draw; the MC loop lives in _stats.py).
        out = self.inference(
            x,
            y,
            batch_index,
            cont_covs=cont_covs,
            cat_covs=cat_covs,
            label=label,
            cell_idx=cell_idx,
            size_factor=size_factor,
            n_samples=1,
            sample_index=sample_index,
            cf_sample=cf_sample,
        )

        # Use the posterior MEAN of u for a deterministic z_base (mirrors MRVI).
        # out["z_base"] was computed from qu.rsample(), which changes each call.
        # Recomputing from qu.mean gives a fixed anchor for the LFC contrast.
        qu = out["qu"]
        sample_index_cf = sample_index if cf_sample is None else cf_sample
        z_base, _, _ = self.qz(qu.mean, sample_index_cf)  # deterministic
        z = z_base + extra_eps           # counterfactual latent
        qz_m = out["qz_m"]               # (batch, n_latent) — deterministic u mean
        libsize_expr = out["libsize_expr"]

        gen_out = self.generative(
            z=z,
            qz_m=qz_m,
            batch_index=batch_index,
            cont_covs=cont_covs,
            cat_covs=cat_covs,
            libsize_expr=libsize_expr,
            use_z_mean=False,
            label=label,
        )

        # RNA: px_scale is deterministic (softmax over the z-dependent NB decoder).
        # Note: MULTIVAE's generative returns px_scale at the top level, not inside px_.
        px_scale = gen_out["px_scale"]   # (batch, n_genes)

        if self.n_input_proteins == 0:
            return px_scale

        # Protein: deterministic background reconstruction (D-021).
        # DecoderADT calls Normal(back_alpha, back_beta).rsample() internally, so
        # py_["scale"] is stochastic.  Using that draw in both x_1 and x_0 introduces
        # independent background noise that does NOT cancel in the LFC.  Fix: use the
        # log-mean back_alpha directly as a deterministic background stand-in.
        py_ = gen_out["py_"]
        rate_back_det = torch.exp(py_["back_alpha"])          # (batch, n_proteins)
        rate_fore_det = rate_back_det * py_["fore_scale"]     # fore_scale is deterministic
        # py_["mixing"] is a raw logit; sigmoid → mixing probability.
        protein_mixing_det = torch.sigmoid(py_["mixing"])
        py_scale_det = F.normalize(
            (1.0 - protein_mixing_det) * rate_fore_det, p=1, dim=-1
        )  # (batch, n_proteins)

        return torch.cat([px_scale, py_scale_det], dim=-1)   # (batch, n_genes + n_proteins)
