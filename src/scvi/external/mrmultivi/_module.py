"""MrMultiVAE — MULTIVAE with MrVI-style hierarchical donor latent space."""

from __future__ import annotations

import torch
import torch.nn as nn
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

    * ``kl_u = KL(q_u \\| N(0,1))`` — from the sample-conditioned u-encoder
      (replaces MULTIVAE's ``kl_divergence_z`` via ``qz_m``/``qz_v`` override).
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
        self.qz_kwargs = qz_kwargs or {}
        self.qu_kwargs = qu_kwargs or {}
        self._use_map = use_map
        self._scale_observations = scale_observations
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
        self.qu = EncoderXU_MultiVI(
            n_input=self.n_latent,
            n_latent=n_latent_u,
            n_sample=n_sample,
            n_batch=self.n_batch,
            n_continuous_cov=self.n_continuous_cov,
            n_cats_per_cov=self.n_cats_per_cov,
            encode_covariates=self.encode_covariates,
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

        # Sample-conditioned u-encoder: qu(u0, real_sample) → Normal(mu_u, sigma_u)
        qu_kwargs = {}
        if self.encode_covariates:
            qu_kwargs = {
                "batch_index": batch_index,
                "cont_covs": cont_covs,
                "cat_covs": cat_covs,
            }
        qu = self.qu(u0, sample_index, **qu_kwargs)
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

        MULTIVAE's ``kl_divergence_z = KL(q_u \\| N(0,1))`` is already
        ``kl_u`` verbatim (the encoder distribution is unchanged).  This
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
