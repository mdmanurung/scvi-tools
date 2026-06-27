from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import torch
import torch.nn.functional as F
from torch.distributions import Categorical, Normal
from torch.distributions import kl_divergence as kl

from scvi import REGISTRY_KEYS
from scvi.external.cytovi._constants import CYTOVI_REGISTRY_KEYS
from scvi.external.cytovi._module import CytoVAE
from scvi.module._classifier import Classifier
from scvi.module._utils import broadcast_labels
from scvi.module.base import LossOutput, SupervisedModuleClass, auto_move_data
from scvi.nn import Decoder, Encoder

from ._continual import ContinualUpdate
from ._hce import hierarchical_cross_entropy_loss

_NON_DATA_INFERENCE_KEYS = frozenset({"batch_index", "cont_covs", "cat_covs", "panel_index"})

if TYPE_CHECKING:
    from torch.distributions import Distribution


class CytoANVAE(SupervisedModuleClass, CytoVAE):
    """Semi-supervised variational auto-encoder for cytometry (CytoANVI).

    Combines the CytoVI protein-intensity model of :cite:p:`Ingelfinger25` with the
    scANVI/M1+M2 semi-supervised objective of :cite:p:`Xu21`. The CytoVI encoder, decoder
    and protein-specific likelihood (Normal/Beta) are reused unchanged for reconstruction; a
    :class:`~scvi.module.Classifier` head is added on the shared latent ``z1`` together with the
    M1+M2 latent hierarchy (``encoder_z2_z1`` / ``decoder_z1_z2``) and a label-marginalized ELBO.

    CytoVI's label-conditioned mixture-of-Gaussians prior is **disabled** here
    (``prior_mixture=False``): the M1+M2 hierarchy supplies the ``z1`` prior, so leaving the
    mixture prior active would double-count label structure. See the package ADR for details.

    Parameters
    ----------
    n_input
        Number of input proteins.
    n_batch
        Number of batches. Default is 0.
    n_labels
        Number of (observed) cell-type labels, excluding the unlabeled category. Default is 0.
    n_hidden
        Number of nodes per hidden layer. Default is 128.
    n_latent
        Dimensionality of the latent space. Default is 10.
    n_layers
        Number of hidden layers used for encoder and decoder NNs. Default is 1.
    dropout_rate
        Dropout rate for the encoder/decoder and the classifier / ``z2`` head. Default is 0.1.
    y_prior
        Prior over labels of shape ``(1, n_labels)``. If ``None``, uniform.
    linear_classifier
        If ``True``, uses a single linear layer for classification instead of an MLP.
    classifier_parameters
        Keyword arguments passed into :class:`~scvi.module.Classifier`.
    use_batch_norm
        Whether to use batch norm in layers. Default is "both".
    use_layer_norm
        Whether to use layer norm in layers. Default is "none".
    **cytovae_kwargs
        Keyword args for :class:`~scvi.external.cytovi.CytoVAE`. ``prior_mixture`` /
        ``prior_mixture_k`` are accepted for signature compatibility but forced off.
    """

    def __init__(
        self,
        n_input: int,
        n_batch: int = 0,
        n_labels: int = 0,
        n_hidden: int = 128,
        n_latent: int = 10,
        n_layers: int = 1,
        dropout_rate: float = 0.1,
        y_prior: torch.Tensor | None = None,
        linear_classifier: bool = False,
        classifier_parameters: dict | None = None,
        use_batch_norm: Literal["encoder", "decoder", "none", "both"] = "both",
        use_layer_norm: Literal["encoder", "decoder", "none", "both"] = "none",
        # accepted for compatibility with CytoVAE/CYTOVI but always forced off (see ADR)
        prior_mixture: bool | None = None,
        prior_mixture_k: int | None = None,
        reachability_matrix: torch.Tensor | None = None,
        **cytovae_kwargs,
    ):
        super().__init__(
            n_input,
            n_batch=n_batch,
            n_labels=n_labels,
            n_hidden=n_hidden,
            n_latent=n_latent,
            n_layers=n_layers,
            dropout_rate=dropout_rate,
            use_batch_norm=use_batch_norm,
            use_layer_norm=use_layer_norm,
            prior_mixture=False,
            **cytovae_kwargs,
        )

        self.n_labels = n_labels
        classifier_parameters = classifier_parameters or {}
        use_batch_norm_encoder = use_batch_norm == "encoder" or use_batch_norm == "both"
        use_layer_norm_encoder = use_layer_norm == "encoder" or use_layer_norm == "both"
        use_batch_norm_decoder = use_batch_norm == "decoder" or use_batch_norm == "both"
        use_layer_norm_decoder = use_layer_norm == "decoder" or use_layer_norm == "both"

        # Classifier takes the shared latent z1 as input (valid under missing-marker encoding,
        # since the classifier never sees raw markers, only post-encoder z1).
        cls_parameters = {
            "n_layers": 0 if linear_classifier else n_layers,
            "n_hidden": 0 if linear_classifier else n_hidden,
            "dropout_rate": dropout_rate,
            "logits": True,
        }
        cls_parameters.update(classifier_parameters)
        self.classifier = Classifier(
            n_latent,
            n_labels=n_labels,
            use_batch_norm=use_batch_norm_encoder,
            use_layer_norm=use_layer_norm_encoder,
            **cls_parameters,
        )

        self.encoder_z2_z1 = Encoder(
            n_latent,
            n_latent,
            n_cat_list=[self.n_labels],
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            use_batch_norm=use_batch_norm_encoder,
            use_layer_norm=use_layer_norm_encoder,
            return_dist=True,
        )

        self.decoder_z1_z2 = Decoder(
            n_latent,
            n_latent,
            n_cat_list=[self.n_labels],
            n_layers=n_layers,
            n_hidden=n_hidden,
            use_batch_norm=use_batch_norm_decoder,
            use_layer_norm=use_layer_norm_decoder,
        )

        if n_labels < 1:
            raise ValueError(
                "CytoANVAE requires n_labels >= 1 (at least one observed cell-type category "
                "excluding the unlabeled category)."
            )
        self.y_prior = torch.nn.Parameter(
            y_prior if y_prior is not None else (1 / n_labels) * torch.ones(1, n_labels),
            requires_grad=False,
        )

        # The configured continual case-control update, set by
        # CytoANVI.load_query_data_with_replay (or reattached in on_load). None = base path
        # (the EWC penalty contributes nothing), so the base model is unaffected.
        self.continual: ContinualUpdate | None = None

        reachability_tensor = (
            torch.as_tensor(reachability_matrix, dtype=torch.float32)
            if reachability_matrix is not None
            else None
        )
        self.register_buffer("reachability_matrix_", reachability_tensor)

    def _set_reachability(self, tensor: torch.Tensor | None) -> None:
        """Re-register the reachability buffer, keeping it in state_dict and device-aware."""
        if tensor is None:
            self.register_buffer("reachability_matrix_", None)
        else:
            self.register_buffer("reachability_matrix_", tensor.to(self.device))

    def loss(
        self,
        tensors: dict[str, torch.Tensor],
        inference_outputs: dict[str, torch.Tensor | Distribution | None],
        generative_outputs: dict[str, Distribution | None],
        kl_weight: float = 1.0,
        labelled_tensors: dict[str, torch.Tensor] | None = None,
        classification_ratio: float | None = None,
    ) -> LossOutput:
        """Compute the semi-supervised loss.

        Mirrors :meth:`~scvi.module.SCANVAE.loss` /
        :meth:`~scvi.external.totalanvi.TOTALANVAE.loss` but uses CytoVI's protein likelihood
        (``px``, Normal/Beta) for the reconstruction term and preserves the ``nan_layer`` mask
        for missing-marker / multi-panel data.
        """
        px: Distribution = generative_outputs["px"]
        qz1: Distribution = inference_outputs["qz"]
        z1: torch.Tensor = inference_outputs["z"]
        if z1.dim() != 2:
            raise ValueError(
                f"CytoANVAE.loss expects 2D z1 (n_samples==1); got shape {tuple(z1.shape)}."
            )
        x: torch.Tensor = tensors[CYTOVI_REGISTRY_KEYS.X_KEY]

        if CYTOVI_REGISTRY_KEYS.PROTEIN_NAN_MASK in tensors.keys():
            nan_mask = tensors[CYTOVI_REGISTRY_KEYS.PROTEIN_NAN_MASK]
        else:
            nan_mask = None

        # Reconstruction under CytoVI protein likelihood, masking unobserved markers.
        reconst_loss_int = -px.log_prob(x)
        if nan_mask is not None:
            reconst_loss = (reconst_loss_int * nan_mask).sum(-1)
        else:
            reconst_loss = reconst_loss_int.sum(-1)

        # Enumerate choices of label (M1 + M2 hierarchy on z1 -> z2).
        # NOTE: assumes z1 is 2D (n_samples == 1, as used during training); the
        # ``.view(n_labels, -1)`` reshape below is not valid for n_samples > 1.
        ys, z1s = broadcast_labels(z1, n_broadcast=self.n_labels)
        qz2, z2 = self.encoder_z2_z1(z1s, ys)
        pz1_m, pz1_v = self.decoder_z1_z2(z2, ys)

        mean = torch.zeros_like(qz2.loc)
        scale = torch.ones_like(qz2.scale)
        kl_divergence_z2 = kl(qz2, Normal(mean, scale)).sum(dim=-1)
        loss_z1_unweight = -Normal(pz1_m, torch.sqrt(pz1_v)).log_prob(z1s).sum(dim=-1)
        loss_z1_weight = qz1.log_prob(z1).sum(dim=-1)

        probs = self.classifier(z1)
        if self.classifier.logits:
            probs = F.softmax(probs, dim=-1)

        reconst_loss = (
            reconst_loss
            + loss_z1_weight
            + (loss_z1_unweight.view(self.n_labels, -1).t() * probs).sum(dim=-1)
        )

        kl_divergence = (kl_divergence_z2.view(self.n_labels, -1).t() * probs).sum(dim=-1)
        kl_divergence = kl_divergence + kl(
            Categorical(probs=probs),
            Categorical(probs=self.y_prior.repeat(probs.size(0), 1)),
        )

        loss = torch.mean(reconst_loss + kl_divergence * kl_weight)

        if labelled_tensors is not None:
            ce_loss, true_labels, logits = self.classification_loss(labelled_tensors)
            loss = loss + ce_loss * classification_ratio
            return LossOutput(
                loss=loss,
                reconstruction_loss=reconst_loss,
                kl_local=kl_divergence,
                classification_loss=ce_loss,
                true_labels=true_labels,
                logits=logits,
            )
        return LossOutput(
            loss=loss,
            reconstruction_loss=reconst_loss,
            kl_local=kl_divergence,
        )

    @auto_move_data
    def classification_loss(
        self, labelled_dataset: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        inference_inputs = self._get_inference_input(labelled_dataset)
        data_inputs = {
            key: inference_inputs[key]
            for key in inference_inputs.keys()
            if key not in _NON_DATA_INFERENCE_KEYS
        }

        y = labelled_dataset[REGISTRY_KEYS.LABELS_KEY]
        batch_idx = labelled_dataset[REGISTRY_KEYS.BATCH_KEY]
        cont_key = REGISTRY_KEYS.CONT_COVS_KEY
        cont_covs = labelled_dataset[cont_key] if cont_key in labelled_dataset.keys() else None

        cat_key = REGISTRY_KEYS.CAT_COVS_KEY
        cat_covs = labelled_dataset[cat_key] if cat_key in labelled_dataset.keys() else None

        logits = self.classify(
            **data_inputs, batch_index=batch_idx, cat_covs=cat_covs, cont_covs=cont_covs
        )
        y_long = y.view(-1).long()
        if self.reachability_matrix_ is None:
            ce_loss = F.cross_entropy(logits, y_long)
        else:
            ce_loss = hierarchical_cross_entropy_loss(
                logits, y_long, self.reachability_matrix_
            )
        return ce_loss, y, logits

    def on_load(self, model, **kwargs):
        """Reattach persisted continual state and hierarchy reachability, if any."""
        super().on_load(model, **kwargs)
        state = getattr(model, "continual_update_state_", None)
        if state is not None:
            self.continual = ContinualUpdate.from_persistable_state(state)
        hierarchy = getattr(model, "hierarchy_reachability_", None)
        if hierarchy is not None:
            self._set_reachability(torch.as_tensor(hierarchy, dtype=torch.float32))

    def loss_with_replay(self, tensors, inference_outputs, generative_outputs, loss_kwargs):
        """Standard CytoANVI loss plus the EWC penalty scaled by ``ewc_importance``."""
        loss_kwargs = dict(loss_kwargs or {})
        ewc_importance = loss_kwargs.pop("ewc_importance", 0.0)
        losses = self.loss(tensors, inference_outputs, generative_outputs, **loss_kwargs)
        # None continual = base path: the EWC penalty contributes nothing.
        penalty = self.continual.penalty(self) if self.continual is not None else 0.0
        # NOTE: ewc penalty is intentionally not placed in extra_metrics, as a non-empty
        # extra_metrics triggers the scib-autotune logging path (which expects z/batch/labels).
        return LossOutput(
            loss=losses.loss + ewc_importance * penalty,
            reconstruction_loss=losses.reconstruction_loss,
            kl_local=losses.kl_local,
            classification_loss=losses.classification_loss,
            true_labels=losses.true_labels,
            logits=losses.logits,
        )

    def _replay_forward(self, tensors, loss_kwargs=None):
        """Forward pass that adds the EWC penalty (used by the continual training plan)."""
        inference_inputs = self._get_inference_input(tensors)
        inference_outputs = self.inference(**inference_inputs)
        generative_inputs = self._get_generative_input(tensors, inference_outputs)
        generative_outputs = self.generative(**generative_inputs)
        losses = self.loss_with_replay(tensors, inference_outputs, generative_outputs, loss_kwargs)
        return inference_outputs, generative_outputs, losses
