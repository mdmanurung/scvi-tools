"""Continual-learning utilities for CytoANVI case-control atlas building.

Ports the approach of ``theislab/comparative_atlas`` (``cscanvi``) to CytoANVI and modern
scvi-tools: regularized incremental query updates that anchor to a reference (and optionally
healthy controls) via an Elastic-Weight-Consolidation (EWC) penalty whose Fisher-style parameter
importances are estimated from a replay buffer of reference cells, plus test-time-augmentation
(TTA) Bregman-Information uncertainty for novelty detection.

The replay buffer and controls are used ONLY to estimate importances at surgery time; at training
time the query batch flows through the usual ELBO with the added EWC penalty (no data replay into
the minibatch), matching the cscanvi implementation.
"""

from __future__ import annotations

import torch

from scvi.train import SemiSupervisedTrainingPlan


def mask_augment(x: torch.Tensor, mask_percentage: float = 0.5) -> torch.Tensor:
    """Randomly zero a fixed fraction of features per cell (shared mask across the batch).

    Default 0.5 matches the paper's TTA (mask 50% of genes per perturbation).
    """
    _, feature_dim = x.shape
    num_masked = int(mask_percentage * feature_dim)
    mask = torch.cat(
        [
            torch.ones(num_masked, dtype=torch.bool),
            torch.zeros(feature_dim - num_masked, dtype=torch.bool),
        ]
    )
    mask = mask[torch.randperm(mask.size(0))]
    mask = mask.unsqueeze(0).expand(x.shape[0], -1).to(x.device)
    return x * mask


def bregman_information_lse(zs: torch.Tensor, axis: int = 0, class_axis: int = -1) -> torch.Tensor:
    """Bregman Information of ``zs`` under the log-sum-exp generator.

    ``BI[Z] = E[LSE(Z)] - LSE(E[Z])``, estimated over the sample ``axis``. Taken from
    https://github.com/MLO-lab/Uncertainty_Estimates_via_BVD. Higher = more uncertain / novel.
    """
    e_of_lse = zs.logsumexp(axis=class_axis).mean(axis)
    lse_of_e = zs.mean(axis).unsqueeze(axis).logsumexp(axis=class_axis).squeeze(axis)
    return e_of_lse - lse_of_e


def compute_uncertainty_scores(inference_inputs: dict, module, tta_rep: int = 10) -> torch.Tensor:
    """Per-cell Bregman-Information uncertainty over ``tta_rep`` mask-augmented latent draws."""
    input_x = inference_inputs["x"]
    with torch.no_grad():
        module.eval()
        all_zs = []
        for _ in range(tta_rep):
            aug_inputs = dict(inference_inputs)
            aug_inputs["x"] = mask_augment(input_x)
            all_zs.append(module.inference(**aug_inputs)["z"])
        zs_out = torch.stack(all_zs).detach().cpu()  # tta_rep x batch x n_latent
    return bregman_information_lse(zs_out)


def zerolike_params_dict(module: torch.nn.Module) -> list[tuple[str, torch.Tensor]]:
    """``[(name, zeros_like(param))]`` for trainable params (Fisher accumulator init)."""
    return [(k, torch.zeros_like(p)) for k, p in module.named_parameters() if p.requires_grad]


class CytoANVIContinualTrainingPlan(SemiSupervisedTrainingPlan):
    """Semi-supervised training plan for continual case-control updates (paper-faithful).

    Implements the paper's loss ``L(theta_query) = ELBO(x_query, x_replay) + (lambda/2) F
    (theta_query - theta_ref)^2``:

    - the query minibatch flows through ``module._replay_forward`` (ELBO + the EWC penalty,
      weighted by ``ewc_importance`` = lambda), and
    - a replay-buffer minibatch (reference cells stored on the module by
      ``CytoANVI.load_query_data_with_replay``) is rehearsed each step by adding its plain ELBO
      (Experience Replay). The replay batches cycle by ``batch_idx``.
    """

    def __init__(self, module, n_classes: int, *, ewc_importance: float = 1.0, **kwargs):
        super().__init__(module, n_classes, **kwargs)
        self.loss_kwargs.update({"ewc_importance": ewc_importance})

    def forward(self, *args, **kwargs):
        """Route the forward pass through the module's replay/EWC forward (ELBO + EWC penalty)."""
        return self.module._replay_forward(*args, **kwargs)

    def _next_replay_batch(self, batch_idx: int):
        """Cycle through the stored replay-buffer minibatches, moved to the module device."""
        batches = getattr(self.module, "_replay_batches", None)
        if not batches:
            return None
        rb = batches[batch_idx % len(batches)]
        return {k: v.to(self.module.device) for k, v in rb.items()}

    def training_step(self, batch, batch_idx):
        """Query ELBO + EWC penalty, plus the rehearsed replay-buffer ELBO."""
        if len(batch) == 2:
            full_dataset, labelled_dataset = batch[0], batch[1]
        else:
            full_dataset, labelled_dataset = batch, None

        if "kl_weight" in self.loss_kwargs:
            self.loss_kwargs.update({"kl_weight": self.kl_weight})
        input_kwargs = {"labelled_tensors": labelled_dataset}
        input_kwargs.update(self.loss_kwargs)

        # query minibatch: ELBO + EWC penalty
        _, _, loss_output = self.module._replay_forward(full_dataset, loss_kwargs=input_kwargs)
        loss = loss_output.loss

        # experience replay: add the plain ELBO of a replay-buffer minibatch (no EWC)
        replay_batch = self._next_replay_batch(batch_idx)
        if replay_batch is not None:
            _, _, replay_out = self.module(replay_batch, loss_kwargs={"kl_weight": self.kl_weight})
            loss = loss + replay_out.loss

        self.log("train_loss", loss, on_epoch=True, batch_size=loss_output.n_obs_minibatch)
        self.compute_and_log_metrics(loss_output, self.train_metrics, "train")
        return loss
