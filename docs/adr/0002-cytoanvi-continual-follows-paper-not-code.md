# CytoANVI's continual update follows the paper, not the released cscanvi code

CytoANVI's Phase-2 continual case-control update ports the comparative-atlas (`cscanvi`) method
(bioRxiv 10.64898/2026.03.03.708171). When the released `theislab/comparative_atlas` **code** and
the **paper Methods** disagree, CytoANVI follows the **paper**.

The paper's loss (Methods, p.19) is
``L(theta_query) = ELBO(x_query, x_replay) + (lambda/2) (F_X_Reference ∘ F_X_QueryCtrl)
(theta_query - theta_ref)^2``:

- **Experience Replay** — the ELBO is computed on the query **and** a replay buffer of ~20%
  reference cells (rehearsal). The released code only uses the replay buffer to estimate Fisher
  importances and never rehearses it in the ELBO. **We implement the replay** (rehearse the buffer
  in the training step).
- **EWC weight = Hadamard product** of the reference-replay Fisher and the query-control Fisher
  (`F_reference ∘ F_query_ctrl`). We default `combine_type="product"` (the code exposed
  additive/product).
- **Query controls are required** — the term is `F_reference ∘ F_query_ctrl`, so query control
  cells must be supplied; controls must exist in both reference and query.
- **TTA masks 50%** of features for the Bregman-Information uncertainty (`get_uncertainty`); the
  earlier port used 15%.

## Consequences

- CytoANVI's continual update is **not** a line-for-line reproduction of the authors' released
  code — a reader comparing the two will see CytoANVI rehearse replay data and default to a product
  Fisher, which the code does not. This is deliberate (paper fidelity), recorded here.
- Mikhael's instruction stands: the primary source is the paper; the repo/README/code are not used
  as a substitute for it.
- `ewc_importance` (= lambda) and the replay-buffer / control sizes are tunable; defaults follow
  the paper's stated ranges (replay ~20%, query control ~5-10%).
