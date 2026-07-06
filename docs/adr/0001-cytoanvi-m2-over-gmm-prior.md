# CytoANVI uses the scANVI M1+M2 architecture and disables CytoVI's mixture prior

CytoANVI is a semi-supervised extension of CytoVI (analogous to how scANVI extends scVI). CytoVI
already injects label structure through a **label-conditioned mixture-of-Gaussians prior on z**
(`prior_mixture`, `prior_label_weight`; `cytovi/_module.py:296-318`), whereas
scANVI/totalANVI inject it through an **M1+M2 latent hierarchy** (`encoder_z2_z1`/`decoder_z1_z2`
+ classifier marginalization) that assumes a plain N(0,I) prior and recomputes the z1 prior
inside the loss (`module/_scanvae.py:194-230`, `totalanvi/_module.py:318-409`). A model cannot
cleanly use both: the M2 loss ignores `generative_outputs["pz"]`, so an active GMM prior would be
dead weight at best and double-count label shaping at worst.

**Decision:** `CytoANVAE(SupervisedModuleClass, CytoVAE)` mirrors `TOTALANVAE` — an M1+M2 port
reusing CytoVI's protein likelihood (Normal/Beta) and `nan_layer` masking for reconstruction —
and **forces `prior_mixture=False`** in the semi-supervised path. CytoVI's mixture prior remains
available in plain `CYTOVI`.

## Considered options

- **M1+M2, GMM off (chosen).** Smallest faithful delta; matches the tested `TOTALANVI` sibling;
  and is the structure the Phase 2 `theislab/comparative_atlas` (`cscanvi`) continual-update
  machinery (Bregman replay, Fisher importances, frozen-encoder surgery) assumes.
- **Keep GMM prior, add a classifier head only (no M2).** Smaller code, but diverges from the
  scANVI/totalANVI architecture, leaves no z1→z2 marginalization, and would not align with the
  Phase 2 `cscanvi` port. Rejected.
- **Keep GMM prior *and* add M2.** Most expressive but double-counts label shaping, is novel and
  unvalidated, and is hard to reason about. Rejected.

## Consequences

- A future reader sees `prior_mixture=False` forced inside `CytoANVAE.__init__` and the CytoVI
  GMM code apparently unused on this path — this ADR records that this is deliberate, not an
  oversight.
- Annotation-aware latent shaping in CytoANVI comes from the M2 hierarchy + classifier, not the
  GMM prior.
