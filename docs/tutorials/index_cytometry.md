# Cytometry

```{toctree}
:maxdepth: 1

parameter_selection/CytoANVI_parameter_selection
```

```{customcard}
:path: parameter_selection/CytoANVI_parameter_selection
:tags: Cytometry, Analysis, Hyperparameters

Choose CytoANVI parameters using evidence-graded guidance, including which defaults are unvalidated
```

## Runnable scripts

```{note}
**CytoANVI end-to-end reference → query example (runnable Python script)**

`vignettes/cytoanvi_example_reference_query.py` — a self-contained script
demonstrating same-panel and panel-divergent CytoANVI reference→query mapping with label transfer
on synthetic data. Run with:

    PYTHONPATH=src:. python vignettes/cytoanvi_example_reference_query.py

`vignettes/cytoanvi_treearches_synthetic.py` exercises direct same-panel surgery and one-shot
learn/update/predict with a deterministic fake scHPL backend. These are engineering fixtures, not
scientific validation. The richer notebook directory is an uninitialized external gitlink; its
required repair is recorded as `blocked_external_submodule` in the usage-readiness packet.
```
