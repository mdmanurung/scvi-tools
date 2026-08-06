# Cytometry

```{toctree}
:maxdepth: 1

notebooks/cytometry/CytoANVI_tutorial
notebooks/cytometry/CytoANVI_treeArches_tutorial
parameter_selection/CytoANVI_parameter_selection
```

```{customcard}
:path: notebooks/cytometry/CytoANVI_tutorial
:tags: Integration, Analysis, Cytometry, Label-transfer

Semi-supervised label transfer and panel-aware query mapping with CytoANVI
```

```{customcard}
:path: notebooks/cytometry/CytoANVI_treeArches_tutorial
:tags: Integration, Analysis, Cytometry, Hierarchy, Template

Template for learning and updating cell-type hierarchies on CytoANVI latents with scHPL
```

```{customcard}
:path: parameter_selection/CytoANVI_parameter_selection
:tags: Cytometry, Analysis, Hyperparameters

Choose CytoANVI parameters using evidence-graded guidance, including which defaults are unvalidated
```

## Runnable scripts

```{note}
**CytoANVI end-to-end reference → query example (runnable Python script)**

`docs/tutorials/notebooks/cytometry/cytoanvi_example_reference_query.py` — a self-contained script
demonstrating same-panel and panel-divergent CytoANVI reference→query mapping with label transfer
on synthetic data. Run with:

    PYTHONPATH=src:. python docs/tutorials/notebooks/cytometry/cytoanvi_example_reference_query.py
```
