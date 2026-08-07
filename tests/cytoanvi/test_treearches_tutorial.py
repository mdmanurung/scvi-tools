from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import numpy as np

MODULE_PATH = Path(__file__).parents[2] / "vignettes/cytoanvi_treearches_synthetic.py"
SPEC = spec_from_file_location("cytoanvi_treearches_synthetic", MODULE_PATH)
assert SPEC is not None
assert SPEC.loader is not None
TUTORIAL = module_from_spec(SPEC)
SPEC.loader.exec_module(TUTORIAL)


def test_direct_same_panel_tutorial_executes() -> None:
    result = TUTORIAL.run_direct_same_panel()
    assert result["predictions"].shape == (result["query"].n_obs,)
    assert result["latent"].shape == (result["query"].n_obs, 4)
    assert np.isfinite(result["latent"]).all()


def test_one_shot_learn_update_predict_tutorial_executes() -> None:
    result = TUTORIAL.run_one_shot_learn_update_predict()
    assert result["learned_tree"].revision == 1
    assert result["updated_tree"].revision == 2
    assert set(result["tree_predictions"]) == {"synthetic-tree-r2"}
