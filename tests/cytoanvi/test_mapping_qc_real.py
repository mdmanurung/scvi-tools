"""mapping_qc tests that exercise the **real** installed mapqc package.

Every other mapping-QC test (``test_mapping_qc_mock.py``) monkeypatches mapqc away, so nothing
verifies that :func:`cytoanvi.mapping_qc._patch_mapqc_empty_mode` actually mutates the real
module, nor that the mutation actually prevents the upstream crash it exists for. That workaround
is what unblocks the B9 benchmark (see F-039), so it deserves signal against the dependency it
patches rather than against a stub.

Skipped cleanly when mapqc is absent; runs for real in the ``scvi-test`` environment, where
``mapqc==0.1.1`` is installed.
"""

import importlib

import numpy as np
import pytest

from cytoanvi import mapping_qc

pytest.importorskip("mapqc")


def _empty_mode_inputs():
    """The exact B9-crash input shape: every neighbourhood passes, so ``mode()`` is 0-row.

    Mirrors ``test_patched_get_per_cell_filtering_info_guards_empty_mode`` in the mock suite so
    both tests fail together if the reproduction itself ever stops being representative.
    """
    import pandas as pd

    n_nhoods, n_cells = 4, 6
    scores = np.array([0.5, np.nan, 1.2, np.nan, np.nan, np.nan])
    mask = np.zeros((n_nhoods, n_cells), dtype=int)
    mask[0, [0, 1]] = 1
    mask[1, [2, 3]] = 1
    mask[2, [4]] = 1  # cell 5 belongs to zero neighbourhoods
    nhood_info = pd.DataFrame({"filter_info": [None] * n_nhoods})
    return scores, mask, nhood_info


def test_real_mapqc_runtime_matches_exact_guard():
    mapping_qc._require_mapqc()


def test_patch_applies_to_real_module_and_prevents_indexerror():
    """The guard must patch the real mapqc module and actually stop the real ``IndexError``.

    Three things are asserted that a mocked test structurally cannot:

    1. **Canary** — the *unpatched* upstream function still raises ``IndexError`` on this input.
       If mapqc ever ships a fix, this assertion fails loudly, which is the signal that
       ``_patch_mapqc_empty_mode`` may now be removable. A failure here is news, not a defect.
    2. The patch rebinds the attribute *on the real module object* that mapqc resolves at call
       time — not merely that our replacement function works in isolation.
    3. Calling through the real module attribute afterwards no longer raises.
    """
    import mapqc._mapqc_scores as _ms

    # Guarantee a pristine, unpatched module regardless of what ran earlier in this session.
    # `_patch_mapqc_empty_mode` is idempotent-by-identity, so an already-patched module from a
    # prior test would silently invalidate the canary below.
    _ms = importlib.reload(_ms)
    try:
        original = _ms._get_per_cell_filtering_info
        assert not getattr(original, "_cytoanvi_guarded", False), (
            "module was already patched after reload; the reload-to-pristine assumption is broken"
        )

        scores, mask, nhood_info = _empty_mode_inputs()

        # 1. Canary: the upstream bug this workaround exists for is still present.
        with pytest.raises(IndexError):
            original(scores, mask, nhood_info)

        # 2. The patch rebinds the real module attribute.
        mapping_qc._patch_mapqc_empty_mode()
        assert _ms._get_per_cell_filtering_info is (
            mapping_qc._patched_get_per_cell_filtering_info
        )

        # 3. Calling through the real module no longer raises, and still labels cells correctly.
        out = _ms._get_per_cell_filtering_info(scores, mask, nhood_info)
        assert out[0] == "pass"
        assert out[2] == "pass"
        assert out[5] == "not sampled"
    finally:
        # Never leave the installed package mutated for the rest of the session.
        importlib.reload(_ms)


def test_patch_is_idempotent_by_identity():
    """A second patch call must leave the *same object* in place, not re-wrap it.

    The existing mock-suite test only shows the installer doesn't raise when called twice. That
    passes even for an implementation that re-wraps on every call, which would nest wrappers
    without bound. The ``_cytoanvi_guarded`` early-return is the real contract, and identity is
    the only assertion that pins it.
    """
    import mapqc._mapqc_scores as _ms

    _ms = importlib.reload(_ms)
    try:
        mapping_qc._patch_mapqc_empty_mode()
        after_first = _ms._get_per_cell_filtering_info

        mapping_qc._patch_mapqc_empty_mode()
        after_second = _ms._get_per_cell_filtering_info

        assert after_second is after_first
    finally:
        importlib.reload(_ms)
