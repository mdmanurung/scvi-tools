"""Shared test utilities for tests/external sub-packages."""


def get_elbo_key(history: dict) -> str:
    """Return whichever ELBO key Lightning populated in this training run.

    The key name varies across Lightning versions and scvi-tools training plan
    configurations.  Raises AssertionError with a diagnostic message if none
    of the known keys is present.
    """
    key = next(
        (k for k in ("elbo_train", "train_loss_epoch", "train_elbo_train") if k in history),
        None,
    )
    assert key is not None, (
        f"No ELBO history key found. Available keys: {list(history.keys())}"
    )
    return key
