"""Versioned latent-integrity policies for prospective RDX-03 runs."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path

_POLICY_PATH = Path(__file__).with_name("latent_integrity_policy_v2.json")
_LATENT_INTEGRITY_POLICY_V2 = json.loads(
    _POLICY_PATH.read_text(encoding="utf-8")
)


def latent_integrity_policy_v2() -> dict[str, object]:
    """Return a detached copy of the complete prospective policy."""
    return deepcopy(_LATENT_INTEGRITY_POLICY_V2)


def latent_integrity_policy_digest_v2() -> str:
    """Return the SHA-256 digest of canonical policy JSON."""
    canonical = json.dumps(
        _LATENT_INTEGRITY_POLICY_V2,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return hashlib.sha256(canonical).hexdigest()


def validate_latent_integrity_policy_v2(
    payload: dict[str, object],
) -> dict[str, object]:
    """Validate and detach the complete canonical prospective policy."""
    if payload != _LATENT_INTEGRITY_POLICY_V2:
        raise ValueError(
            "Latent-integrity policy does not match the canonical v2 policy."
        )
    return deepcopy(payload)
