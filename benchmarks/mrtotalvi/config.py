"""Frozen MrTotalVI C0-C4 benchmark configurations."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Literal


@dataclass(frozen=True)
class CandidateConfig:
    """One preregistered MrTotalVI candidate."""

    name: Literal["C0", "C1", "C2", "C3", "C4"]
    u_prior: Literal["mog", "vamp"]
    init_prior_from_data: bool
    freeze_prior_after_init: bool
    hierarchy_mode: Literal["legacy", "centered_v2"]
    u_encoder_mode: Literal["sample_conditioned", "sample_blind"]
    scale_observations: bool
    scientific_role: str

    def model_axes(self) -> dict[str, str | bool]:
        """Return only axes that may differ in the frozen factorial."""
        values = asdict(self)
        return {
            key: values[key]
            for key in (
                "u_prior",
                "init_prior_from_data",
                "freeze_prior_after_init",
                "hierarchy_mode",
                "u_encoder_mode",
                "scale_observations",
            )
        }


def candidate_configs() -> dict[str, CandidateConfig]:
    """Return C0-C4 in frozen evaluation order."""
    common = {
        "u_prior": "vamp",
        "init_prior_from_data": True,
        "freeze_prior_after_init": True,
        "hierarchy_mode": "legacy",
        "u_encoder_mode": "sample_conditioned",
        "scale_observations": False,
    }
    return {
        "C0": CandidateConfig(
            name="C0",
            u_prior="mog",
            init_prior_from_data=False,
            freeze_prior_after_init=False,
            hierarchy_mode="legacy",
            u_encoder_mode="sample_conditioned",
            scale_observations=False,
            scientific_role="accepted baseline",
        ),
        "C1": CandidateConfig(
            name="C1",
            **common,
            scientific_role="unvalidated baseline candidate",
        ),
        "C2": CandidateConfig(
            name="C2",
            **(common | {"hierarchy_mode": "centered_v2"}),
            scientific_role="latent-decoding diagnostic",
        ),
        "C3": CandidateConfig(
            name="C3",
            **(
                common
                | {
                    "hierarchy_mode": "centered_v2",
                    "scale_observations": True,
                }
            ),
            scientific_role="sample-balance diagnostic",
        ),
        "C4": CandidateConfig(
            name="C4",
            **(
                common
                | {
                    "hierarchy_mode": "centered_v2",
                    "u_encoder_mode": "sample_blind",
                }
            ),
            scientific_role="only primary-DA-eligible v2 candidate",
        ),
    }
