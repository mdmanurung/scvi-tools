"""Opt-in MrTotalVI v2 package contracts."""

from __future__ import annotations

import hashlib
import shutil
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.distributions import Normal

import scvi
from scvi import REGISTRY_KEYS
from scvi.external import (
    MrTotalVI,
    combine_mrtotalvi_seed_results,
)
from tests.external.mrtotalvi.legacy_oracle.cpu_portability import (
    EXPECTED_CHECKSUM_MANIFEST_SHA256,
    EXPECTED_PORTABILITY_POLICY_SHA256,
    EXPECTED_SOURCE_COMMIT,
    PORTABILITY_POLICY_PATH,
    canonical_json_bytes,
    canonical_json_sha256,
    load_portability_policy,
    run_assessment_subprocess,
    verify_oracle_inventory,
)

ORACLE_ROOT = Path(__file__).with_name("legacy_oracle") / "d8c8e997"


def _make_adata(n_samples: int = 3):
    adata = scvi.data.synthetic_iid(
        batch_size=4,
        n_genes=5,
        n_proteins=3,
        n_regions=2,
        n_batches=2,
        n_labels=2,
    )
    adata.obs["sample"] = np.asarray(
        [f"sample_{index % n_samples}" for index in range(adata.n_obs)]
    )
    MrTotalVI.setup_anndata(
        adata,
        protein_expression_obsm_key="protein_expression",
        sample_key="sample",
        batch_key="batch",
    )
    return adata


def _make_v2_model(*, u_encoder_mode: str = "sample_blind"):
    adata = _make_adata(n_samples=3)
    model = MrTotalVI(
        adata,
        sample_key="sample",
        n_latent=3,
        n_latent_u=2,
        hierarchy_mode="centered_v2",
        u_encoder_mode=u_encoder_mode,
    )
    model.is_trained_ = True
    return model, adata


def _make_context_v2_model():
    adata = scvi.data.synthetic_iid(
        batch_size=6,
        n_genes=5,
        n_proteins=3,
        n_regions=2,
        n_batches=2,
        n_labels=2,
    )
    adata.obs["sample"] = np.asarray(
        ["sample_0"] * 6 + ["sample_1"] * 4 + ["sample_2"] * 2
    )
    adata.obs["batch"] = np.asarray(
        ["batch_0"] * 4
        + ["batch_1"] * 2
        + ["batch_0"]
        + ["batch_1"] * 3
        + ["batch_0", "batch_1"]
    )
    adata.obs["panel"] = np.where(
        adata.obs["batch"].to_numpy() == "batch_0",
        "panel_0",
        "panel_1",
    )
    adata.obs["size_factor"] = np.arange(1, adata.n_obs + 1, dtype=np.float32)
    adata.obs["technical_cat"] = np.where(
        np.arange(adata.n_obs) % 2 == 0,
        "cat_0",
        "cat_1",
    )
    adata.obs["technical_cont"] = np.linspace(
        0.0,
        1.0,
        adata.n_obs,
        dtype=np.float32,
    )
    protein = np.asarray(adata.obsm["protein_expression"], dtype=np.float32) + 1.0
    protein[adata.obs["panel"].to_numpy() == "panel_0", 0] = 0.0
    adata.obsm["protein_expression"] = protein
    MrTotalVI.setup_anndata(
        adata,
        protein_expression_obsm_key="protein_expression",
        sample_key="sample",
        batch_key="batch",
        panel_key="panel",
        size_factor_key="size_factor",
        categorical_covariate_keys=["technical_cat"],
        continuous_covariate_keys=["technical_cont"],
    )
    with pytest.warns(UserWarning, match="Some proteins have all 0 counts"):
        model = MrTotalVI(
            adata,
            sample_key="sample",
            n_latent=3,
            n_latent_u=2,
            hierarchy_mode="centered_v2",
            u_encoder_mode="sample_blind",
        )
    model.is_trained_ = True
    return model, adata


def _make_enrichment_v2_model():
    adata = _make_adata(n_samples=4)
    sample_number = adata.obs["sample"].str.removeprefix("sample_").astype(int)
    adata.obs["group"] = np.where(sample_number % 2 == 0, "numerator", "denominator")
    adata.obs["donor"] = np.where(sample_number < 2, "donor_0", "donor_1")
    model = MrTotalVI(
        adata,
        sample_key="sample",
        n_latent=3,
        n_latent_u=2,
        hierarchy_mode="centered_v2",
        u_encoder_mode="sample_blind",
    )
    model.is_trained_ = True
    return model, adata


def test_pre_v2_oracle_checksum_manifest_and_policy_are_immutable():
    """The oracle inventory and CPU-portability policy have exact authority."""
    inventory = verify_oracle_inventory(ORACLE_ROOT)
    assert inventory["checksum_manifest_sha256"] == EXPECTED_CHECKSUM_MANIFEST_SHA256
    assert inventory["external_paths"] == ["../generate_legacy_oracle.py"]
    assert inventory["source_commit"] == EXPECTED_SOURCE_COMMIT
    assert inventory["verified_file_count"] == 14

    policy, policy_bytes, policy_sha256 = load_portability_policy(
        PORTABILITY_POLICY_PATH
    )
    assert policy_bytes == canonical_json_bytes(policy)
    assert policy_sha256 == EXPECTED_PORTABILITY_POLICY_SHA256
    assert policy["schema"] == "mrtotalvi-pre-v2-oracle-cpu-portability-v1"
    assert policy["oracle"] == {
        "checksum_manifest_sha256": EXPECTED_CHECKSUM_MANIFEST_SHA256,
        "source_commit": EXPECTED_SOURCE_COMMIT,
    }
    assert policy["execution"] == {"device": "cpu", "dtype": "float32"}
    assert policy["tolerances"]["strict"] == {"atol": 1e-7, "rtol": 1e-6}
    assert policy["decision_use"] == {"promotion": False, "selection": False}


@pytest.mark.parametrize(
    "mutation",
    [
        "unauthorized_parent",
        "absolute_path",
        "duplicate_record",
        "symlink",
        "missing_target",
        "extra_path",
    ],
)
def test_pre_v2_oracle_inventory_rejects_every_non_authoritative_path(
    tmp_path,
    mutation,
):
    """Traversal, ambiguity, links, omissions, and additions all fail closed."""
    legacy_root = tmp_path / "legacy_oracle"
    oracle_root = legacy_root / ORACLE_ROOT.name
    legacy_root.mkdir()
    shutil.copytree(ORACLE_ROOT, oracle_root)
    shutil.copy2(
        ORACLE_ROOT.parent / "generate_legacy_oracle.py",
        legacy_root / "generate_legacy_oracle.py",
    )
    checksum_path = oracle_root / "checksums.sha256"
    expected_manifest_sha256 = EXPECTED_CHECKSUM_MANIFEST_SHA256

    if mutation == "unauthorized_parent":
        checksum_path.write_bytes(
            checksum_path.read_bytes() + f"{'0' * 64}  ../../escape.py\n".encode()
        )
    elif mutation == "absolute_path":
        checksum_path.write_bytes(
            checksum_path.read_bytes() + f"{'0' * 64}  /tmp/escape.py\n".encode()
        )
    elif mutation == "duplicate_record":
        first_line = checksum_path.read_bytes().splitlines(keepends=True)[0]
        checksum_path.write_bytes(checksum_path.read_bytes() + first_line)
    elif mutation == "symlink":
        target = oracle_root / "environment_manifest.json"
        external = tmp_path / "environment_manifest.json"
        shutil.copy2(target, external)
        target.unlink()
        target.symlink_to(external)
    elif mutation == "missing_target":
        (oracle_root / "environment_manifest.json").unlink()
    elif mutation == "extra_path":
        (oracle_root / "unexpected.txt").write_text("unexpected\n", encoding="utf-8")
    else:  # pragma: no cover - the parameterization is closed above
        raise AssertionError(mutation)

    if mutation in {"unauthorized_parent", "absolute_path", "duplicate_record"}:
        expected_manifest_sha256 = hashlib.sha256(checksum_path.read_bytes()).hexdigest()
    with pytest.raises(ValueError):
        verify_oracle_inventory(
            oracle_root,
            expected_manifest_sha256=expected_manifest_sha256,
        )


@pytest.fixture(scope="module")
def pre_v2_cpu_portability_assessment():
    """Run the compatibility assessment once in a fixed-thread subprocess."""
    return run_assessment_subprocess(
        oracle_root=ORACLE_ROOT,
        policy_path=PORTABILITY_POLICY_PATH,
    )


def test_pre_v2_oracle_strict_manifest_tolerance_is_diagnostic(
    pre_v2_cpu_portability_assessment,
):
    """Manifest tolerances remain strict and host rounding is explicit, not hidden."""
    assessment = pre_v2_cpu_portability_assessment
    diagnostic = assessment["summary"]["strict_diagnostic"]
    policy, _, _ = load_portability_policy(PORTABILITY_POLICY_PATH)
    rows = [
        row
        for model in assessment["models"].values()
        for row in model["rows"]
    ]
    assert diagnostic["status"] in {"pass", "host_rounding_mismatches"}
    assert diagnostic["tolerances"] == {"atol": 1e-7, "rtol": 1e-6}
    assert diagnostic["failing_element_count"] == sum(
        row["failing_count"]["strict"]
        for model in assessment["models"].values()
        for row in model["rows"]
    )
    if diagnostic["status"] == "pass":
        assert diagnostic["failing_element_count"] == 0
    else:
        assert diagnostic["failing_element_count"] > 0
        assert diagnostic["classification"] == "cpu_float32_host_rounding"
        assert max(
            row["max_abs_delta"]
            for row in rows
            if row["failing_count"]["strict"] > 0
        ) == policy["derivation"]["observed_floor_requiring_max_abs_delta"]
    assert max(row["max_abs_delta"] for row in rows) == policy["derivation"][
        "observed_overall_max_abs_delta"
    ]


def test_pre_v2_oracle_cpu_portable_assessment_passes(
    pre_v2_cpu_portability_assessment,
):
    """A checksum-clean CPU replay passes only the separately named portable policy."""
    assessment = pre_v2_cpu_portability_assessment
    policy, _, policy_sha256 = load_portability_policy(PORTABILITY_POLICY_PATH)
    binding = assessment["binding"]
    assert binding["policy_sha256"] == policy_sha256
    assert binding["oracle_checksum_manifest_sha256"] == (
        EXPECTED_CHECKSUM_MANIFEST_SHA256
    )
    assert binding["source_commit"] == EXPECTED_SOURCE_COMMIT
    assert binding["assessment_rows_sha256"] == canonical_json_sha256(
        {
            model_name: assessment["models"][model_name]["rows"]
            for model_name in ("mrtotalvi", "mrmultivi")
        }
    )
    assert binding["provenance_sha256"] == canonical_json_sha256(
        assessment["provenance"]
    )
    assert binding["binding_sha256"] == canonical_json_sha256(
        {key: value for key, value in binding.items() if key != "binding_sha256"}
    )
    assert assessment["summary"]["portable_verdict"] == "pass"
    assert assessment["summary"]["portable_failing_element_count"] == 0
    assert assessment["summary"]["nonfinite_count"] == 0
    assert assessment["provenance"]["execution"]["device"] == "cpu"
    assert assessment["provenance"]["execution"]["dtype"] == "float32"
    assert assessment["provenance"]["determinism"]["algorithms_enabled"] is True
    assert assessment["provenance"]["threads"]["torch_num_threads"] == 1
    assert assessment["provenance"]["threads"]["torch_num_interop_threads"] == 1
    assert set(assessment["provenance"]["runtime"]) == {
        "anndata",
        "executable",
        "numpy",
        "platform",
        "python",
        "scvi_tools",
        "torch",
    }
    assert set(assessment["provenance"]["cpu"]) == {
        "flags",
        "flags_count",
        "flags_sha256",
        "machine",
        "model",
        "processor",
        "torch_capability",
    }
    assert assessment["provenance"]["cpu"]["flags"] == sorted(
        assessment["provenance"]["cpu"]["flags"]
    )
    assert assessment["provenance"]["cpu"]["flags_count"] == len(
        assessment["provenance"]["cpu"]["flags"]
    )
    assert assessment["provenance"]["execution"]["cuda_available"] is False
    assert assessment["provenance"]["execution"]["cuda_visible_devices"] == ""
    assert set(
        assessment["provenance"]["threads"]["environment"].values()
    ) <= {"", "0", "1"}

    required_row_fields = {
        "category",
        "dtype",
        "failing_count",
        "key",
        "max_abs_delta",
        "max_normalized_error",
        "n",
        "nonfinite",
        "shape",
    }
    for model_name, model in assessment["models"].items():
        run_manifest = model["run_manifest"]
        assert run_manifest["source_commit"] == EXPECTED_SOURCE_COMMIT
        assert run_manifest["model_seed"] == 7301
        assert run_manifest["forward_seed"] == 9817
        assert run_manifest["batch_size"] == 7
        assert run_manifest["kl_weight"] == 0.73
        assert run_manifest["sample_order"] == ["sample_0", "sample_1", "sample_2"]
        assert run_manifest["tolerances"] == policy["tolerances"]["strict"]
        assert run_manifest["protein_reconstruction_weight"] == (
            0.61 if model_name == "mrtotalvi" else None
        )
        assert model["requirements"] == {
            "array_keys_exact": True,
            "cell_indices_exact": True,
            "finite": True,
            "gradient_manifest_exact": True,
            "shapes_exact": True,
            "state_manifest_exact": True,
        }
        assert {row["key"] for row in model["rows"]} == set(model["assessed_keys"])
        assert len(model["rows"]) == (115 if model_name == "mrtotalvi" else 135)
        assert all(set(row) == required_row_fields for row in model["rows"])
        assert all(row["dtype"] == "float32" for row in model["rows"])
        assert all(row["failing_count"]["portable"] == 0 for row in model["rows"])
        assert all(row["nonfinite"] == {"actual": 0, "oracle": 0} for row in model["rows"])


def test_mode_contract_is_opt_in_validated_and_topology_preserving():
    """Modes are explicit metadata, with centered v2 restricted to the frozen MAP contract."""
    adata = _make_adata()
    legacy = MrTotalVI(adata, sample_key="sample", n_latent=3)
    assert legacy.hierarchy_mode == "legacy"
    assert legacy.u_encoder_mode == "sample_conditioned"
    assert legacy.init_params_["non_kwargs"]["hierarchy_mode"] == "legacy"
    assert legacy.init_params_["non_kwargs"]["u_encoder_mode"] == "sample_conditioned"

    with pytest.raises(ValueError, match="hierarchy_mode"):
        MrTotalVI(adata, sample_key="sample", hierarchy_mode="unknown")
    with pytest.raises(ValueError, match="u_encoder_mode"):
        MrTotalVI(adata, sample_key="sample", u_encoder_mode="unknown")
    with pytest.raises(ValueError, match="use_map=True"):
        MrTotalVI(
            adata,
            sample_key="sample",
            hierarchy_mode="centered_v2",
            use_map=False,
        )
    with pytest.raises(ValueError, match="z_u_prior=True"):
        MrTotalVI(
            adata,
            sample_key="sample",
            hierarchy_mode="centered_v2",
            z_u_prior=False,
        )

    state_contract = {
        key: tuple(value.shape) for key, value in legacy.module.state_dict().items()
    }
    for hierarchy_mode in ("legacy", "centered_v2"):
        for u_encoder_mode in ("sample_conditioned", "sample_blind"):
            model = MrTotalVI(
                adata,
                sample_key="sample",
                n_latent=3,
                hierarchy_mode=hierarchy_mode,
                u_encoder_mode=u_encoder_mode,
            )
            assert {
                key: tuple(value.shape)
                for key, value in model.module.state_dict().items()
            } == state_contract


def test_sample_blind_encoder_bypasses_sample_only_and_keeps_technical_covariates():
    """Sample-blind u is sample invariant while declared technical inputs remain active."""
    adata = scvi.data.synthetic_iid(
        batch_size=4,
        n_genes=5,
        n_proteins=3,
        n_regions=2,
        n_batches=2,
        n_labels=2,
    )
    adata.obs["sample"] = np.asarray(
        [f"sample_{index % 3}" for index in range(adata.n_obs)]
    )
    adata.obs["technical_category"] = np.asarray(
        [f"tech_{index % 2}" for index in range(adata.n_obs)]
    )
    adata.obs["technical_continuous"] = np.linspace(0.0, 1.0, adata.n_obs)
    MrTotalVI.setup_anndata(
        adata,
        protein_expression_obsm_key="protein_expression",
        sample_key="sample",
        batch_key="batch",
        categorical_covariate_keys=["technical_category"],
        continuous_covariate_keys=["technical_continuous"],
    )
    model = MrTotalVI(
        adata,
        sample_key="sample",
        n_latent=3,
        u_encoder_mode="sample_blind",
        encode_covariates=True,
    )
    qu = model.module.qu.eval()

    x = torch.as_tensor(np.asarray(adata.X[:4]), dtype=torch.float32)
    y = torch.as_tensor(
        np.asarray(adata.obsm["protein_expression"][:4]),
        dtype=torch.float32,
    )
    sample_0 = torch.zeros(4, 1, dtype=torch.long)
    sample_1 = torch.ones(4, 1, dtype=torch.long)
    batch_0 = torch.zeros(4, 1, dtype=torch.long)
    batch_1 = torch.ones(4, 1, dtype=torch.long)
    cat_covs = torch.zeros(4, 1, dtype=torch.long)
    cont_covs = torch.zeros(4, 1)

    blind_0 = qu(
        x,
        y,
        sample_0,
        batch_index=batch_0,
        cat_covs=cat_covs,
        cont_covs=cont_covs,
    )
    blind_1 = qu(
        x,
        y,
        sample_1,
        batch_index=batch_0,
        cat_covs=cat_covs,
        cont_covs=cont_covs,
    )
    torch.testing.assert_close(blind_0.loc, blind_1.loc, rtol=0.0, atol=0.0)
    torch.testing.assert_close(blind_0.scale, blind_1.scale, rtol=0.0, atol=0.0)

    changed_technical_context = qu(
        x,
        y,
        sample_0,
        batch_index=batch_1,
        cat_covs=cat_covs,
        cont_covs=cont_covs,
    )
    assert not torch.allclose(blind_0.loc, changed_technical_context.loc)

    qu.zero_grad(set_to_none=True)
    (blind_0.loc.sum() + blind_0.scale.sum()).backward()
    conditioning_parameters = (
        qu.cond_norm1.gamma_embedding.weight,
        qu.cond_norm1.beta_embedding.weight,
        qu.cond_norm2.gamma_embedding.weight,
        qu.cond_norm2.beta_embedding.weight,
        qu.sample_embed.weight,
    )
    assert all(
        parameter.grad is None or torch.count_nonzero(parameter.grad) == 0
        for parameter in conditioning_parameters
    )
    assert qu.fc1.weight.grad is not None
    assert torch.isfinite(qu.fc1.weight.grad).all()
    assert torch.count_nonzero(qu.fc1.weight.grad) > 0


def test_centered_hierarchy_uses_the_full_registered_sample_universe():
    """Centered inference exposes raw/full residuals and exact factual gathers."""
    adata = _make_adata(n_samples=3)
    model = MrTotalVI(
        adata,
        sample_key="sample",
        n_latent=3,
        hierarchy_mode="centered_v2",
    )
    module = model.module.eval()
    tensors = next(
        iter(model._make_data_loader(adata=adata, indices=np.arange(4), batch_size=4))
    )
    torch.manual_seed(117)
    inference_outputs = module.inference(**module._get_inference_input(tensors))

    eps_raw = inference_outputs["eps_raw_all"]
    eps_centered = inference_outputs["eps_centered_all"]
    z_all = inference_outputs["z_all"]
    z_base = inference_outputs["z_base"]
    assert eps_raw.shape == (4, 3, 3)
    assert eps_centered.shape == (4, 3, 3)
    assert z_all.shape == (4, 3, 3)
    torch.testing.assert_close(
        eps_centered.mean(dim=1),
        torch.zeros_like(z_base),
        rtol=0.0,
        atol=1e-6,
    )
    torch.testing.assert_close(z_all.mean(dim=1), z_base, rtol=0.0, atol=1e-6)
    torch.testing.assert_close(
        z_all[:, 0] - z_all[:, 2],
        eps_centered[:, 0] - eps_centered[:, 2],
        rtol=0.0,
        atol=1e-6,
    )

    observed_sample = tensors[REGISTRY_KEYS.SAMPLE_KEY].long().flatten()
    row = torch.arange(observed_sample.numel())
    torch.testing.assert_close(
        inference_outputs["eps"],
        eps_centered[row, observed_sample],
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        inference_outputs["z"],
        z_all[row, observed_sample],
        rtol=0.0,
        atol=0.0,
    )


def test_centered_loss_penalizes_raw_residuals_for_every_registered_sample():
    """The v2 KL term is the equal-sample raw penalty and trains absent sample rows."""
    adata = _make_adata(n_samples=3)
    model = MrTotalVI(
        adata,
        sample_key="sample",
        n_latent=3,
        hierarchy_mode="centered_v2",
        kl_u_weight=0.7,
        kl_z_weight=1.3,
    )
    module = model.module.train()
    factual_sample_0 = np.asarray([0, 3, 6])
    tensors = next(
        iter(
            model._make_data_loader(
                adata=adata,
                indices=factual_sample_0,
                batch_size=len(factual_sample_0),
            )
        )
    )
    assert torch.count_nonzero(tensors[REGISTRY_KEYS.SAMPLE_KEY]) == 0

    module.zero_grad(set_to_none=True)
    torch.manual_seed(431)
    inference_outputs, _, loss_output = module(
        tensors,
        loss_kwargs={"kl_weight": 0.4, "pro_recons_weight": 0.8},
    )
    raw_penalty = (
        -Normal(
            0.0,
            torch.exp(module.pz_scale.clamp(min=-4.0)),
        )
        .log_prob(inference_outputs["eps_raw_all"])
        .sum(dim=-1)
        .mean(dim=1)
    )
    kl_u = module.kl_u(
        inference_outputs["qu"],
        inference_outputs["u"],
        tensors[REGISTRY_KEYS.LABELS_KEY],
    )
    torch.testing.assert_close(
        loss_output.kl_local["kl_div_z"],
        0.7 * kl_u + 1.3 * raw_penalty,
        rtol=1e-6,
        atol=1e-7,
    )

    loss_output.loss.backward()
    embedding_gradient = module.qz.embedding.weight.grad
    assert embedding_gradient is not None
    assert torch.isfinite(embedding_gradient).all()
    assert torch.all(torch.count_nonzero(embedding_gradient, dim=1) > 0)


def test_centered_single_sample_and_common_raw_shift_contracts():
    """S=1 centers to zero, while a common raw shift changes only the raw penalty."""
    single_adata = _make_adata(n_samples=1)
    single = MrTotalVI(
        single_adata,
        sample_key="sample",
        n_latent=3,
        hierarchy_mode="centered_v2",
    ).module.eval()
    u_single = torch.randn(2, single.n_latent_u)
    z_base, eps_raw, eps_centered, z_all = single._all_sample_residuals(u_single)
    torch.testing.assert_close(
        eps_centered,
        torch.zeros_like(eps_centered),
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(z_all[:, 0], z_base, rtol=0.0, atol=0.0)
    single_penalty = -Normal(
        0.0, torch.exp(single.pz_scale.clamp(min=-4.0))
    ).log_prob(eps_raw).sum()
    assert torch.isfinite(single_penalty)
    assert single_penalty > 0

    adata = _make_adata(n_samples=3)
    module = MrTotalVI(
        adata,
        sample_key="sample",
        n_latent=3,
        hierarchy_mode="centered_v2",
    ).module.eval()
    u = torch.randn(4, module.n_latent_u)
    _, raw_before, centered_before, z_before = module._all_sample_residuals(u)
    shift = torch.tensor([2.0, -1.0, 0.5])
    with torch.no_grad():
        module.qz.attention_block.mlp_residual.fc.bias.add_(shift)
    _, raw_after, centered_after, z_after = module._all_sample_residuals(u)

    torch.testing.assert_close(
        raw_after - raw_before,
        shift[None, None, :].expand_as(raw_before),
        rtol=0.0,
        atol=1e-6,
    )
    torch.testing.assert_close(centered_after, centered_before, rtol=0.0, atol=1e-6)
    torch.testing.assert_close(z_after, z_before, rtol=0.0, atol=1e-6)
    peps = Normal(0.0, torch.exp(module.pz_scale.clamp(min=-4.0)))
    assert not torch.allclose(
        -peps.log_prob(raw_before).sum(),
        -peps.log_prob(raw_after).sum(),
    )


def test_centered_target_chunks_preserve_values_and_gradients():
    """Target chunks 1, interior, and S have matching residual values and gradients."""
    adata = _make_adata(n_samples=3)
    module = MrTotalVI(
        adata,
        sample_key="sample",
        n_latent=3,
        hierarchy_mode="centered_v2",
    ).module.eval()
    base_u = torch.randn(2, module.n_latent_u)
    snapshots = {}
    for chunk_size in (1, 2, 3):
        module.zero_grad(set_to_none=True)
        u = base_u.detach().clone().requires_grad_(True)
        z_base, eps_raw, eps_centered, z_all = module._all_sample_residuals(
            u,
            target_chunk_size=chunk_size,
        )
        objective = (
            z_base.square().sum()
            + eps_raw.square().sum()
            + eps_centered.square().sum()
            + z_all.square().sum()
        )
        objective.backward()
        snapshots[chunk_size] = {
            "z_base": z_base.detach(),
            "eps_raw": eps_raw.detach(),
            "eps_centered": eps_centered.detach(),
            "z_all": z_all.detach(),
            "u_gradient": u.grad.detach(),
            "embedding_gradient": module.qz.embedding.weight.grad.detach().clone(),
        }

    expected = snapshots[3]
    for chunk_size in (1, 2):
        for key, value in snapshots[chunk_size].items():
            torch.testing.assert_close(
                value,
                expected[key],
                rtol=1e-6,
                atol=1e-6,
                msg=f"chunk_size={chunk_size} drifted for {key}",
            )


def test_centered_cell_batches_preserve_values_and_gradients():
    """Cell batches 1, interior, and full preserve all-target values and gradients."""
    adata = _make_adata(n_samples=3)
    module = MrTotalVI(
        adata,
        sample_key="sample",
        n_latent=3,
        hierarchy_mode="centered_v2",
    ).module.eval()
    base_u = torch.randn(4, module.n_latent_u)
    snapshots = {}
    for cell_batch_size in (1, 2, 4):
        module.zero_grad(set_to_none=True)
        u = base_u.detach().clone().requires_grad_(True)
        outputs = [[] for _ in range(4)]
        objective = torch.zeros(())
        for start in range(0, u.shape[0], cell_batch_size):
            result = module._all_sample_residuals(
                u[start : start + cell_batch_size],
                target_chunk_size=2,
            )
            for output_parts, value in zip(outputs, result, strict=True):
                output_parts.append(value)
                objective = objective + value.square().sum()
        objective.backward()
        snapshots[cell_batch_size] = {
            "outputs": [
                torch.cat(output_parts, dim=1 if output_parts[0].ndim == 4 else 0)
                for output_parts in outputs
            ],
            "u_gradient": u.grad.detach().clone(),
            "embedding_gradient": module.qz.embedding.weight.grad.detach().clone(),
        }

    expected = snapshots[4]
    for cell_batch_size in (1, 2):
        for actual, wanted in zip(
            snapshots[cell_batch_size]["outputs"],
            expected["outputs"],
            strict=True,
        ):
            torch.testing.assert_close(
                actual,
                wanted,
                rtol=1e-6,
                atol=1e-6,
            )
        torch.testing.assert_close(
            snapshots[cell_batch_size]["u_gradient"],
            expected["u_gradient"],
            rtol=1e-6,
            atol=1e-6,
        )
        torch.testing.assert_close(
            snapshots[cell_batch_size]["embedding_gradient"],
            expected["embedding_gradient"],
            rtol=1e-6,
            atol=1e-6,
        )


def test_centered_scale_observations_uses_exact_sample_balanced_weights():
    """C3 uses N/(S*n_s), with mean-one weights and duplicate-cell invariance."""
    adata = _make_adata(n_samples=3)
    model = MrTotalVI(
        adata,
        sample_key="sample",
        n_latent=3,
        hierarchy_mode="centered_v2",
        scale_observations=True,
    )
    module = model.module.train()
    tensors = next(
        iter(
            model._make_data_loader(
                adata=adata,
                indices=np.arange(adata.n_obs),
                batch_size=adata.n_obs,
            )
        )
    )
    torch.manual_seed(921)
    _, _, loss_output = module(
        tensors,
        loss_kwargs={"kl_weight": 0.4, "pro_recons_weight": 0.8},
    )
    sample_index = tensors[REGISTRY_KEYS.SAMPLE_KEY].long().flatten()
    n_obs = module.n_obs_per_sample[sample_index]
    weights = module.n_obs_per_sample.sum() / (module._n_sample * n_obs)
    torch.testing.assert_close(weights.mean(), torch.ones(()), rtol=0.0, atol=1e-7)

    per_cell = (
        loss_output.reconstruction_loss["reconst_loss_gene"]
        + 0.4 * 0.8 * loss_output.reconstruction_loss["reconst_loss_protein"]
        + 0.4 * loss_output.kl_local["kl_div_z"]
        + loss_output.kl_local["kl_div_l_gene"]
        + 0.4 * loss_output.kl_local["kl_div_back_pro"]
    )
    torch.testing.assert_close(
        loss_output.loss,
        (per_cell * weights).mean(),
        rtol=1e-6,
        atol=1e-7,
    )

    duplicate_mask = sample_index == 0
    duplicated_values = torch.cat([per_cell.detach(), per_cell.detach()[duplicate_mask]])
    duplicated_samples = torch.cat([sample_index, sample_index[duplicate_mask]])
    duplicated_counts = torch.bincount(duplicated_samples, minlength=module._n_sample)
    duplicated_weights = duplicated_samples.numel() / (
        module._n_sample * duplicated_counts[duplicated_samples]
    )
    torch.testing.assert_close(
        (duplicated_values * duplicated_weights).mean(),
        (per_cell.detach() * weights).mean(),
        rtol=1e-6,
        atol=1e-7,
    )


def test_checkpoint_modes_default_roundtrip_and_require_explicit_semantic_override(
    tmp_path,
):
    """Missing metadata defaults safely; differing overrides are explicit and recorded."""
    verify_oracle_inventory(ORACLE_ROOT)
    old = MrTotalVI.load(
        ORACLE_ROOT / "mrtotalvi" / "checkpoint",
        accelerator="cpu",
    )
    assert old.hierarchy_mode == "legacy"
    assert old.u_encoder_mode == "sample_conditioned"
    assert old.init_params_["non_kwargs"]["hierarchy_mode"] == "legacy"
    assert old.init_params_["non_kwargs"]["u_encoder_mode"] == "sample_conditioned"

    adata = _make_adata(n_samples=3)
    v2 = MrTotalVI(
        adata,
        sample_key="sample",
        n_latent=3,
        hierarchy_mode="centered_v2",
        u_encoder_mode="sample_blind",
    )
    save_path = tmp_path / "centered-v2"
    v2.save(save_path, save_anndata=True)
    roundtrip = MrTotalVI.load(save_path, accelerator="cpu")
    assert roundtrip.hierarchy_mode == "centered_v2"
    assert roundtrip.u_encoder_mode == "sample_blind"

    with pytest.raises(ValueError, match="allow_semantic_override=True"):
        MrTotalVI.load(
            save_path,
            accelerator="cpu",
            hierarchy_mode_override="legacy",
        )
    with pytest.raises(ValueError, match="allow_semantic_override=True"):
        MrTotalVI.load(
            save_path,
            accelerator="cpu",
            u_encoder_mode_override="sample_conditioned",
        )

    with pytest.warns(UserWarning, match="semantic override"):
        hierarchy_override = MrTotalVI.load(
            save_path,
            accelerator="cpu",
            hierarchy_mode_override="legacy",
            allow_semantic_override=True,
        )
    assert hierarchy_override.loaded_hierarchy_mode == "centered_v2"
    assert hierarchy_override.resolved_hierarchy_mode == "legacy"
    assert hierarchy_override.hierarchy_mode == "legacy"
    assert hierarchy_override.u_encoder_mode == "sample_blind"
    assert hierarchy_override.init_params_["non_kwargs"]["hierarchy_mode"] == "legacy"
    hierarchy_resave_path = tmp_path / "hierarchy-override-resaved"
    hierarchy_override.save(hierarchy_resave_path, save_anndata=True)
    hierarchy_resaved = MrTotalVI.load(
        hierarchy_resave_path,
        accelerator="cpu",
    )
    assert hierarchy_resaved.hierarchy_mode == "legacy"
    assert hierarchy_resaved.u_encoder_mode == "sample_blind"

    with pytest.warns(UserWarning, match="semantic override"):
        encoder_override = MrTotalVI.load(
            save_path,
            accelerator="cpu",
            u_encoder_mode_override="sample_conditioned",
            allow_semantic_override=True,
        )
    assert encoder_override.loaded_u_encoder_mode == "sample_blind"
    assert encoder_override.resolved_u_encoder_mode == "sample_conditioned"
    assert encoder_override.hierarchy_mode == "centered_v2"
    assert encoder_override.u_encoder_mode == "sample_conditioned"
    assert (
        encoder_override.init_params_["non_kwargs"]["u_encoder_mode"]
        == "sample_conditioned"
    )
    encoder_resave_path = tmp_path / "encoder-override-resaved"
    encoder_override.save(encoder_resave_path, save_anndata=True)
    encoder_resaved = MrTotalVI.load(
        encoder_resave_path,
        accelerator="cpu",
    )
    assert encoder_resaved.hierarchy_mode == "centered_v2"
    assert encoder_resaved.u_encoder_mode == "sample_conditioned"


def test_counterfactual_latent_schema_order_and_v2_only_contract():
    """The latent API freezes names, dimensions, requested order, and mode errors."""
    model, adata = _make_v2_model()
    indices = np.asarray([3, 1, 5])
    targets = ["sample_2", "sample_0"]
    result = model.get_counterfactual_latent(
        adata=adata,
        indices=indices,
        target_samples=targets,
        inference_mode="latent_mean",
        n_draws=1,
        reference_indices=np.arange(6),
        batch_size=1,
        target_chunk_size=1,
    )
    assert dict(result.sizes) == {
        "draw": 1,
        "cell_name": 3,
        "latent_u_dim": 2,
        "latent_dim": 3,
        "target_sample": 2,
    }
    assert set(result.data_vars) == {
        "u",
        "z_base",
        "eps_raw",
        "eps_centered",
        "z",
        "admissible",
        "target_support",
        "observed_sample",
    }
    assert result.cell_name.to_numpy().tolist() == adata.obs_names[indices].tolist()
    assert result.target_sample.to_numpy().tolist() == targets
    assert result["u"].dims == ("draw", "cell_name", "latent_u_dim")
    assert result["z_base"].dims == ("draw", "cell_name", "latent_dim")
    assert result["eps_raw"].dims == (
        "draw",
        "cell_name",
        "target_sample",
        "latent_dim",
    )
    assert result["eps_centered"].dims == result["eps_raw"].dims
    assert result["z"].dims == result["eps_raw"].dims
    assert result["admissible"].dims == ("cell_name", "target_sample")
    assert result["target_support"].dims == ("cell_name", "target_sample")
    assert result["observed_sample"].dims == ("cell_name",)
    assert result["u"].dtype == np.float32
    assert result["admissible"].dtype == np.bool_
    assert result["target_support"].dtype == np.bool_
    assert result.attrs["schema_version"] == "mrtotalvi-counterfactual-v1"
    assert result.attrs["hierarchy_mode"] == "centered_v2"
    assert result.attrs["u_encoder_mode"] == "sample_blind"
    assert result.attrs["interpretation"] == "registered-sample model transformation; non-causal"

    legacy_adata = _make_adata(n_samples=3)
    legacy = MrTotalVI(legacy_adata, sample_key="sample", n_latent=3)
    legacy.is_trained_ = True
    with pytest.raises(RuntimeError, match="centered_v2"):
        legacy.get_counterfactual_latent()
    with pytest.raises(ValueError, match="duplicate"):
        model.get_counterfactual_latent(target_samples=["sample_0", "sample_0"])
    with pytest.raises(ValueError, match="Unknown target"):
        model.get_counterfactual_latent(target_samples=["not-registered"])
    with pytest.raises(ValueError, match="latent_mean.*n_draws=1"):
        model.get_counterfactual_latent(inference_mode="latent_mean", n_draws=2)

    singleton_support = model.get_counterfactual_latent(
        indices=[0],
        target_samples=["sample_0", "sample_1"],
        reference_indices=[0, 1],
    )
    assert not singleton_support["target_support"].sel(
        target_sample="sample_0"
    ).item()
    assert singleton_support["target_support"].sel(
        target_sample="sample_1"
    ).item()


def test_counterfactual_latent_posterior_is_batch_target_and_subset_invariant():
    """Counter-keyed posterior draws and full-universe centering are partition invariant."""
    model, adata = _make_v2_model()
    indices = np.asarray([5, 0, 7, 2])
    full = model.get_counterfactual_latent(
        adata=adata,
        indices=indices,
        inference_mode="posterior_mc",
        n_draws=3,
        batch_size=4,
        target_chunk_size=3,
        random_state=42,
    )
    assert full.target_sample.to_numpy().tolist() == list(map(str, model.sample_order))
    subset = model.get_counterfactual_latent(
        adata=adata,
        indices=indices,
        target_samples=["sample_2", "sample_0"],
        inference_mode="posterior_mc",
        n_draws=3,
        batch_size=1,
        target_chunk_size=1,
        random_state=42,
    )
    torch.testing.assert_close(
        torch.as_tensor(full["u"].to_numpy()),
        torch.as_tensor(subset["u"].to_numpy()),
        rtol=1e-6,
        atol=1e-6,
    )
    for variable in ("eps_raw", "eps_centered", "z"):
        np.testing.assert_allclose(
            full[variable].sel(
                target_sample=["sample_2", "sample_0"]
            ).to_numpy(),
            subset[variable].to_numpy(),
            rtol=1e-6,
            atol=1e-6,
        )

    np.testing.assert_allclose(
        full["eps_centered"].mean("target_sample").to_numpy(),
        0.0,
        rtol=0.0,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        full["z"].mean("target_sample").to_numpy(),
        full["z_base"].to_numpy(),
        rtol=0.0,
        atol=1e-6,
    )
    for variable in ("u", "z_base", "eps_raw", "eps_centered", "z"):
        np.testing.assert_allclose(
            full[f"{variable}_posterior_mean"].to_numpy(),
            full[variable].mean("draw").to_numpy(),
            rtol=1e-6,
            atol=1e-7,
        )
        assert full[f"{variable}_posterior_quantile"].dims[0] == "quantile"

    reordered = model.get_counterfactual_latent(
        adata=adata,
        indices=indices[::-1],
        target_samples=["sample_0"],
        inference_mode="posterior_mc",
        n_draws=3,
        batch_size=2,
        target_chunk_size=2,
        random_state=42,
    )
    np.testing.assert_allclose(
        full["u"].sel(cell_name=reordered.cell_name).to_numpy(),
        reordered["u"].to_numpy(),
        rtol=1e-6,
        atol=1e-6,
    )

    with pytest.raises(ValueError, match="strictly increasing"):
        model.get_counterfactual_latent(
            inference_mode="posterior_mc",
            n_draws=2,
            quantiles=(0.5, 0.25),
        )
    with pytest.raises(MemoryError, match="512"):
        model.get_counterfactual_latent(
            indices=[0],
            inference_mode="posterior_mc",
            n_draws=100_000_000,
        )


def test_counterfactual_invalid_requests_fail_before_inference(monkeypatch):
    """Invalid modes, features, and technical contexts fail before encoding."""
    model, adata = _make_v2_model()
    gene = str(adata.var_names[0])
    protein = str(
        model.adata_manager.get_state_registry(
            REGISTRY_KEYS.PROTEIN_EXP_KEY
        ).column_names[0]
    )

    import scvi.external.mrtotalvi._counterfactual as counterfactual_module

    def unexpected_inference(*args, **kwargs):
        raise AssertionError("model inference ran before request validation")

    monkeypatch.setattr(
        counterfactual_module,
        "_encode_u_params",
        unexpected_inference,
    )
    monkeypatch.setattr(
        counterfactual_module,
        "_collect_observed_context",
        unexpected_inference,
    )

    invalid_calls = (
        (
            lambda: model.get_counterfactual_latent(
                target_samples=["not-registered"],
            ),
            "Unknown target",
        ),
        (
            lambda: model.get_counterfactual_latent(
                target_samples=["sample_0", "sample_0"],
            ),
            "duplicate targets",
        ),
        (
            lambda: model.get_counterfactual_latent(
                inference_mode="posterior_mc",
                n_draws=1,
            ),
            "at least two draws",
        ),
        (
            lambda: model.get_counterfactual_latent(
                inference_mode="posterior_mc",
                n_draws=2,
                quantiles=(0.5, 0.5),
            ),
            "unique",
        ),
        (
            lambda: model.get_counterfactual_latent(
                inference_mode="posterior_mc",
                n_draws=2,
                quantiles=(0.0, 0.5),
            ),
            "strictly inside",
        ),
        (
            lambda: model.get_counterfactual_expression(
                gene_list=["not-a-gene"],
            ),
            "Unknown gene",
        ),
        (
            lambda: model.get_counterfactual_expression(
                protein_list=["not-a-protein"],
            ),
            "Unknown protein",
        ),
        (
            lambda: model.get_counterfactual_expression(
                gene_list=[gene],
                protein_list=[protein, protein],
            ),
            "duplicate features",
        ),
        (
            lambda: model.get_counterfactual_expression(
                batch_policy="not-a-policy",
            ),
            "batch_policy",
        ),
        (
            lambda: model.get_counterfactual_expression(
                batch_policy="specified",
                specified_batch="not-a-batch",
            ),
            "Unsupported specified_batch",
        ),
        (
            lambda: model.get_counterfactual_expression(
                panel_policy="specified",
                specified_panel="not-a-panel",
            ),
            "Unsupported specified_panel",
        ),
        (
            lambda: model.get_counterfactual_expression(
                library_policy="specified",
                specified_library_size=0.0,
            ),
            "finite and positive",
        ),
        (
            lambda: model.get_counterfactual_expression(
                indices=[0, 1],
                library_policy="specified",
                specified_library_size=np.asarray([1.0, 2.0, 3.0]),
            ),
            "cell-aligned vector",
        ),
    )
    for call, match in invalid_calls:
        with pytest.raises(ValueError, match=match):
            call()


def test_counterfactual_expression_schema_specified_library_and_protein_means():
    """Expression output has named deterministic estimands and exact mixture identities."""
    model, adata = _make_v2_model()
    genes = [str(adata.var_names[3]), str(adata.var_names[0])]
    protein_names = list(
        map(
            str,
            model.adata_manager.get_state_registry(
                REGISTRY_KEYS.PROTEIN_EXP_KEY
            ).column_names,
        )
    )
    proteins = [protein_names[2], protein_names[0]]
    batch_labels = list(
        map(
            str,
            model.adata_manager.get_state_registry(
                REGISTRY_KEYS.BATCH_KEY
            ).categorical_mapping,
        )
    )

    torch.manual_seed(1)
    result = model.get_counterfactual_expression(
        adata=adata,
        indices=[0, 2],
        target_samples=["sample_2", "sample_0"],
        gene_list=genes,
        protein_list=proteins,
        inference_mode="latent_mean",
        n_draws=1,
        batch_policy="specified",
        specified_batch=batch_labels[1],
        panel_policy="observed",
        library_policy="specified",
        specified_library_size=10.0,
        batch_size=1,
        target_chunk_size=1,
        feature_chunk_size=1,
    )
    assert dict(result.sizes) == {
        "draw": 1,
        "cell_name": 2,
        "target_sample": 2,
        "gene": 2,
        "protein": 2,
    }
    assert set(result.data_vars) == {
        "rna_scale",
        "rna_rate",
        "protein_background_component_mean",
        "protein_foreground_component_mean",
        "protein_foreground_probability",
        "protein_background_contribution",
        "protein_foreground_contribution",
        "protein_total_mean",
        "protein_batch_efficiency",
        "protein_available",
    }
    assert result.gene.to_numpy().tolist() == genes
    assert result.protein.to_numpy().tolist() == proteins
    assert result.target_sample.to_numpy().tolist() == ["sample_2", "sample_0"]
    assert result["rna_scale"].dtype == np.float32
    assert result["protein_available"].dtype == np.bool_
    np.testing.assert_allclose(
        result["rna_rate"].to_numpy(),
        10.0 * result["rna_scale"].to_numpy(),
        rtol=1e-6,
        atol=1e-7,
    )

    background = result["protein_background_component_mean"]
    foreground = result["protein_foreground_component_mean"]
    probability = result["protein_foreground_probability"]
    np.testing.assert_allclose(
        result["protein_background_contribution"],
        (1.0 - probability) * background,
        rtol=1e-6,
        atol=1e-7,
    )
    np.testing.assert_allclose(
        result["protein_foreground_contribution"],
        probability * foreground,
        rtol=1e-6,
        atol=1e-7,
    )
    np.testing.assert_allclose(
        result["protein_total_mean"],
        result["protein_background_contribution"]
        + result["protein_foreground_contribution"],
        rtol=1e-6,
        atol=1e-7,
    )
    assert result["protein_available"].to_numpy().all()
    assert result.attrs["batch_policy"] == "specified"
    assert result.attrs["library_policy"] == "specified"
    assert result.attrs["protein_mean_formula"].startswith(
        "efficiency * exp(back_alpha"
    )
    assert "context_table_sha256" in result.attrs

    torch.manual_seed(999)
    repeated = model.get_counterfactual_expression(
        adata=adata,
        indices=[0, 2],
        target_samples=["sample_2", "sample_0"],
        gene_list=genes,
        protein_list=proteins,
        batch_policy="specified",
        specified_batch=batch_labels[1],
        library_policy="specified",
        specified_library_size=10.0,
        batch_size=2,
        target_chunk_size=2,
    )
    for variable in result.data_vars:
        np.testing.assert_allclose(
            result[variable].to_numpy(),
            repeated[variable].to_numpy(),
            rtol=1e-6,
            atol=1e-6,
        )


def test_counterfactual_expression_sample_balanced_joint_marginal_contexts():
    """Marginal contexts weight samples equally and preserve joint support."""
    model, adata = _make_context_v2_model()
    batches = list(
        map(
            str,
            model.adata_manager.get_state_registry(
                REGISTRY_KEYS.BATCH_KEY
            ).categorical_mapping,
        )
    )
    panels = list(
        map(
            str,
            model.adata_manager.get_state_registry("panel").categorical_mapping,
        )
    )
    protein_names = list(
        map(
            str,
            model.adata_manager.get_state_registry(
                REGISTRY_KEYS.PROTEIN_EXP_KEY
            ).column_names,
        )
    )
    common = {
        "adata": adata,
        "indices": [0, 1],
        "target_samples": ["sample_0"],
        "gene_list": [str(adata.var_names[0])],
        "protein_list": protein_names[:2],
    }
    marginal = model.get_counterfactual_expression(
        **common,
        batch_policy="sample_balanced_marginal",
        panel_policy="sample_balanced_marginal",
        marginal_reference_indices=np.arange(adata.n_obs),
    )
    context_0 = model.get_counterfactual_expression(
        **common,
        batch_policy="specified",
        specified_batch=batches[0],
        panel_policy="specified",
        specified_panel=panels[0],
    )
    context_1 = model.get_counterfactual_expression(
        **common,
        batch_policy="specified",
        specified_batch=batches[1],
        panel_policy="specified",
        specified_panel=panels[1],
    )

    # Per-sample context-0 proportions are 4/6, 1/4, and 1/2.
    context_0_weight = np.mean([4 / 6, 1 / 4, 1 / 2])
    for variable in ("rna_scale", "rna_rate"):
        np.testing.assert_allclose(
            marginal[variable],
            context_0_weight * context_0[variable]
            + (1.0 - context_0_weight) * context_1[variable],
            rtol=1e-6,
            atol=1e-6,
        )
    assert not marginal["protein_available"].sel(
        protein=protein_names[0]
    ).to_numpy().any()
    assert marginal["protein_available"].sel(
        protein=protein_names[1]
    ).to_numpy().all()
    assert np.isnan(
        marginal["protein_total_mean"].sel(protein=protein_names[0])
    ).all()
    np.testing.assert_allclose(
        marginal["rna_rate"].isel(cell_name=0),
        adata.obs["size_factor"].iloc[0] * marginal["rna_scale"].isel(cell_name=0),
        rtol=1e-6,
        atol=1e-7,
    )

    library_marginal = model.get_counterfactual_expression(
        **common,
        batch_policy="observed",
        panel_policy="observed",
        library_policy="sample_balanced_marginal",
        marginal_reference_indices=np.arange(adata.n_obs),
    )
    sample_equal_library = np.mean([3.5, 8.5, 11.5])
    np.testing.assert_allclose(
        library_marginal["rna_rate"],
        sample_equal_library * library_marginal["rna_scale"],
        rtol=1e-6,
        atol=1e-6,
    )
    specified_vector = model.get_counterfactual_expression(
        **common,
        batch_policy="observed",
        panel_policy="observed",
        library_policy="specified",
        specified_library_size=np.asarray([7.0, 9.0]),
    )
    np.testing.assert_allclose(
        specified_vector["rna_rate"],
        np.asarray([7.0, 9.0])[None, :, None, None]
        * specified_vector["rna_scale"],
        rtol=1e-6,
        atol=1e-7,
    )

    with pytest.raises(ValueError, match="marginalized jointly"):
        model.get_counterfactual_expression(
            **common,
            batch_policy="sample_balanced_marginal",
            panel_policy="observed",
        )
    with pytest.raises(ValueError, match="Unsupported specified_batch"):
        model.get_counterfactual_expression(
            **common,
            batch_policy="specified",
            specified_batch="not-a-batch",
        )


def test_counterfactual_expression_observed_policy_uses_latent_library_mean():
    """Observed policy keeps the factual latent-library expectation when configured."""
    adata = _make_adata(n_samples=3)
    model = MrTotalVI(
        adata,
        sample_key="sample",
        n_latent=3,
        n_latent_u=2,
        hierarchy_mode="centered_v2",
        u_encoder_mode="sample_blind",
        use_observed_lib_size=False,
    )
    model.is_trained_ = True
    with torch.no_grad():
        model.module.encoder.l_gene_mean_encoder.weight.zero_()
        model.module.encoder.l_gene_mean_encoder.bias.fill_(np.log(123.0))
        model.module.encoder.l_gene_var_encoder.weight.zero_()
        model.module.encoder.l_gene_var_encoder.bias.fill_(-20.0)

    indices = np.asarray([0, 2])
    result = model.get_counterfactual_expression(
        adata=adata,
        indices=indices,
        target_samples=["sample_1"],
        gene_list=[str(adata.var_names[0])],
        protein_list=[
            str(
                model.adata_manager.get_state_registry(
                    REGISTRY_KEYS.PROTEIN_EXP_KEY
                ).column_names[0]
            )
        ],
        library_policy="observed",
    )
    expected_library = model.get_latent_library_size(
        adata=adata,
        indices=indices,
        give_mean=True,
    ).reshape(1, indices.size, 1, 1)
    np.testing.assert_allclose(
        result["rna_rate"],
        expected_library * result["rna_scale"],
        rtol=1e-6,
        atol=1e-6,
    )
    assert result.attrs["observed_library_estimand"] == "posterior_lognormal_mean"


def test_counterfactual_expression_posterior_common_noise_and_subsetting():
    """Posterior expression draws and summaries are subset/batch invariant."""
    model, adata = _make_v2_model()
    genes = [str(adata.var_names[index]) for index in (3, 0)]
    proteins = list(
        map(
            str,
            model.adata_manager.get_state_registry(
                REGISTRY_KEYS.PROTEIN_EXP_KEY
            ).column_names[:2],
        )
    )
    full = model.get_counterfactual_expression(
        adata=adata,
        indices=[0, 2],
        target_samples=["sample_2", "sample_0"],
        gene_list=genes,
        protein_list=proteins,
        inference_mode="posterior_mc",
        n_draws=3,
        random_state=91,
        batch_size=1,
        target_chunk_size=1,
    )
    subset = model.get_counterfactual_expression(
        adata=adata,
        indices=[2, 0],
        target_samples=["sample_0"],
        gene_list=[genes[0]],
        protein_list=[proteins[1]],
        inference_mode="posterior_mc",
        n_draws=3,
        random_state=91,
        batch_size=2,
        target_chunk_size=3,
    )
    for variable in ("rna_scale", "rna_rate"):
        np.testing.assert_allclose(
            full[variable]
            .sel(
                cell_name=subset.cell_name,
                target_sample=subset.target_sample,
                gene=subset.gene,
            )
            .to_numpy(),
            subset[variable].to_numpy(),
            rtol=1e-6,
            atol=1e-6,
        )
    for variable in (
        "protein_background_component_mean",
        "protein_foreground_component_mean",
        "protein_foreground_probability",
        "protein_total_mean",
    ):
        np.testing.assert_allclose(
            full[variable]
            .sel(
                cell_name=subset.cell_name,
                target_sample=subset.target_sample,
                protein=subset.protein,
            )
            .to_numpy(),
            subset[variable].to_numpy(),
            rtol=1e-6,
            atol=1e-6,
        )
    for variable in ("rna_scale", "rna_rate", "protein_total_mean"):
        np.testing.assert_allclose(
            full[f"{variable}_posterior_mean"],
            full[variable].mean("draw"),
            rtol=1e-6,
            atol=1e-7,
        )
        assert full[f"{variable}_posterior_quantile"].dims[0] == "quantile"

    with pytest.raises(ValueError, match="duplicate features"):
        model.get_counterfactual_expression(gene_list=[genes[0], genes[0]])
    with pytest.raises(MemoryError, match="512"):
        model.get_counterfactual_expression(
            indices=[0],
            inference_mode="posterior_mc",
            n_draws=100_000_000,
        )


def test_counterfactual_atomic_zarr_roundtrip_and_refusal(tmp_path, monkeypatch):
    """Both public datasets round-trip lazily through atomic region stores."""
    model, adata = _make_v2_model()
    latent_memory = model.get_counterfactual_latent(
        adata=adata,
        indices=[0, 2],
        target_samples=["sample_2", "sample_0"],
        inference_mode="posterior_mc",
        n_draws=2,
        random_state=5,
    )
    latent_path = tmp_path / "latent.zarr"
    latent_zarr = model.get_counterfactual_latent(
        adata=adata,
        indices=[0, 2],
        target_samples=["sample_2", "sample_0"],
        inference_mode="posterior_mc",
        n_draws=2,
        random_state=5,
        zarr_path=latent_path,
        zarr_chunks={"draw": 1, "cell_name": 1, "target_sample": 1},
    )
    assert latent_path.is_dir()
    assert hasattr(latent_zarr["z"].data, "chunks")
    for variable in latent_memory.data_vars:
        np.testing.assert_equal(
            latent_zarr[variable].to_numpy(),
            latent_memory[variable].to_numpy(),
        )
    assert not list(tmp_path.glob(".latent.zarr.tmp-*"))
    with pytest.raises(FileExistsError, match="Refusing"):
        model.get_counterfactual_latent(
            indices=[0],
            zarr_path=latent_path,
        )

    expression_memory = model.get_counterfactual_expression(
        adata=adata,
        indices=[0, 2],
        target_samples=["sample_1"],
        gene_list=[str(adata.var_names[0])],
        protein_list=[
            str(
                model.adata_manager.get_state_registry(
                    REGISTRY_KEYS.PROTEIN_EXP_KEY
                ).column_names[0]
            )
        ],
    )
    expression_path = tmp_path / "expression.zarr"
    expression_zarr = model.get_counterfactual_expression(
        adata=adata,
        indices=[0, 2],
        target_samples=["sample_1"],
        gene_list=[str(adata.var_names[0])],
        protein_list=[
            str(
                model.adata_manager.get_state_registry(
                    REGISTRY_KEYS.PROTEIN_EXP_KEY
                ).column_names[0]
            )
        ],
        zarr_path=expression_path,
        zarr_chunks={
            "draw": 1,
            "cell_name": 1,
            "target_sample": 1,
            "gene": 1,
            "protein": 1,
        },
    )
    assert hasattr(expression_zarr["rna_rate"].data, "chunks")
    for variable in expression_memory.data_vars:
        np.testing.assert_equal(
            expression_zarr[variable].to_numpy(),
            expression_memory[variable].to_numpy(),
        )
    assert "chunks" in expression_zarr.attrs
    assert not list(tmp_path.glob(".expression.zarr.tmp-*"))

    import builtins

    real_import = builtins.__import__

    def block_parallel_storage_imports(name, *args, **kwargs):
        if name in {"dask", "zarr"}:
            raise ImportError(name)
        return real_import(name, *args, **kwargs)

    missing_dependency_path = tmp_path / "missing-dependency.zarr"
    with monkeypatch.context() as context:
        context.setattr(
            builtins,
            "__import__",
            block_parallel_storage_imports,
        )
        with pytest.raises(ImportError, match="parallel"):
            model.get_counterfactual_latent(
                indices=[0],
                zarr_path=missing_dependency_path,
            )
    assert not missing_dependency_path.exists()

    import scvi.external.mrtotalvi._counterfactual as counterfactual_module

    streamed_memory = model.get_counterfactual_expression(
        indices=np.arange(adata.n_obs),
        target_samples=["sample_1"],
        gene_list=[str(adata.var_names[0])],
        protein_list=[
            str(
                model.adata_manager.get_state_registry(
                    REGISTRY_KEYS.PROTEIN_EXP_KEY
                ).column_names[0]
            )
        ],
    )
    streamed_path = tmp_path / "streamed-expression.zarr"
    original_limit = counterfactual_module.MAX_IN_MEMORY_BYTES
    counterfactual_module.MAX_IN_MEMORY_BYTES = 100
    try:
        streamed = model.get_counterfactual_expression(
            indices=np.arange(adata.n_obs),
            target_samples=["sample_1"],
            gene_list=[str(adata.var_names[0])],
            protein_list=[
                str(
                    model.adata_manager.get_state_registry(
                        REGISTRY_KEYS.PROTEIN_EXP_KEY
                    ).column_names[0]
                )
            ],
            zarr_path=streamed_path,
            zarr_chunks={"cell_name": 2},
        )
    finally:
        counterfactual_module.MAX_IN_MEMORY_BYTES = original_limit
    assert streamed.attrs["storage_mode"] == "atomic_zarr_cell_regions"
    for variable in streamed_memory.data_vars:
        if np.issubdtype(streamed_memory[variable].dtype, np.floating):
            np.testing.assert_allclose(
                streamed[variable].to_numpy(),
                streamed_memory[variable].to_numpy(),
                rtol=1e-6,
                atol=1e-6,
            )
        else:
            np.testing.assert_equal(
                streamed[variable].to_numpy(),
                streamed_memory[variable].to_numpy(),
            )

    import zarr

    failure_path = tmp_path / "failed.zarr"

    def fail_consolidation(*args, **kwargs):
        raise RuntimeError("injected consolidation failure")

    monkeypatch.setattr(zarr, "consolidate_metadata", fail_consolidation)
    with pytest.raises(RuntimeError, match="injected"):
        model.get_counterfactual_latent(
            indices=[0],
            zarr_path=failure_path,
        )
    assert not failure_path.exists()
    assert not list(tmp_path.glob(".failed.zarr.tmp-*"))


def test_local_sample_enrichment_self_exclusion_singletons_and_group_logmeanexp():
    """Descriptive densities exclude only factual selves and weight samples equally."""
    model, adata = _make_enrichment_v2_model()
    singleton = model.local_sample_enrichment(
        adata=adata,
        indices=[0],
        target_samples=["sample_0", "sample_1"],
        reference_indices=[0, 1],
    )
    assert dict(singleton.sizes) == {
        "draw": 1,
        "cell_name": 1,
        "target_sample": 2,
    }
    assert singleton["self_excluded"].sel(target_sample="sample_0").item()
    assert not singleton["self_excluded"].sel(target_sample="sample_1").item()
    assert singleton["n_reference_cells"].sel(target_sample="sample_0").item() == 0
    assert singleton["n_reference_cells"].sel(target_sample="sample_1").item() == 1
    assert not singleton["finite_support"].sel(target_sample="sample_0").item()
    assert singleton["finite_support"].sel(target_sample="sample_1").item()
    assert np.isnan(
        singleton["log_density"].sel(target_sample="sample_0")
    ).all()
    assert np.isfinite(
        singleton["log_density"].sel(target_sample="sample_1")
    ).all()

    grouped = model.local_sample_enrichment(
        adata=adata,
        indices=[0, 1],
        group_key="group",
        contrast=("numerator", "denominator"),
        donor_key="donor",
        inference_mode="posterior_mc",
        n_draws=3,
        random_state=7,
        batch_size=1,
        reference_chunk_size=1,
    )
    assert grouped["group_log_density"].dims == (
        "draw",
        "cell_name",
        "group",
    )
    assert grouped.group.to_numpy().tolist() == ["numerator", "denominator"]
    numerator = grouped["log_density"].sel(
        target_sample=["sample_0", "sample_2"]
    )
    expected_numerator = np.logaddexp(
        numerator.isel(target_sample=0),
        numerator.isel(target_sample=1),
    ) - np.log(2.0)
    np.testing.assert_allclose(
        grouped["group_log_density"].sel(group="numerator"),
        expected_numerator,
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        grouped["log_ratio"],
        grouped["group_log_density"].sel(group="numerator")
        - grouped["group_log_density"].sel(group="denominator"),
        rtol=1e-6,
        atol=1e-6,
    )
    assert grouped["donor_log_ratio"].dims == (
        "draw",
        "cell_name",
        "donor",
    )
    np.testing.assert_allclose(
        grouped["donor_log_ratio"].sel(donor="donor_0"),
        grouped["log_density"].sel(target_sample="sample_0")
        - grouped["log_density"].sel(target_sample="sample_1"),
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        grouped["donor_log_ratio_mean"],
        grouped["donor_log_ratio"].mean("donor"),
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        grouped["donor_log_ratio_median"],
        grouped["donor_log_ratio"].median("donor"),
        rtol=1e-6,
        atol=1e-6,
    )
    assert "log_density_posterior_mean" in grouped
    assert grouped.attrs["interpretation"].endswith("non-causal")

    with pytest.raises(ValueError, match="exactly one numerator"):
        model.local_sample_enrichment(
            adata=adata,
            target_samples=["sample_0", "sample_1", "sample_2"],
            group_key="group",
            contrast=("numerator", "denominator"),
            donor_key="donor",
        )
    adata.obs.loc[adata.obs["sample"] == "sample_0", "bad_group"] = [
        "a",
        "b",
    ]
    adata.obs["bad_group"] = adata.obs["bad_group"].fillna("c")
    with pytest.raises(ValueError, match="constant within sample"):
        model.local_sample_enrichment(
            adata=adata,
            group_key="bad_group",
        )


def test_combine_mrtotalvi_seed_results_separates_draw_and_seed_uncertainty():
    """Seed aggregation retains draws and never pools uncertainty sources."""
    import xarray as xr

    first = xr.Dataset(
        {
            "value": (
                ("draw", "cell_name"),
                np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
            ),
            "support": ("cell_name", np.asarray([True, False])),
        },
        coords={"draw": [0, 1], "cell_name": ["a", "b"]},
        attrs={"schema_version": "test-schema"},
    )
    second = first.copy(deep=True)
    second["value"] = second["value"] + 4.0

    result = combine_mrtotalvi_seed_results({11: second, 3: first})
    assert result.training_seed.to_numpy().tolist() == [3, 11]
    assert result["value"].dims == ("training_seed", "draw", "cell_name")
    np.testing.assert_allclose(
        result["value_within_seed_posterior_mean"],
        result["value"].mean("draw"),
    )
    np.testing.assert_allclose(
        result["value_within_seed_posterior_sd"],
        result["value"].std("draw", ddof=1),
    )
    np.testing.assert_allclose(
        result["value_between_seed_mean"],
        result["value_within_seed_posterior_mean"].mean("training_seed"),
    )
    np.testing.assert_allclose(
        result["value_between_seed_sd"],
        result["value_within_seed_posterior_mean"].std(
            "training_seed",
            ddof=1,
        ),
    )
    assert "draw" in result.dims
    assert result.attrs["uncertainty_separation"] == (
        "within-seed posterior draws and between-seed training variation "
        "are reported separately and never pooled"
    )

    with pytest.raises(ValueError, match="non-empty"):
        combine_mrtotalvi_seed_results({})


def test_centered_local_representation_routes_through_full_universe():
    """Existing local-representation entry points use centered-v2 semantics."""
    model, adata = _make_v2_model()
    indices = np.asarray([0, 1, 2])
    counterfactual = model.get_counterfactual_latent(
        adata=adata,
        indices=indices,
    )
    local = model.get_local_sample_representation(
        adata=adata,
        indices=indices,
        batch_size=1,
    )
    np.testing.assert_allclose(
        local.to_numpy(),
        counterfactual["z"].isel(draw=0).to_numpy(),
        rtol=1e-6,
        atol=1e-6,
    )

    observed_labels = adata.obs["sample"].iloc[indices].astype(str).to_numpy()
    expected_factual = np.stack(
        [
            counterfactual["z"]
            .isel(draw=0, cell_name=position)
            .sel(target_sample=sample)
            .to_numpy()
            for position, sample in enumerate(observed_labels)
        ]
    )
    np.testing.assert_allclose(
        model.get_latent_representation(
            adata=adata,
            indices=indices,
            give_mean=True,
            give_z=True,
            batch_size=2,
        ),
        expected_factual,
        rtol=1e-6,
        atol=1e-6,
    )

    distances = model.get_local_sample_distances(
        adata=adata,
        indices=indices,
        batch_size=2,
    )
    expected_distances = np.sqrt(
        np.square(
            local.to_numpy()[:, :, None, :]
            - local.to_numpy()[:, None, :, :]
        ).sum(axis=-1)
    )
    np.testing.assert_allclose(
        distances,
        expected_distances,
        rtol=1e-6,
        atol=1e-6,
    )


def test_centered_de_fails_closed_and_grouped_da_only_warns(monkeypatch):
    """Unvalidated v2 DE is refused; grouped DA keeps its exact delegated output."""
    centered, adata = _make_v2_model()
    with pytest.raises(RuntimeError, match="not validated for centered_v2"):
        centered.differential_expression(adata=adata)

    legacy_adata = _make_adata()
    legacy = MrTotalVI(
        legacy_adata,
        sample_key="sample",
        n_latent=3,
        n_latent_u=2,
    )
    legacy.is_trained_ = True
    sentinel = object()
    monkeypatch.setattr(
        "scvi.external.mrtotalvi._model._differential_abundance",
        lambda *args, **kwargs: sentinel,
    )
    with pytest.warns(UserWarning, match="descriptive and non-inferential"):
        assert (
            legacy.differential_abundance(
                sample_cov_keys=["sample"],
            )
            is sentinel
        )
    import warnings

    with warnings.catch_warnings(record=True) as warning_record:
        warnings.simplefilter("always")
        assert legacy.differential_abundance() is sentinel
    assert not warning_record
