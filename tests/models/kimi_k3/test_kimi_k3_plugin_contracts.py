"""CPU-level contracts for the HCU-owned Kimi-K3 plugin surface."""

from types import SimpleNamespace

import pytest
import torch

from vllm_hcu.patch.platform.core_fix import (
    patch_kimi_k3_config,
    patch_kimi_k3_model_config,
)
from vllm_hcu.patch.worker.core_fix import patch_kimi_k3_model
from vllm_hcu.models.kimi_k3.amd.ops.situ import SituAndMul
from vllm_hcu.model_executor.layers.quantization.kimi_k3_w4a8 import (
    KIMI_K3_W4A8_PACKING,
    KimiK3W4A8MoEMethod,
    pack_int4_twos_complement,
    unpack_int4_twos_complement,
    validate_kimi_k3_w4a8_metadata,
)
from vllm_hcu.model_executor.layers.quantization.slimquant_facade import (
    KimiK3W4A8Facade,
)


def _ht_moe_config(ht=True, ll=False):
    return SimpleNamespace(
        activation=SimpleNamespace(value="situ"), swiglu_beta=None,
        moe_parallel_config=SimpleNamespace(
            use_deepep_ht_kernels=ht, use_deepep_ll_kernels=ll),
    )


def test_kimi_ht_ll_selection_rejects_ambiguous_backend():
    with pytest.raises(ValueError, match="simultaneously"):
        KimiK3W4A8MoEMethod(_ht_moe_config(ll=True))


def test_kimi_ht_quant_config_uses_int8_token_scales_and_high_nibble_compensation():
    method = KimiK3W4A8MoEMethod(_ht_moe_config())
    layer = SimpleNamespace(w13_weight_scale=torch.ones(2, 64, 1),
                            w2_weight_scale=torch.ones(2, 32, 1))
    quant = method.get_fused_moe_quant_config(layer)
    assert quant is not None
    assert quant.quant_dtype == torch.int8
    assert quant.per_act_token_quant
    assert torch.equal(quant.w1_scale, layer.w13_weight_scale * 16)
    assert torch.equal(quant.w2_scale, layer.w2_weight_scale * 16)
    assert torch.equal(layer.w13_weight_scale, torch.ones(2, 64, 1))


def test_kimi_ht_cannot_fall_back_to_direct_aiter_apply():
    method = KimiK3W4A8MoEMethod(_ht_moe_config())
    with pytest.raises(RuntimeError, match="modular"):
        method.apply(None, None, None, None, None, None)


def test_kimi_feature_off_has_no_modular_quant_config():
    method = KimiK3W4A8MoEMethod(_ht_moe_config(ht=False))
    assert method.get_fused_moe_quant_config(None) is None


def test_kimi_ht_situ_quantization_preserves_source_parameters():
    from vllm_hcu.model_executor.layers.quantization.kimi_k3_ht_runtime import (
        KimiK3HTExperts,
    )
    expert = object.__new__(KimiK3HTExperts)
    expert._situ_beta, expert._situ_linear_beta = 1.25, 2.0
    calls = []
    result = (object(), object())
    expert._situ_quant = lambda x, **kw: (calls.append((x, kw)) or result)
    gateup = object()
    assert expert._quantize_activation(gateup, None, None) is result
    assert calls == [(gateup, {"beta": 1.25, "linear_beta": 2.0})]
    assert expert._permute_scale_kwargs(3584) == {"block_size": 3584}
    assert expert.adjust_N_for_activation(2048, "situ") == 1024
    with pytest.raises(ValueError, match="SiTU"):
        expert._validate_activation("silu")


def test_kimi_ht_rejects_batched_expert_format():
    from vllm.model_executor.layers.fused_moe.modular_kernel import (
        FusedMoEActivationFormat,
    )
    method = KimiK3W4A8MoEMethod(_ht_moe_config())
    pf = SimpleNamespace(activation_format=FusedMoEActivationFormat.BatchedExperts)
    with pytest.raises(ValueError, match="Standard"):
        method.select_gemm_impl(pf, None)


def test_kimi_ht_selects_contiguous_expert_and_binds_loaded_weights(monkeypatch):
    from vllm.model_executor.layers.fused_moe.modular_kernel import (
        FusedMoEActivationFormat,
    )
    from vllm_hcu.model_executor.layers.quantization import kimi_k3_ht_runtime

    calls = []

    class Expert:
        def __init__(self, moe, quant):
            calls.append((moe, quant))

        def process_weights_after_loading(self, layer):
            calls.append(layer)

    monkeypatch.setattr(kimi_k3_ht_runtime, "KimiK3HTExperts", Expert)
    method = KimiK3W4A8MoEMethod(_ht_moe_config())
    method.moe_quant_config = object()
    layer = object()
    pf = SimpleNamespace(activation_format=FusedMoEActivationFormat.Standard)
    assert isinstance(method.select_gemm_impl(pf, layer), Expert)
    assert calls == [(method.moe, method.moe_quant_config), layer]


def test_kimi_ht_missing_situ_api_fails_before_weight_packing(monkeypatch):
    monkeypatch.setenv("VLLM_HCU_KIMI_HT_SITU_BACKEND", "lightop")
    import sys
    from vllm_hcu.model_executor.layers.quantization.kimi_k3_ht_runtime import (
        KimiK3HTExperts, DeepEPDeepGemmW4A8ContiguousExperts,
    )
    monkeypatch.setattr(DeepEPDeepGemmW4A8ContiguousExperts, "__init__",
                        lambda *args: None)
    monkeypatch.setitem(sys.modules, "lightop", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "lightop.activation", SimpleNamespace())
    config = SimpleNamespace(activation_situ_beta=1.0,
                             activation_situ_linear_beta=2.0)
    with pytest.raises(ImportError, match="fuse_situ_mul_quant"):
        KimiK3HTExperts(config, None)


def test_kimi_k3_w4a8_nibble_round_trip():
    values = torch.tensor([[-8, -1, 0, 7, 1, -7]], dtype=torch.int8)
    packed = pack_int4_twos_complement(values)
    assert packed.tolist() == [[0x8F, 0x07, 0x19]]
    assert torch.equal(unpack_int4_twos_complement(packed), values)


def test_kimi_k3_w4a8_metadata_rejects_wrong_packing():
    config = {
        "quant_method": "kimi_k3_w4a8",
        "format": "kimi-k3-int4-w4a8-v1",
        "model_version": "k3-test",
        "weight_bits": 4,
        "activation_bits": 8,
        "group_size": 32,
        "symmetric": True,
        "scale_dtype": "float32",
        "packing": "wrong",
        "num_experts": 2,
        "top_k": 1,
        "hidden_size": 32,
        "intermediate_size": 32,
    }
    try:
        validate_kimi_k3_w4a8_metadata(config)
    except ValueError as exc:
        assert "packing" in str(exc)
    else:
        raise AssertionError("invalid packing must fail closed")


@pytest.mark.parametrize(
    ("config", "message"),
    [
        (
            {
                "quant_method": "slimquant_w4a8",
                "num_experts": 2,
                "top_k": 3,
                "hidden_size": 32,
                "intermediate_size": 32,
            },
            "top_k <= num_experts",
        ),
        (
            {
                "quant_method": "slimquant_w4a8",
                "num_experts": 2,
                "top_k": 1,
                "hidden_size": 31,
                "intermediate_size": 32,
            },
            "divisible by 32",
        ),
    ],
)
def test_slimquant_w4a8_metadata_rejects_invalid_kernel_dimensions(config, message):
    with pytest.raises(ValueError, match=message):
        validate_kimi_k3_w4a8_metadata(config)


def test_situ_activation_shape_and_finiteness():
    activation = SituAndMul(beta=1.0, linear_beta=2.0)
    output = activation(torch.randn(4, 8))
    assert output.shape == (4, 4)
    assert torch.isfinite(output).all()


def test_packing_contract_constant_is_explicit():
    assert KIMI_K3_W4A8_PACKING == "twos-complement-high-even-low-odd"


def test_kimi_cli_quant_facade_materializes_without_hf_metadata():
    facade = KimiK3W4A8Facade()
    facade.maybe_update_config("Kimi-K3-INT4")
    assert facade.get_quant_method(torch.nn.Linear(4, 4), "model.layers.0") is None


def test_kimi_k3_patch_inventory_is_exact_and_version_neutral():
    assert patch_kimi_k3_config.TARGET_MODULE == "vllm.transformers_utils.config"
    assert patch_kimi_k3_model_config.TARGET_MODULE == "vllm.model_executor.models.config"
    assert patch_kimi_k3_model.TARGET_MODULE == "vllm.model_executor.models.registry"
    assert all(item.startswith("_vllm_hcu_") for item in (
        patch_kimi_k3_config._MARKER,
        patch_kimi_k3_model_config._MARKER,
        patch_kimi_k3_model._MARKER,
    ))
