"""CPU-level contracts for the HCU-owned Kimi-K3 plugin surface."""

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
    pack_int4_twos_complement,
    unpack_int4_twos_complement,
    validate_kimi_k3_w4a8_metadata,
)
from vllm_hcu.model_executor.layers.quantization.slimquant_facade import (
    KimiK3W4A8Facade,
)


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
