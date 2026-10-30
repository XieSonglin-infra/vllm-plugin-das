from types import SimpleNamespace as NS

import pytest
import torch

from vllm_hcu.patch.worker.op_opt.patch_compressed_tensors_moe_w8a8_fp8 import (
    _restore_preprocessed_channel_fp8,
)


def _case():
    module = NS(QuantizationStrategy=NS(CHANNEL="channel", TOKEN="token"))
    parallel = NS(tp_size=1, dp_size=1, ep_size=1, pcp_size=1, sp_size=1)
    method = NS(weight_quant=NS(strategy="channel"), input_quant=NS(strategy="token"),
                fp8_backend="HCU_DEEPGEMM", static_input_scales=False,
                moe=NS(moe_parallel_config=parallel))
    layer = NS(local_num_experts=1,
               w13_weight=torch.empty((1, 64, 256, 4, 16, 16), dtype=torch.float8_e4m3fn, device="meta"),
               w2_weight=torch.empty((1, 32, 256, 4, 16, 16), dtype=torch.float8_e4m3fn, device="meta"),
               w13_weight_scale=torch.empty((1, 4096, 1), dtype=torch.float32, device="meta"),
               w2_weight_scale=torch.empty((1, 4096, 1), dtype=torch.float32, device="meta"))
    return module, method, layer


def test_restore_validated_single_expert_layout():
    module, method, layer = _case()
    weight = layer.w13_weight
    _restore_preprocessed_channel_fp8(method, layer, module)
    assert layer._dsv4_channel_deepgemm_layout == "contiguous"
    assert layer.w13_weight is weight


@pytest.mark.parametrize("change", ["topology", "shape", "dtype", "stride", "scheme"])
def test_restore_rejects_unvalidated_contract(change):
    module, method, layer = _case()
    if change == "topology":
        method.moe.moe_parallel_config.dp_size = 8
    elif change == "shape":
        layer.local_num_experts = 2
    elif change == "dtype":
        layer.w2_weight_scale = layer.w2_weight_scale.to(torch.float16)
    elif change == "stride":
        layer.w13_weight = layer.w13_weight.transpose(-1, -2)
    else:
        method.input_quant.strategy = "tensor"
    with pytest.raises(RuntimeError, match="unsupported preprocessed"):
        _restore_preprocessed_channel_fp8(method, layer, module)


def test_restore_shared_auto_requires_explicit_metadata_and_exact_topology():
    module, method, layer = _case()
    method.moe.moe_parallel_config = NS(tp_size=1, dp_size=8, ep_size=8,
                                         pcp_size=1, sp_size=1,
                                         all2all_backend="deepep_auto")
    layer.local_num_experts = 32
    layer.w13_weight = torch.empty((32, 64, 256, 4, 16, 16), dtype=torch.float8_e4m3fn, device="meta")
    layer.w2_weight = torch.empty((32, 32, 256, 4, 16, 16), dtype=torch.float8_e4m3fn, device="meta")
    layer.w13_weight_scale = torch.empty((32, 4096, 1), dtype=torch.float32, device="meta")
    layer.w2_weight_scale = torch.empty((32, 4096, 1), dtype=torch.float32, device="meta")
    with pytest.raises(RuntimeError, match="unsupported preprocessed"):
        _restore_preprocessed_channel_fp8(method, layer, module)
    layer.w13_weight._hcu_dsv4_deepgemm_layout = "contiguous=masked"
    _restore_preprocessed_channel_fp8(method, layer, module)
    assert layer._dsv4_channel_deepgemm_layout == "contiguous=masked"
    method.moe.moe_parallel_config.dp_size = 4
    with pytest.raises(RuntimeError, match="unsupported preprocessed"):
        _restore_preprocessed_channel_fp8(method, layer, module)


def test_export_attaches_layout_to_packed_weight_and_rejects_unknown():
    from vllm_hcu.runtime_compat.weight_cache import install_expert_export_metadata
    model = torch.nn.Module()
    model.register_parameter("w13_weight", torch.nn.Parameter(torch.ones(1)))
    model._dsv4_channel_deepgemm_layout = "contiguous=masked"
    entry = NS()
    module = NS(export_entries=lambda _model: ({"w13_weight": entry}, {}))
    install_expert_export_metadata(module)
    entries, _ = module.export_entries(model)
    assert entries["w13_weight"]._hcu_dsv4_deepgemm_layout == "contiguous=masked"
    model._dsv4_channel_deepgemm_layout = "contiguous+masked"
    with pytest.raises(RuntimeError, match="unsupported DeepGEMM IPC layout"):
        module.export_entries(model)
