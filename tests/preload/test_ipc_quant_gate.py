# SPDX-License-Identifier: Apache-2.0
"""Contract tests for the IPC quantization allowlist installed by the plugin.

The allowlist predicate is the only scheme-level gate; per-model kernel and
layout checks live in the quantization patches. These tests pin the accepted
scheme to the channel-FP8 / token-dynamic-activation configuration that was
validated on HCU and prove every nearby variant stays rejected.
"""
import copy
from types import SimpleNamespace

import pytest

from vllm_hcu.runtime_compat.weight_cache import (
    _is_channel_fp8_quant_config,
    install_ipc_quant_support,
)

CHANNEL_FP8_CONFIG = {
    "config_groups": {
        "group_0": {
            "targets": ["Linear"],
            "weights": {
                "dynamic": False,
                "group_size": None,
                "num_bits": 8,
                "observer": "minmax",
                "observer_kwargs": {},
                "strategy": "channel",
                "symmetric": True,
                "type": "float",
            },
            "input_activations": {
                "dynamic": True,
                "group_size": None,
                "num_bits": 8,
                "observer": "minmax",
                "observer_kwargs": {},
                "strategy": "token",
                "symmetric": True,
                "type": "float",
            },
        }
    },
    "format": "float-quantized",
    "ignore": ["re:.*attn.wo_a.*"],
    "quant_method": "compressed-tensors",
}


def _set(path, value):
    def mutate(config):
        target = config
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = value
    return mutate


def _remove(path):
    def mutate(config):
        target = config
        for key in path[:-1]:
            target = target[key]
        del target[path[-1]]
    return mutate


def test_reference_channel_fp8_scheme_is_accepted():
    assert _is_channel_fp8_quant_config(CHANNEL_FP8_CONFIG)


@pytest.mark.parametrize(
    "mutate",
    [
        _set(("format",), "pack-quantized"),
        _set(("weight_block_size",), [128, 128]),
        _set(("config_groups", "group_0", "weights", "strategy"), "tensor"),
        _set(("config_groups", "group_0", "weights", "strategy"), "group"),
        _set(("config_groups", "group_0", "weights", "dynamic"), True),
        _set(("config_groups", "group_0", "weights", "symmetric"), False),
        _set(("config_groups", "group_0", "weights", "group_size"), 32),
        _set(("config_groups", "group_0", "weights", "num_bits"), 4),
        _set(("config_groups", "group_0", "weights", "type"), "int"),
        _set(("config_groups", "group_0", "input_activations", "dynamic"), False),
        _set(("config_groups", "group_0", "input_activations", "strategy"), "tensor"),
        _set(("config_groups", "group_0", "input_activations", "num_bits"), 16),
        _set(("config_groups", "group_0", "targets"), ["Linear", "MoE"]),
        _remove(("config_groups", "group_0", "input_activations")),
        _remove(("config_groups",)),
    ],
)
def test_unverified_variants_are_rejected(mutate):
    config = copy.deepcopy(CHANNEL_FP8_CONFIG)
    mutate(config)
    assert not _is_channel_fp8_quant_config(config)


def test_non_mapping_config_is_rejected():
    assert not _is_channel_fp8_quant_config(None)
    assert not _is_channel_fp8_quant_config("compressed-tensors")


def test_install_registers_all_entries_idempotently():
    class Protocol:
        IPC_QUANT_ALLOWLIST = {None: lambda _config: True}

    install_ipc_quant_support(Protocol)
    compressed = Protocol.IPC_QUANT_ALLOWLIST["compressed-tensors"]
    assert compressed(CHANNEL_FP8_CONFIG)
    install_ipc_quant_support(Protocol)
    assert Protocol.IPC_QUANT_ALLOWLIST["compressed-tensors"] is compressed
    assert "slimquant_w4a8" in Protocol.IPC_QUANT_ALLOWLIST


def test_install_restores_a_missing_entry_while_another_is_present():
    class Protocol:
        IPC_QUANT_ALLOWLIST = {None: lambda _config: True}

    install_ipc_quant_support(Protocol)
    del Protocol.IPC_QUANT_ALLOWLIST["compressed-tensors"]
    install_ipc_quant_support(Protocol)
    assert "compressed-tensors" in Protocol.IPC_QUANT_ALLOWLIST


def _model_config(quantization, quant_config):
    return SimpleNamespace(
        quantization=quantization,
        model_arch_config=SimpleNamespace(quantization_config=quant_config),
        hf_config=SimpleNamespace(),
    )


def test_installed_protocol_accepts_reference_and_rejects_variants():
    from vllm.model_executor.model_loader.weight_cache import protocol

    install_ipc_quant_support(protocol)
    protocol.check_ipc_quant_support(
        _model_config("compressed-tensors", copy.deepcopy(CHANNEL_FP8_CONFIG)),
        where="test",
    )
    rejected = copy.deepcopy(CHANNEL_FP8_CONFIG)
    rejected["config_groups"]["group_0"]["weights"]["strategy"] = "tensor"
    with pytest.raises(protocol.UnsupportedQuantForIPCError):
        protocol.check_ipc_quant_support(
            _model_config("compressed-tensors", rejected), where="test"
        )
