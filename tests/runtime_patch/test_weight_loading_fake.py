# SPDX-License-Identifier: Apache-2.0
"""Contracts for Kimi metadata-only fake safetensors loading."""

from __future__ import annotations

import sys
from types import ModuleType

import torch
from safetensors.torch import save_file

from vllm_hcu.runtime_compat import weight_loading


def test_fake_iterator_uses_metadata_and_preserves_safe_tensor_contract(
    monkeypatch,
    tmp_path,
):
    checkpoint = tmp_path / "weights.safetensors"
    save_file(
        {
            "language_model.model.embed_tokens.weight": torch.full(
                (2, 3), 9, dtype=torch.bfloat16
            ),
            "language_model.model.layers.0.norm.weight": torch.full(
                (3,), 9, dtype=torch.bfloat16
            ),
            "language_model.model.layers.0.experts.0.weight": torch.full(
                (2, 2), -7, dtype=torch.int8
            ),
            "language_model.model.layers.0.experts.0.weight_scale": torch.full(
                (2, 1), -7, dtype=torch.float32
            ),
            "language_model.model.layers.1.norm.weight": torch.full(
                (3,), 9, dtype=torch.bfloat16
            ),
        },
        str(checkpoint),
    )

    weight_utils = ModuleType("vllm.model_executor.model_loader.weight_utils")
    default_loader = ModuleType("vllm.model_executor.model_loader.default_loader")
    original_calls = []

    def original_iterator(*args, **kwargs):
        original_calls.append((args, kwargs))
        raise AssertionError("fake mode must not call the payload iterator")
        yield  # pragma: no cover

    weight_utils.safetensors_weights_iterator = original_iterator
    default_loader.safetensors_weights_iterator = original_iterator
    monkeypatch.setitem(sys.modules, weight_utils.__name__, weight_utils)
    monkeypatch.setenv("VLLM_HCU_KIMI_FAKE_WEIGHT_DEBUG", "1")
    monkeypatch.setenv("VLLM_HCU_KIMI_FAKE_WEIGHT_LAYERS", "1")

    weight_loading.install_weight_debug_skip_compat(weight_utils, default_loader)
    weights = dict(
        default_loader.safetensors_weights_iterator(
            [str(checkpoint)],
            use_tqdm_on_load=False,
            safetensors_load_strategy="lazy",
        )
    )

    assert original_calls == []
    assert set(weights) == {
        "language_model.model.embed_tokens.weight",
        "language_model.model.layers.0.norm.weight",
        "language_model.model.layers.0.experts.0.weight",
        "language_model.model.layers.0.experts.0.weight_scale",
    }
    assert weights["language_model.model.embed_tokens.weight"].shape == (2, 3)
    assert weights["language_model.model.embed_tokens.weight"].dtype == torch.bfloat16
    assert torch.equal(
        weights["language_model.model.layers.0.norm.weight"],
        torch.ones(3, dtype=torch.bfloat16),
    )
    assert torch.equal(
        weights["language_model.model.layers.0.experts.0.weight"],
        torch.full((2, 2), 0x11, dtype=torch.int8),
    )
    assert torch.all(
        weights["language_model.model.layers.0.experts.0.weight_scale"] > 0
    )


def test_fake_iterator_rejects_non_lazy_safetensors_strategy(monkeypatch, tmp_path):
    checkpoint = tmp_path / "weights.safetensors"
    save_file({"weight": torch.ones(1)}, str(checkpoint))
    weight_utils = ModuleType("vllm.model_executor.model_loader.weight_utils")
    default_loader = ModuleType("vllm.model_executor.model_loader.default_loader")

    def original_iterator(*args, **kwargs):
        yield from ()

    weight_utils.safetensors_weights_iterator = original_iterator
    default_loader.safetensors_weights_iterator = original_iterator
    monkeypatch.setenv("VLLM_HCU_KIMI_FAKE_WEIGHT_DEBUG", "1")
    weight_loading.install_weight_debug_skip_compat(weight_utils, default_loader)

    try:
        next(
            default_loader.safetensors_weights_iterator(
                [str(checkpoint)],
                use_tqdm_on_load=False,
                safetensors_load_strategy="eager",
            )
        )
    except ValueError as error:
        assert "requires the default lazy safetensors loader" in str(error)
    else:
        raise AssertionError("fake mode must reject eager safetensors loading")
