"""Check Kimi PP auxiliary capture against a single decoder execution."""
from types import ModuleType, SimpleNamespace

import pytest
import torch
from vllm.model_executor.models.interfaces import EagleModelMixin
from vllm.sequence import IntermediateTensors
from vllm_hcu.patch.worker.op_opt import patch_kimi_k3_pp_aux as patch


@pytest.mark.parametrize("attn_res", [False, True])
def test_aux_values_and_order_across_pipeline_boundary(attn_res):
    pp = SimpleNamespace(world_size=2, is_first_rank=True, is_last_rank=False)

    class Decoder(EagleModelMixin):
        def __init__(self, **kwargs):
            pass

        def forward(self, *args, **kwargs):
            raise AssertionError("PP auxiliary forward was bypassed")

        def _maybe_add_hidden_state(self, states, idx, hidden, residual):
            return super()._maybe_add_hidden_state(
                states, idx, hidden, None if attn_res else residual
            )

    module = ModuleType(patch.TARGET_MODULE)
    module.torch = torch
    module.IntermediateTensors = IntermediateTensors
    module.get_pp_group = lambda: pp
    module.cdiv = lambda a, b: (a + b - 1) // b
    module._apply_attn_res = lambda hidden, *args: hidden
    module.KimiLinearModel = Decoder
    patch.apply_to_module(module)

    def layer(positions, hidden_states, residual):
        return hidden_states + 1, (residual if attn_res else torch.zeros_like(hidden_states))

    def decoder(start, end, upstream):
        obj = object.__new__(Decoder)
        obj.config = SimpleNamespace(attn_res_block_size=12 if attn_res else None)
        obj.start_layer, obj.end_layer = start, end
        obj.layers = [layer] * 12
        obj.aux_hidden_state_layers = (2, 4, 6, 8, 10)
        obj._aux_slot_base_cached = upstream
        obj._aux_upstream_total_cached = upstream if pp.is_last_rank else 0
        obj.output_attn_res_proj = obj.output_attn_res_norm = None
        return obj

    positions = torch.tensor([0, 1])
    initial = torch.zeros(2, 3)
    first = decoder(0, 6, 0)
    intermediate = first.forward(None, positions, None, inputs_embeds=initial)
    assert set(intermediate.tensors) == {
        "hidden_states", "residual", "aux_hidden_states_0",
        "aux_hidden_states_1", "aux_hidden_states_2",
    }
    pp.is_first_rank, pp.is_last_rank = False, True
    last = decoder(6, 12, 3)
    hidden, aux = last.forward(None, positions, intermediate)
    torch.testing.assert_close(hidden, initial + 12)
    assert len(aux) == 5  # Layer 6 is captured once, at the sending stage.
    for value, expected in zip(aux, (2, 4, 6, 8, 10)):
        torch.testing.assert_close(value, initial + expected)
