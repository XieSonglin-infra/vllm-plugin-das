# SPDX-License-Identifier: Apache-2.0
"""K3 SlimQuant mixed projections, preserving checkpoint branch precision."""
from __future__ import annotations

import torch
from torch import nn


def is_mixed_slimquant(config) -> bool:
    return (config is not None
            and config.get_name() == "slimquant_w4a8"
            and getattr(config, "w8a8_include", None) is not None)


class KimiK3MixedKDAProjection(nn.Module):
    """INT8 Q/K/V plus the native replicated-Fa floating gate projection.

    The caller's packed output order and padding are unchanged. Keeping the
    branches separate avoids casting integer checkpoint values into BF16 or
    quantizing the full-rank gate with Q/K/V's scales.
    """

    _hcu_mixed_kda = True

    def __init__(self, input_size, output_sizes, replicated_shard_id, tp_size,
                 *, column_linear, merged_linear, quant_config, prefix, **kwargs):
        super().__init__()
        if replicated_shard_id != 4 or len(output_sizes) not in (6, 7):
            raise ValueError("Mixed K3 KDA requires full-rank Q/K/V/G/Fa/B layout")
        parent = prefix.removesuffix(".in_proj_qkvgfab")
        for name, size in zip(("q_proj", "k_proj", "v_proj"), output_sizes[:3]):
            layer = column_linear(input_size, size, quant_config=quant_config,
                                  prefix=f"{parent}.{name}", **kwargs)
            if layer.weight.dtype != torch.int8 or not hasattr(layer, "weight_scale"):
                raise ValueError(f"Mixed K3 KDA {parent}.{name} must select W8A8")
            self.add_module(name, layer)
        self.fp_gate = merged_linear(
            input_size, output_sizes[3:], replicated_shard_id=1, tp_size=tp_size,
            quant_config=None, prefix=f"{prefix}.fp_gate", **kwargs)

    @property
    def weight(self):
        # Upstream zeros only the final padding rows through this property.
        return self.fp_gate.weight

    def forward(self, x):
        from vllm_hcu.platforms import envs as henvs

        projections = (self.q_proj, self.k_proj, self.v_proj)
        share = (
            henvs.VLLM_HCU_KIMI_SHARE_QKV_QUANT
            and henvs.VLLM_HCU_USE_CUSTOM_OPS
            and (henvs.VLLM_HCU_USE_FUSED_SILU_MUL_QUANT
                 or henvs.VLLM_HCU_USE_FUSED_RMS_QUANT)
            and x.dtype in (torch.bfloat16, torch.float16)
            and x.ndim >= 2 and x.numel() > 0
        )
        if share:
            for layer in projections:
                scheme = getattr(layer, "scheme", None)
                supports = getattr(scheme, "supports_quanted_inputs", None)
                if not (
                    callable(getattr(layer, "_forward_with_hcu_quanted", None))
                    and callable(supports) and supports()
                    and getattr(scheme, "input_symmetric", False)
                    and not getattr(scheme, "is_static_input_scheme", True)
                    and getattr(layer, "input_scale", None) is None
                    and getattr(layer, "input_zero_point", None) is None
                    and getattr(layer, "azp_adj", None) is None
                ):
                    share = False
                    break
        if share:
            # Use the same quantizer as apply_int8_linear. Only the activation
            # and its per-token scale are shared; GEMMs and branch precision
            # keep their existing implementation and checkpoint contracts.
            from lightop.quant import per_token_quant_int8

            quantized = per_token_quant_int8(x)
            merged = None
            if henvs.VLLM_HCU_KIMI_MERGE_QKV_GEMM:
                from .kimi_qkv_merge import merged_qkv

                merged = merged_qkv(self, projections, x, quantized)
            if merged is not None:
                if (henvs.VLLM_HCU_KIMI_MERGE_QKV_VALIDATE
                        and not getattr(self, "_hcu_qkv_validated", False)):
                    reference = torch.cat([
                        layer._forward_with_hcu_quanted(x, quantized)[0]
                        for layer in projections], dim=-1)
                    torch.testing.assert_close(merged, reference, rtol=0, atol=0)
                    self._hcu_qkv_validated = True
                    print("KDA merged QKV exact validation passed", tuple(x.shape),
                          "cache_bytes", self._hcu_qkv_cache_bytes, flush=True)
                values = [merged]
            else:
                values = [layer._forward_with_hcu_quanted(x, quantized)[0]
                          for layer in projections]
        else:
            values = [layer(x)[0] for layer in projections]
        values.append(self.fp_gate(x)[0])
        return torch.cat(values, dim=-1), None

    def load_branch(self, branch, suffix, tensor):
        # Loading can mutate an inference Parameter without a version counter.
        self._hcu_merged_qkv_cache = None
        self._hcu_qkv_validated = False
        if branch in ("q_proj", "k_proj", "v_proj"):
            if suffix not in ("weight", "weight_scale"):
                raise ValueError(f"Unsupported K3 W8A8 parameter {branch}.{suffix}")
            if suffix == "weight" and tensor.dtype != torch.int8:
                raise TypeError(f"{branch}.weight must be INT8")
            if suffix == "weight_scale" and tensor.dtype != torch.float32:
                raise TypeError(f"{branch}.weight_scale must be FP32")
            param = getattr(getattr(self, branch), suffix)
            param.weight_loader(param, tensor)
            return f"{branch}.{suffix}"
        if suffix != "weight" or not tensor.is_floating_point():
            raise ValueError(f"K3 floating gate requires {branch}.weight")
        shard = {"g_proj": 0, "f_a_proj": 1, "b_proj": 2}[branch]
        param = self.fp_gate.weight
        param.weight_loader(param, tensor, shard)
        return "fp_gate.weight"
