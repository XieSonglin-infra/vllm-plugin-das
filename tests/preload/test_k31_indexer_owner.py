# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace

import torch
from torch import nn

from vllm_hcu.models.kimi_k31 import KimiK31ForConditionalGeneration


def test_nested_mla_keeps_outer_checkpoint_owner():
    module_names=('kv_a_layernorm','kv_b_proj','rotary_emb','o_proj',
                  'fused_qkv_a_proj','kv_a_proj_with_mqa','q_a_layernorm',
                  'q_b_proj','q_proj','g_proj')
    class Wrapper(nn.Module):
        def __init__(self,hidden_size,num_heads,scale,nope,rope,vhead,qrank,kvrank,
                     modules,cache,quant,prefix):
            super().__init__()
            self.hidden_size=hidden_size;self.num_heads=num_heads
            self.qk_nope_head_dim=nope;self.qk_rope_head_dim=rope
            self.v_head_dim=vhead;self.q_lora_rank=qrank;self.kv_lora_rank=kvrank
            self.prefix=prefix
            for name in module_names:setattr(self,name,getattr(modules,name))
            self.indexer=modules.indexer
            self.mla_attn=nn.Module()
            self.mla_attn.layer_name=prefix+'.attn';self.mla_attn.scale=scale
            self.mla_attn.indexer=modules.indexer

    modules=SimpleNamespace(**{name:None for name in module_names},indexer=None)
    old=Wrapper(8,2,1.0,4,0,4,4,4,modules,None,None,'layer3')
    indexer=nn.Module();indexer.k_norm=nn.LayerNorm(4)
    attention=nn.Module();attention.mla_attn=old;attention.indexer=indexer
    cfg=SimpleNamespace(compilation_config=SimpleNamespace(
        static_forward_context={old.mla_attn.layer_name:old.mla_attn}),cache_config=None)
    model=SimpleNamespace(vllm_config=cfg,quant_config=None,topk_indices_buffer=None)
    KimiK31ForConditionalGeneration._replace_with_sparse_dsa(model,attention,indexer)
    names=dict(attention.named_parameters())
    assert set(names)=={'indexer.k_norm.weight','indexer.k_norm.bias'}
    assert names['indexer.k_norm.bias'] is indexer.k_norm.bias
    assert attention.mla_attn.indexer is indexer
    assert attention.mla_attn.mla_attn.indexer is indexer
    assert 'indexer' not in attention.mla_attn._modules
    assert 'indexer' not in attention.mla_attn.mla_attn._modules
    assert attention.indexer is indexer
