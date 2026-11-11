# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace

import pytest

from vllm_hcu.runtime_compat.weight_cache import install_daemon_workspace


@pytest.mark.parametrize('initialized', [False,True])
def test_workspace_before_model(monkeypatch,initialized):
    from vllm.v1.worker import workspace

    state={'ready':initialized};calls=[]
    monkeypatch.setattr(workspace,'is_workspace_manager_initialized',lambda:state['ready'])
    def init(device,**kwargs):
        calls.append(('init',device,kwargs));state['ready']=True
    monkeypatch.setattr(workspace,'init_workspace_manager',init)
    class Daemon:
        tp_rank=3
        vllm_config=SimpleNamespace(parallel_config=SimpleNamespace(enable_dbo=False))
        def load_model(self):
            assert state['ready'], 'WorkspaceManager not initialized'
            calls.append(('load',))
            return 'loaded'
    module=SimpleNamespace(WeightCacheDaemon=Daemon,torch=SimpleNamespace(
        accelerator=SimpleNamespace(set_device_index=lambda rank:calls.append(('device',rank))),
        device=lambda kind,index:(kind,index)))
    if not initialized:
        with pytest.raises(AssertionError,match='not initialized'):Daemon().load_model()
    install_daemon_workspace(module)
    wrapped=Daemon.load_model
    install_daemon_workspace(module)
    assert Daemon.load_model is wrapped
    assert Daemon().load_model()=='loaded'
    assert calls[0]==('device',3)
    inits=[e for e in calls if e[0]=='init']
    assert len(inits)==(0 if initialized else 1)
    if inits: assert inits[0][1]==('cuda',3)
    assert calls[-1]==('load',)
