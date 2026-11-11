# SPDX-License-Identifier: Apache-2.0
import os
import socket

import pytest

from vllm_hcu.runtime_compat.weight_cache import (
    _bounded_socket_path, validate_preload_socket_dir,
)


def path_for(identifier, directory):
    return os.path.join(directory, 'vllm_weight_cache_gpu'+str(identifier)+'.sock')


def test_short():
    assert _bounded_socket_path(path_for, 'uuid-abc', '/tmp/wc') == path_for('uuid-abc','/tmp/wc')


def test_long(tmp_path):
    directory = tmp_path / ('d' * max(1,70-len(os.fsencode(tmp_path))-1))
    directory.mkdir()
    identifier = 'uuid-'+'12345678-'*5+'pp0'
    path = _bounded_socket_path(path_for, identifier, str(directory))
    assert path == _bounded_socket_path(path_for, identifier, str(directory))
    assert path != _bounded_socket_path(path_for, identifier+'-pp1', str(directory))
    assert len(os.fsencode(path+'.draft')) <= 107
    for suffix in ['', '.draft']:
        sock = socket.socket(socket.AF_UNIX)
        try:
            sock.bind(path+suffix)
        finally:
            sock.close()
            os.unlink(path+suffix)


def test_too_long():
    with pytest.raises(ValueError,match='too long'):
        _bounded_socket_path(path_for,'uuid-'+'a'*40,'/'+('x'*90))


def test_preflight(tmp_path):
    directory = tmp_path / 'cache'
    assert validate_preload_socket_dir(str(directory)) == str(directory)
    assert not list(directory.iterdir())
    assert directory.stat().st_mode & 0o777 == 0o700
    alias = tmp_path / 'alias'
    alias.symlink_to(directory,target_is_directory=True)
    with pytest.raises(Exception,match='symlinked socket directory'):
        validate_preload_socket_dir(str(alias))
