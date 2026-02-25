"""Tests for rabbitllm.utils.splitting."""

import shutil
from unittest.mock import MagicMock

import pytest
import torch

from rabbitllm.utils.memory import NotEnoughSpaceException
from rabbitllm.utils.splitting import (
    check_space,
    load_layer,
    remove_real_and_linked_file,
)


def test_load_layer_returns_state_dict_from_persister():
    persister = MagicMock()
    persister.load_model.return_value = {"weight": torch.randn(2, 2)}
    result = load_layer("/fake/path", "model.layers.0", persister=persister)
    assert isinstance(result, dict)
    assert "weight" in result
    persister.load_model.assert_called_once_with("model.layers.0", "/fake/path")


def test_load_layer_with_profiling_returns_tuple():
    persister = MagicMock()
    persister.load_model.return_value = {"weight": torch.randn(2, 2)}
    result = load_layer("/fake/path", "model.layers.0", profiling=True, persister=persister)
    assert isinstance(result, tuple)
    state_dict, elapsed = result
    assert isinstance(state_dict, dict)
    assert isinstance(elapsed, (int, float))


def test_check_space_passes_when_enough_space(tmp_path):
    (tmp_path / "dummy.bin").write_bytes(b"x" * 100)
    check_space(tmp_path)


def test_check_space_raises_when_not_enough_space(tmp_path):
    (tmp_path / "model-00001-of-00001.safetensors").write_bytes(b"x" * 1024)
    original_disk_usage = shutil.disk_usage

    def mock_disk_usage(_p):
        return (1000, 1000, 0)

    shutil.disk_usage = mock_disk_usage
    try:
        with pytest.raises(NotEnoughSpaceException):
            check_space(tmp_path)
    finally:
        shutil.disk_usage = original_disk_usage


def test_remove_real_and_linked_file_removes_file(tmp_path):
    path = tmp_path / "to_delete.bin"
    path.write_bytes(b"x")
    assert path.exists()
    remove_real_and_linked_file(path)
    assert not path.exists()


def test_sequential_shard_processing_logic(tmp_path):
    import json
    from unittest.mock import patch
    from rabbitllm.utils.splitting import split_and_save_layers

    index_data = {
        "weight_map": {
            "model.embed_tokens.weight": "pytorch_model-00001-of-00002.bin",
            "model.layers.0.self_attn.q_proj.weight": "pytorch_model-00001-of-00002.bin",
            "model.layers.0.self_attn.k_proj.weight": "pytorch_model-00002-of-00002.bin",
            "model.norm.weight": "pytorch_model-00002-of-00002.bin",
            "lm_head.weight": "pytorch_model-00002-of-00002.bin"
        }
    }
    
    with open(tmp_path / "pytorch_model.bin.index.json", "w") as f:
        json.dump(index_data, f)
        
    shard1 = {
        "model.embed_tokens.weight": torch.randn(2, 2),
        "model.layers.0.self_attn.q_proj.weight": torch.randn(2, 2)
    }
    torch.save(shard1, tmp_path / "pytorch_model-00001-of-00002.bin")
    
    shard2 = {
        "model.layers.0.self_attn.k_proj.weight": torch.randn(2, 2),
        "model.norm.weight": torch.randn(2, 2),
        "lm_head.weight": torch.randn(2, 2)
    }
    torch.save(shard2, tmp_path / "pytorch_model-00002-of-00002.bin")
    
    layer_names = {
        "embed": "model.embed_tokens",
        "layer_prefix": "model.layers",
        "norm": "model.norm",
        "lm_head": "lm_head"
    }

    mock_persister = MagicMock()
    mock_persister_instance = MagicMock()
    mock_persister_instance.model_persist_exist.return_value = False
    mock_persister.get_model_persister.return_value = mock_persister_instance

    with patch("rabbitllm.utils.splitting.ModelPersister", mock_persister):
        split_and_save_layers(
            checkpoint_path=tmp_path,
            layer_shards_saving_path=tmp_path,
            layer_names=layer_names,
            delete_original=False,
            sequential_shard_processing=True,
        )
        
    calls = mock_persister_instance.persist_model.call_args_list
    saved_layers = [call.args[1] for call in calls]
    
    assert "model.embed_tokens." in saved_layers
    assert "model.layers.0." in saved_layers
    assert "model.norm." in saved_layers
    assert "lm_head." in saved_layers
    assert (tmp_path / "pytorch_model-00001-of-00002.bin").exists()


def test_sequential_shard_processing_delete_original(tmp_path):
    import json
    from unittest.mock import patch
    from rabbitllm.utils.splitting import split_and_save_layers

    index_data = {
        "weight_map": {
            "model.embed_tokens.weight": "pytorch_model-00001-of-00002.bin",
        }
    }
    
    with open(tmp_path / "pytorch_model.bin.index.json", "w") as f:
        json.dump(index_data, f)
        
    shard1 = {
        "model.embed_tokens.weight": torch.randn(2, 2),
    }
    torch.save(shard1, tmp_path / "pytorch_model-00001-of-00002.bin")
    
    mock_persister = MagicMock()
    mock_persister_instance = MagicMock()
    mock_persister_instance.model_persist_exist.return_value = False
    mock_persister.get_model_persister.return_value = mock_persister_instance

    with patch("rabbitllm.utils.splitting.ModelPersister", mock_persister):
        split_and_save_layers(
            checkpoint_path=tmp_path,
            layer_shards_saving_path=tmp_path,
            layer_names={"embed": "model.embed_tokens", "layer_prefix": "model.layers", "norm": "model.norm", "lm_head": "lm_head"},
            delete_original=True,
            sequential_shard_processing=True,
        )
        
    assert not (tmp_path / "pytorch_model-00001-of-00002.bin").exists()
