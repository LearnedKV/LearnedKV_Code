# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import os
import tempfile

import pytest
import torch
from torch import nn

from kvpress.presses.adaptive_per_head_press import AdaptivePerHeadPress
from kvpress.presses.observed_attention_press import ObservedAttentionPress


@pytest.fixture
def mock_policy_checkpoint():
    """Create a mock policy checkpoint for testing."""
    from policy.features import HeadFeatureExtractor
    from policy.model import PolicyMLP
    
    d_head = 64
    extractor = HeadFeatureExtractor(d_head=d_head)
    d_feat = extractor.feature_dim
    policy = PolicyMLP(d_feat=d_feat, d_hid=64)
    
    checkpoint = {
        "extractor": extractor.state_dict(),
        "policy": policy.state_dict(),
        "config": {
            "L": 4,
            "H": 8,
            "T_kv": 128,
            "d_head": d_head,
            "d_feat": d_feat,
            "hidden_dim": 64,
        },
        "epoch": 1,
        "global_step": 100,
    }
    
    # Save to temporary file
    with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.pt') as f:
        torch.save(checkpoint, f)
        return f.name


def test_adaptive_per_head_press_init(mock_policy_checkpoint):
    """Test initialization of AdaptivePerHeadPress."""
    base_press = ObservedAttentionPress()
    
    adaptive_press = AdaptivePerHeadPress(
        press=base_press,
        policy_checkpoint=mock_policy_checkpoint,
        c_target=0.5,
        min_tokens_per_head=1.0,
    )
    
    assert adaptive_press.c_target == 0.5
    assert adaptive_press.min_tokens_per_head == 1.0
    assert adaptive_press.d_head == 64
    assert adaptive_press.feature_extractor is not None
    assert adaptive_press.policy_mlp is not None
    
    # Cleanup
    os.unlink(mock_policy_checkpoint)


def test_adaptive_per_head_press_invalid_checkpoint():
    """Test that invalid checkpoint path raises error."""
    base_press = ObservedAttentionPress()
    
    with pytest.raises(FileNotFoundError):
        AdaptivePerHeadPress(
            press=base_press,
            policy_checkpoint="nonexistent_checkpoint.pt",
            c_target=0.5,
        )


def test_adaptive_per_head_press_invalid_c_target(mock_policy_checkpoint):
    """Test that invalid c_target raises error."""
    base_press = ObservedAttentionPress()
    
    with pytest.raises(AssertionError):
        AdaptivePerHeadPress(
            press=base_press,
            policy_checkpoint=mock_policy_checkpoint,
            c_target=1.5,  # Invalid: > 1.0
        )
    
    with pytest.raises(AssertionError):
        AdaptivePerHeadPress(
            press=base_press,
            policy_checkpoint=mock_policy_checkpoint,
            c_target=0.0,  # Invalid: <= 0
        )
    
    # Cleanup
    os.unlink(mock_policy_checkpoint)


def test_extract_features(mock_policy_checkpoint):
    """Test feature extraction for a layer."""
    base_press = ObservedAttentionPress()
    adaptive_press = AdaptivePerHeadPress(
        press=base_press,
        policy_checkpoint=mock_policy_checkpoint,
        c_target=0.5,
    )
    
    # Create mock KV tensors
    batch_size = 2
    num_heads = 8
    seq_len = 128
    head_dim = 64
    
    keys = torch.randn(batch_size, num_heads, seq_len, head_dim)
    values = torch.randn(batch_size, num_heads, seq_len, head_dim)
    
    # Extract features
    features = adaptive_press._extract_features_for_layer(
        keys, values, layer_idx=0, num_layers=4
    )
    
    # Check shape
    assert features.shape == (batch_size, num_heads, adaptive_press.d_feat)
    
    # Cleanup
    os.unlink(mock_policy_checkpoint)


def test_compute_per_head_k(mock_policy_checkpoint):
    """Test computation of per-head k values."""
    base_press = ObservedAttentionPress()
    adaptive_press = AdaptivePerHeadPress(
        press=base_press,
        policy_checkpoint=mock_policy_checkpoint,
        c_target=0.5,
        min_tokens_per_head=1.0,
    )
    
    # Create mock KV tensors
    batch_size = 2
    num_heads = 8
    seq_len = 128
    head_dim = 64
    
    keys = torch.randn(batch_size, num_heads, seq_len, head_dim)
    values = torch.randn(batch_size, num_heads, seq_len, head_dim)
    
    # Compute per-head k
    k_per_head = adaptive_press._compute_per_head_k(
        keys, values, layer_idx=0, num_layers=4
    )
    
    # Check shape
    assert k_per_head.shape == (batch_size, num_heads)
    
    # Check values are in valid range
    assert (k_per_head >= adaptive_press.min_tokens_per_head).all()
    assert (k_per_head <= seq_len).all()
    
    # Cleanup
    os.unlink(mock_policy_checkpoint)


def test_compression_ratio_property(mock_policy_checkpoint):
    """Test compression_ratio property."""
    base_press = ObservedAttentionPress()
    adaptive_press = AdaptivePerHeadPress(
        press=base_press,
        policy_checkpoint=mock_policy_checkpoint,
        c_target=0.6,
    )
    
    assert adaptive_press.compression_ratio == pytest.approx(0.4)
    
    # Test setter
    adaptive_press.compression_ratio = 0.3
    assert adaptive_press.c_target == pytest.approx(0.7)
    
    # Cleanup
    os.unlink(mock_policy_checkpoint)


def test_get_per_head_k_stats(mock_policy_checkpoint):
    """Test retrieval of per-head k statistics."""
    base_press = ObservedAttentionPress()
    adaptive_press = AdaptivePerHeadPress(
        press=base_press,
        policy_checkpoint=mock_policy_checkpoint,
        c_target=0.5,
    )
    
    # Initially empty
    stats = adaptive_press.get_per_head_k_stats()
    assert len(stats) == 0
    
    # Add some mock stats
    adaptive_press._per_head_k["layer_0"] = torch.tensor([10, 20, 30, 40])
    adaptive_press._per_head_k["layer_1"] = torch.tensor([15, 25, 35, 45])
    
    stats = adaptive_press.get_per_head_k_stats()
    assert len(stats) == 2
    assert "layer_0" in stats
    assert "layer_1" in stats
    
    # Cleanup
    os.unlink(mock_policy_checkpoint)


def test_compress_basic(mock_policy_checkpoint):
    """Test basic compression functionality."""
    from unittest.mock import MagicMock
    
    base_press = ObservedAttentionPress()
    adaptive_press = AdaptivePerHeadPress(
        press=base_press,
        policy_checkpoint=mock_policy_checkpoint,
        c_target=0.5,
        min_tokens_per_head=1.0,
    )
    
    # Create mock inputs
    batch_size = 1
    num_heads = 8
    seq_len = 128
    head_dim = 64
    hidden_dim = 512
    
    keys = torch.randn(batch_size, num_heads, seq_len, head_dim)
    values = torch.randn(batch_size, num_heads, seq_len, head_dim)
    hidden_states = torch.randn(batch_size, seq_len, hidden_dim)
    
    # Mock module
    module = MagicMock(spec=nn.Module)
    module.config.num_hidden_layers = 4
    module.config.num_attention_heads = 32
    module.config.num_key_value_heads = num_heads
    module.layer_idx = 0
    module.head_dim = head_dim
    
    # Mock score method to return uniform scores
    def mock_score(module, hidden_states, keys, values, attentions, kwargs):
        return torch.ones(batch_size, num_heads, seq_len)
    
    base_press.score = mock_score
    
    # Compress
    compressed_keys, compressed_values = adaptive_press.compress(
        module=module,
        hidden_states=hidden_states,
        keys=keys,
        values=values,
        attentions=None,
        kwargs={},
    )
    
    # Check output shapes
    assert compressed_keys.shape[0] == batch_size
    assert compressed_keys.shape[1] == num_heads
    assert compressed_keys.shape[3] == head_dim
    assert compressed_keys.shape[2] < seq_len  # Should be compressed
    
    assert compressed_values.shape == compressed_keys.shape
    
    # Check stats were recorded
    stats = adaptive_press.get_per_head_k_stats()
    assert "layer_0" in stats
    
    # Cleanup
    os.unlink(mock_policy_checkpoint)


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])

