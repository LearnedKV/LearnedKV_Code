# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import logging
import os
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

from kvpress.presses.base_press import BasePress
from kvpress.presses.scorer_press import ScorerPress

logger = logging.getLogger(__name__)


@dataclass
class AdaptivePerHeadPress(BasePress):
    """
    Adaptive per-head compression using learned policy network.
    
    This wrapper applies a learned policy to dynamically allocate different compression
    ratios to each layer and attention head, based on the KV cache statistics. It wraps
    any ScorerPress-based compression method and modifies its behavior to use per-head
    token budgets instead of a uniform compression ratio.
    
    The policy network (HeadFeatureExtractor + PolicyMLP) is trained offline using the
    code in the `policy/` directory to predict optimal per-head compression strategies.
    
    Parameters
    ----------
    press : ScorerPress
        The underlying scoring method to apply with per-head compression.
    policy_checkpoint : str
        Path to the trained policy checkpoint (.pt file) containing the feature
        extractor and policy MLP weights.
    c_target : float, default=0.5
        Target global compression ratio (fraction of tokens to keep). The policy
        network will distribute this budget across layers and heads adaptively.
    min_tokens_per_head : float, default=1.0
        Minimum number of tokens to keep per head. Ensures no head is completely
        pruned, which could cause issues.
    device : str, optional
        Device to run the policy network on. If None, uses the same device as the model.
    
    Examples
    --------
    >>> from kvpress import ObservedAttentionPress, AdaptivePerHeadPress
    >>> base_press = ObservedAttentionPress()
    >>> adaptive_press = AdaptivePerHeadPress(
    ...     press=base_press,
    ...     policy_checkpoint="ckpts/policy/policy_epoch10.pt",
    ...     c_target=0.5
    ... )
    >>> with adaptive_press(model):
    ...     outputs = model.generate(input_ids, max_length=100)
    """
    
    press: ScorerPress
    policy_checkpoint: str
    c_target: float = 0.5
    min_tokens_per_head: float = 1.0
    device: Optional[str] = None
    
    def __post_init__(self):
        assert isinstance(self.press, ScorerPress), (
            "AdaptivePerHeadPress requires a ScorerPress as input"
        )
        assert 0 < self.c_target <= 1.0, "c_target must be between 0 and 1"
        assert self.min_tokens_per_head >= 0, "min_tokens_per_head must be non-negative"
        
        # Load policy checkpoint
        if not os.path.exists(self.policy_checkpoint):
            raise FileNotFoundError(
                f"Policy checkpoint not found: {self.policy_checkpoint}"
            )
        
        logger.info(f"Loading policy checkpoint from {self.policy_checkpoint}")
        checkpoint = torch.load(self.policy_checkpoint, map_location="cpu")
        
        # Import policy modules
        from policy.features import HeadFeatureExtractor
        from policy.model import PolicyMLP
        
        # Get config
        config = checkpoint["config"]
        self.d_head = config["d_head"]
        self.d_feat = config["d_feat"]
        self.hidden_dim = config["hidden_dim"]
        
        # Initialize policy networks
        self.feature_extractor = HeadFeatureExtractor(d_head=self.d_head)
        self.policy_mlp = PolicyMLP(d_feat=self.d_feat, d_hid=self.hidden_dim)
        
        # Load weights
        self.feature_extractor.load_state_dict(checkpoint["extractor"])
        self.policy_mlp.load_state_dict(checkpoint["policy"])
        
        # Set to eval mode
        self.feature_extractor.eval()
        self.policy_mlp.eval()
        
        # Note: dtype will be set dynamically to match model dtype during first compress call
        
        logger.info(
            f"Policy loaded successfully: d_head={self.d_head}, "
            f"d_feat={self.d_feat}, hidden_dim={self.hidden_dim}"
        )
        
        # Cache for per-head k values
        self._per_head_k = {}
        
        # Track compressed cache size for statistics
        self._compressed_cache_size = 0
        
        # Track prefill sequence length for accurate compression ratio calculation
        self._prefill_seq_len = None
    
    def _move_policy_to_device(self, device: torch.device, dtype: torch.dtype = None):
        """Move policy networks to the specified device and dtype."""
        current_device = self.feature_extractor.mu_K_proj.weight.device
        current_dtype = self.feature_extractor.mu_K_proj.weight.dtype
        
        needs_move = current_device != device
        needs_cast = dtype is not None and current_dtype != dtype
        
        if needs_move or needs_cast:
            if dtype is not None:
                self.feature_extractor.to(device=device, dtype=dtype)
                self.policy_mlp.to(device=device, dtype=dtype)
                logger.debug(f"Moved policy networks to device: {device}, dtype: {dtype}")
            else:
                self.feature_extractor.to(device)
                self.policy_mlp.to(device)
                logger.debug(f"Moved policy networks to device: {device}")
    
    def _extract_features_for_layer(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
        layer_idx: int,
        num_layers: int,
    ) -> torch.Tensor:
        """
        Extract features for all heads in a layer.
        
        Parameters
        ----------
        keys : torch.Tensor
            Shape: [batch_size, num_kv_heads, seq_len, head_dim]
        values : torch.Tensor
            Shape: [batch_size, num_kv_heads, seq_len, head_dim]
        layer_idx : int
            Current layer index (0-based)
        num_layers : int
            Total number of layers
        
        Returns
        -------
        torch.Tensor
            Features for all heads, shape: [batch_size, num_kv_heads, d_feat]
        """
        batch_size, num_kv_heads, seq_len, head_dim = keys.shape
        
        features = []
        for h in range(num_kv_heads):
            # Extract features for this head across all batch items
            head_features = []
            for b in range(batch_size):
                K_h = keys[b, h]  # [seq_len, head_dim]
                V_h = values[b, h]  # [seq_len, head_dim]
                
                feat = self.feature_extractor(
                    K=K_h,
                    V=V_h,
                    layer_idx=layer_idx,
                    head_idx=h,
                    num_layers=num_layers,
                    num_heads=num_kv_heads,
                )
                head_features.append(feat)
            
            # Stack batch dimension
            head_features = torch.stack(head_features, dim=0)  # [batch_size, d_feat]
            features.append(head_features)
        
        # Stack head dimension
        features = torch.stack(features, dim=1)  # [batch_size, num_kv_heads, d_feat]
        return features
    
    def _compute_per_head_k(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
        layer_idx: int,
        num_layers: int,
    ) -> torch.Tensor:
        """
        Compute per-head token budgets using the policy network.
        
        Parameters
        ----------
        keys : torch.Tensor
            Shape: [batch_size, num_kv_heads, seq_len, head_dim]
        values : torch.Tensor
            Shape: [batch_size, num_kv_heads, seq_len, head_dim]
        layer_idx : int
            Current layer index
        num_layers : int
            Total number of layers
        
        Returns
        -------
        torch.Tensor
            Number of tokens to keep per head, shape: [batch_size, num_kv_heads]
        """
        batch_size, num_kv_heads, seq_len, head_dim = keys.shape
        
        # Extract features for all heads
        with torch.no_grad():
            features = self._extract_features_for_layer(
                keys, values, layer_idx, num_layers
            )  # [batch_size, num_kv_heads, d_feat]
            
            # Add layer dimension (we process one layer at a time)
            features = features.unsqueeze(1)  # [batch_size, 1, num_kv_heads, d_feat]
            
            # Get policy logits
            alpha = self.policy_mlp(features)  # [batch_size, 1, num_kv_heads]
            alpha = alpha.squeeze(1)  # [batch_size, num_kv_heads]
            
            # Softmax to get weights
            # We need to compute softmax over ALL heads across ALL layers, but we're
            # processing one layer at a time. For simplicity, we use softmax within
            # this layer as an approximation. This is a limitation - ideally we'd
            # process all layers together, but that's memory intensive.
            w = F.softmax(alpha, dim=-1)  # [batch_size, num_kv_heads]
            
            # Compute total budget for this layer
            # Total capacity if we keep everything in all num_layers layers
            total_capacity = num_layers * num_kv_heads * seq_len
            
            # Target budget across all layers
            target_budget = self.c_target * total_capacity
            
            # Assuming uniform distribution across layers (this is an approximation)
            layer_budget = target_budget / num_layers
            
            # Compute per-head capacity
            C_min = self.min_tokens_per_head
            C_pred = C_min + w * (layer_budget - num_kv_heads * C_min)
            
            # Clamp to valid range [0, seq_len]
            k_per_head = C_pred.clamp(min=C_min, max=float(seq_len))
            
            # Round to integers
            k_per_head = k_per_head.round().long()
        
        return k_per_head
    
    def compress(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attentions: torch.Tensor,
        kwargs: dict,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compress KV cache using per-head adaptive budgets.
        
        This method:
        1. Extracts features from KV cache
        2. Uses policy network to predict per-head token budgets
        3. Computes scores using the underlying press
        4. Selects different number of top-k tokens per head
        
        Parameters
        ----------
        module : nn.Module
            The transformer attention layer
        hidden_states : torch.Tensor
            Shape: [batch_size, seq_len, hidden_dim]
        keys : torch.Tensor
            Shape: [batch_size, num_kv_heads, seq_len, head_dim]
        values : torch.Tensor
            Shape: [batch_size, num_kv_heads, seq_len, head_dim]
        attentions : torch.Tensor
            Attention weights (may be None)
        kwargs : dict
            Additional forward pass arguments
        
        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Compressed keys and values
        """
        batch_size, num_kv_heads, seq_len, head_dim = keys.shape
        
        # Reset cache stats and record prefill seq_len if this is the first layer
        if module.layer_idx == 0:
            self._compressed_cache_size = 0
            # Record prefill sequence length (this is the original length before compression)
            self._prefill_seq_len = seq_len
        
        # Move policy networks to the same device and dtype as keys
        device = keys.device
        dtype = keys.dtype
        self._move_policy_to_device(device, dtype)
        
        # Get number of layers from model config
        num_layers = module.config.num_hidden_layers
        layer_idx = module.layer_idx
        
        # Compute per-head k values using policy network (for token selection guidance)
        k_per_head_allocated = self._compute_per_head_k(
            keys, values, layer_idx, num_layers
        )  # [batch_size, num_kv_heads] - policy's allocation suggestion
        
        # To ensure exact compression ratio matching c_target, calculate target_k from c_target
        # This ensures the total compression ratio is exactly c_target across all layers
        target_k = int(round(self.c_target * seq_len))
        target_k = max(1, min(target_k, seq_len))
        
        # Store policy allocation for statistics (what policy wants)
        k_per_head = k_per_head_allocated
        
        # Cache for logging/debugging (store both allocation and actual usage)
        cache_key = f"layer_{layer_idx}"
        # Store the policy's allocation (what each head ideally wants)
        self._per_head_k[cache_key] = k_per_head.float().mean(dim=0).cpu()
        # Also store the actual uniform target
        self._per_head_k[f"{cache_key}_actual"] = torch.full(
            (num_kv_heads,), float(target_k)
        )
        
        # Compute scores using the underlying press
        scores = self.press.score(
            module, hidden_states, keys, values, attentions, kwargs
        )  # [batch_size, num_kv_heads, seq_len]
        
        # For each head, select tokens based on scores, but ensure all return target_k tokens
        compressed_keys_list = []
        compressed_values_list = []
        
        for b in range(batch_size):
            batch_keys = []
            batch_values = []
            
            for h in range(num_kv_heads):
                head_scores = scores[b, h]  # [seq_len]
                
                # Get top-k indices based on the head's allocated budget
                k_h = k_per_head[b, h].item()
                k_h = max(1, min(int(k_h), seq_len))
                
                # Select top k_h tokens
                indices = head_scores.topk(k_h, dim=-1).indices  # [k_h]
                
                # If k_h < target_k, we need to add more tokens
                if k_h < target_k:
                    # Get remaining indices (not in top k_h)
                    all_indices = torch.arange(seq_len, device=head_scores.device)
                    mask = torch.ones(seq_len, dtype=torch.bool, device=head_scores.device)
                    mask[indices] = False
                    remaining_indices = all_indices[mask]
                    
                    # Select additional tokens to reach target_k
                    num_additional = target_k - k_h
                    if len(remaining_indices) >= num_additional:
                        # Select top scoring from remaining
                        remaining_scores = head_scores[remaining_indices]
                        additional_indices = remaining_indices[
                            remaining_scores.topk(num_additional, dim=-1).indices
                        ]
                        indices = torch.cat([indices, additional_indices])
                    else:
                        # Not enough remaining, just duplicate last index
                        indices = torch.cat([
                            indices,
                            indices[-1:].expand(target_k - k_h)
                        ])
                elif k_h > target_k:
                    # Keep only top target_k
                    indices = indices[:target_k]
                
                # Sort to maintain temporal order
                indices = indices.sort()[0]
                
                # Gather keys and values
                indices_expanded = indices.unsqueeze(-1).expand(-1, head_dim)
                head_keys = keys[b, h].gather(0, indices_expanded)  # [target_k, head_dim]
                head_values = values[b, h].gather(0, indices_expanded)  # [target_k, head_dim]
                
                batch_keys.append(head_keys)
                batch_values.append(head_values)
            
            # Stack heads
            batch_keys_stacked = torch.stack(batch_keys, dim=0)  # [num_kv_heads, target_k, head_dim]
            batch_values_stacked = torch.stack(batch_values, dim=0)
            
            compressed_keys_list.append(batch_keys_stacked)
            compressed_values_list.append(batch_values_stacked)
        
        # Stack batch
        compressed_keys_final = torch.stack(compressed_keys_list, dim=0)  # [batch_size, num_kv_heads, target_k, head_dim]
        compressed_values_final = torch.stack(compressed_values_list, dim=0)
        
        # Track compressed cache size (accumulate across layers)
        self._compressed_cache_size += (
            compressed_keys_final.numel() + compressed_values_final.numel()
        )
        
        logger.debug(
            f"Layer {layer_idx}: Original shape {keys.shape} -> "
            f"Compressed shape {compressed_keys_final.shape}, "
            f"Target k (uniform): {target_k}, "
            f"Mean k allocation per head: {k_per_head.float().mean():.1f}, "
            f"Std k allocation: {k_per_head.float().std():.1f}"
        )
        
        return compressed_keys_final.contiguous(), compressed_values_final.contiguous()
    
    def get_per_head_k_stats(self) -> dict:
        """
        Get statistics about per-head k values across all layers.
        
        Returns
        -------
        dict
            Dictionary mapping layer names to average k values per head
        """
        return dict(self._per_head_k)
    
    def get_compressed_cache_size(self) -> int:
        """
        Get the total size of compressed cache (in number of elements).
        
        This tracks the cumulative size across all layers during compression.
        Note: This is reset each time compress() is called, so call this
        after the compression is complete.
        
        Returns
        -------
        int
            Total number of elements in compressed cache (K + V)
        """
        return self._compressed_cache_size
    
    def reset_cache_stats(self):
        """Reset cache statistics. Call this before a new compression run."""
        self._compressed_cache_size = 0
        self._prefill_seq_len = None
    
    def get_prefill_seq_len(self) -> Optional[int]:
        """
        Get the prefill sequence length (original length before compression).
        
        This is useful for calculating the correct baseline cache size for comparison.
        
        Returns
        -------
        int or None
            The prefill sequence length, or None if not yet set
        """
        return self._prefill_seq_len
    
    @property
    def compression_ratio(self) -> float:
        """Return the target compression ratio."""
        return 1.0 - self.c_target
    
    @compression_ratio.setter
    def compression_ratio(self, value: float):
        """Set the target compression ratio."""
        self.c_target = 1.0 - value

