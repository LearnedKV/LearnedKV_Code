"""
Adaptive Per-Head KV Cache Compression Example

This script demonstrates how to use AdaptivePerHeadPress to apply
learned per-layer per-head compression strategies to any KV cache
compression algorithm in kvpress.

Usage:
    python examples/adaptive_per_head_example.py \
        --model-name meta-llama/Llama-3.2-1B-Instruct \
        --policy-checkpoint ckpts/policy/llama-3.2-1B/policy_epoch10.pt \
        --compression-method observed \
        --c-target 0.5
"""

import argparse
import sys
import time
from pathlib import Path

# Add local kvpress to path (prioritize local development version)
REPO_ROOT = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(REPO_ROOT))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

from kvpress import (
    AdaptivePerHeadPress,
    ExpectedAttentionPress,
    KnormPress,
    ObservedAttentionPress,
    SnapKVPress,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Adaptive per-head KV cache compression example"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="meta-llama/Llama-3.2-1B-Instruct",
        help="HuggingFace model name",
    )
    parser.add_argument(
        "--policy-checkpoint",
        type=str,
        required=True,
        help="Path to trained policy checkpoint",
    )
    parser.add_argument(
        "--compression-method",
        type=str,
        default="observed",
        choices=["observed", "snapkv", "expected", "knorm"],
        help="Base compression method to use",
    )
    parser.add_argument(
        "--c-target",
        type=float,
        default=0.5,
        help="Target compression ratio (fraction of tokens to keep)",
    )
    parser.add_argument(
        "--min-tokens-per-head",
        type=float,
        default=1.0,
        help="Minimum tokens to keep per head",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=50,
        help="Maximum number of new tokens to generate",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run on",
    )
    parser.add_argument(
        "--compare-baseline",
        action="store_true",
        help="Also run baseline without compression for comparison",
    )
    parser.add_argument(
        "--compare-fixed",
        action="store_true",
        help="Also run with fixed compression ratio for comparison",
    )
    return parser.parse_args()


def create_base_press(method: str):
    """Create base compression press based on method name."""
    if method == "observed":
        return ObservedAttentionPress()
    elif method == "snapkv":
        return SnapKVPress(window_size=64, kernel_size=5)
    elif method == "expected":
        return ExpectedAttentionPress()
    elif method == "knorm":
        return KnormPress()
    else:
        raise ValueError(f"Unknown compression method: {method}")


def get_cache_size(cache):
    """Calculate total KV cache size in number of elements."""
    if cache is None:
        return 0
    
    # Try different cache structures
    try:
        # Method 1: Direct key_cache/value_cache (older transformers)
        if hasattr(cache, "key_cache") and hasattr(cache, "value_cache"):
            if len(cache.key_cache) > 0 and len(cache.value_cache) > 0:
                return sum(
                    k.numel() + v.numel() 
                    for k, v in zip(cache.key_cache, cache.value_cache)
                )
        
        # Method 2: DynamicCache with layers (newer transformers)
        if hasattr(cache, "layers") and len(cache.layers) > 0:
            total = 0
            for layer in cache.layers:
                if hasattr(layer, "keys") and hasattr(layer, "values"):
                    if layer.keys.numel() > 0:
                        total += layer.keys.numel()
                    if layer.values.numel() > 0:
                        total += layer.values.numel()
            if total > 0:
                return total
        
        # Method 3: Check if cache has get_seq_length method
        if hasattr(cache, "get_seq_length"):
            seq_len = cache.get_seq_length()
            if seq_len > 0 and hasattr(cache, "layers") and len(cache.layers) > 0:
                # Estimate from first layer
                first_layer = cache.layers[0]
                if hasattr(first_layer, "keys") and first_layer.keys.numel() > 0:
                    head_dim = first_layer.keys.shape[-1]
                    num_heads = first_layer.keys.shape[1]
                    num_layers = len(cache.layers)
                    return num_layers * 2 * num_heads * seq_len * head_dim
        
        return 0
    except Exception as e:
        # Silently return 0 if we can't calculate
        return 0


def main():
    args = parse_args()
    
    print(f"Loading model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16 if args.device == "cuda" else torch.float32,
        device_map=args.device,
        attn_implementation="eager",  # Required for attention-based methods
    )
    
    # Set pad_token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"Model loaded on {args.device}")
    print(f"  Layers: {model.config.num_hidden_layers}")
    print(f"  Heads: {model.config.num_attention_heads}")
    print(f"  KV Heads: {model.config.num_key_value_heads}")
    
    # Prepare input
    long_context = """Artificial Intelligence has revolutionized many aspects of modern life. 
    From healthcare to transportation, AI systems are being deployed to solve complex problems.
    Machine learning enables computers to learn from data without explicit programming.
    Deep learning using neural networks has achieved remarkable success in image recognition, 
    natural language processing, and game playing. Large Language Models like GPT have 
    demonstrated impressive capabilities in understanding and generating human-like text.
    The KV cache is a crucial optimization for efficient inference in transformer models.
    It stores key and value vectors from previous tokens to avoid redundant computation.
    However, the KV cache grows linearly with sequence length, consuming significant memory.
    Various compression techniques have been proposed to reduce KV cache size while 
    maintaining model performance. These include dropping less important tokens,
    quantizing the cache, or using more efficient data structures."""
    
    question = "What is the main challenge with KV cache in transformers?"
    prompt = f"Context: {long_context}\n\nQuestion: {question}\nAnswer:"
    
    print(f"\nPrompt: {prompt[:100]}...")
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(args.device)
    print(f"Input length: {input_ids.shape[1]} tokens")
    
    results = {}
    
    # Baseline: No compression
    if args.compare_baseline:
        print("\n" + "=" * 60)
        print("BASELINE: No Compression")
        print("=" * 60)
        
        # Note: DynamicCache may not be populated when using past_key_values
        # without a compression method. This is expected behavior.
        cache = DynamicCache()
        start_time = time.time()
        
        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                max_new_tokens=args.max_new_tokens,
                past_key_values=cache,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        
        elapsed = time.time() - start_time
        output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        cache_size = get_cache_size(cache)
        
        # If cache is empty, estimate size from model config
        if cache_size == 0:
            # Estimate: num_layers * 2 (K+V) * batch_size * num_kv_heads * seq_len * head_dim
            total_seq_len = input_ids.shape[1] + args.max_new_tokens
            estimated_size = (
                model.config.num_hidden_layers * 2 * 
                model.config.num_key_value_heads * 
                total_seq_len * 
                (model.config.hidden_size // model.config.num_attention_heads)
            )
            print(f"Warning: Cache not populated. Using estimated size: {estimated_size:,} elements")
            cache_size = estimated_size
        
        results["baseline"] = {
            "output": output_text[len(prompt):],
            "cache_size": cache_size,
            "time": elapsed,
        }
        
        print(f"Output: {results['baseline']['output']}")
        print(f"Cache size: {cache_size:,} elements")
        print(f"Time: {elapsed:.2f}s")
    
    # Fixed compression ratio
    if args.compare_fixed:
        print("\n" + "=" * 60)
        print(f"FIXED COMPRESSION: {args.compression_method} "
              f"(ratio={(1-args.c_target)*100:.0f}%)")
        print("=" * 60)
        
        cache = DynamicCache()
        base_press = create_base_press(args.compression_method)
        
        # Set compression ratio for fixed method
        # compression_ratio = 0.5 means remove 50%, keep 50%
        if hasattr(base_press, "compression_ratio"):
            base_press.compression_ratio = 1.0 - args.c_target
            print(f"Set compression_ratio to {base_press.compression_ratio:.2f} (keep {args.c_target*100:.0f}%)")
        
        start_time = time.time()
        
        with torch.no_grad():
            with base_press(model):
                outputs = model.generate(
                    input_ids,
                    max_new_tokens=args.max_new_tokens,
                    past_key_values=cache,
                    do_sample=False,
                    output_attentions=True,
                    pad_token_id=tokenizer.eos_token_id,
                )
        
        elapsed = time.time() - start_time
        output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        cache_size = get_cache_size(cache)
        
        # Calculate expected prefill compressed cache size
        # We'll use this for accurate comparison if cache_size includes decoding
        expected_prefill_compressed = None
        if "baseline" in results:
            # Try to get prefill_seq_len from adaptive press if available
            prefill_seq_len = input_ids.shape[1]  # Use input length as prefill length
            expected_prefill_compressed = (
                model.config.num_hidden_layers * 2 *  # K + V
                model.config.num_key_value_heads *
                int(round(prefill_seq_len * args.c_target)) *  # Compressed length
                (model.config.hidden_size // model.config.num_attention_heads)
            )
        
        # If cache_size seems too large (includes decoding), use expected prefill size
        if expected_prefill_compressed is not None and cache_size > expected_prefill_compressed * 1.2:
            print(f"Note: Cache size ({cache_size:,}) appears to include decoding cache.")
            print(f"Using expected prefill compressed size: {expected_prefill_compressed:,} elements")
            cache_size = expected_prefill_compressed
        
        # If cache size is 0, estimate from compression ratio
        if cache_size == 0 and expected_prefill_compressed is not None:
            cache_size = expected_prefill_compressed
            print(f"Warning: Cache size not available. Using expected prefill compressed size: {cache_size:,} elements")
        
        results["fixed"] = {
            "output": output_text[len(prompt):],
            "cache_size": cache_size,
            "time": elapsed,
        }
        
        print(f"Output: {results['fixed']['output']}")
        print(f"Cache size: {cache_size:,} elements")
        print(f"Time: {elapsed:.2f}s")
        
        # Calculate baseline prefill cache size for comparison
        if "baseline" in results:
            prefill_seq_len = input_ids.shape[1]
            baseline_prefill_size = (
                model.config.num_hidden_layers * 2 *
                model.config.num_key_value_heads *
                prefill_seq_len *
                (model.config.hidden_size // model.config.num_attention_heads)
            )
            reduction = (1 - cache_size / baseline_prefill_size) * 100
            retention = (cache_size / baseline_prefill_size) * 100
            print(f"Cache reduction vs prefill baseline: {reduction:.1f}%")
            print(f"Retention ratio: {retention:.1f}% (target: {args.c_target*100:.0f}%)")
            if abs(retention - args.c_target*100) > 2.0:
                print(f"Warning: Actual retention ({retention:.1f}%) differs from target ({args.c_target*100:.0f}%)")
    
    # Adaptive per-head compression
    print("\n" + "=" * 60)
    print(f"ADAPTIVE PER-HEAD: {args.compression_method} "
          f"(c_target={args.c_target})")
    print("=" * 60)
    
    cache = DynamicCache()
    base_press = create_base_press(args.compression_method)
    adaptive_press = AdaptivePerHeadPress(
        press=base_press,
        policy_checkpoint=args.policy_checkpoint,
        c_target=args.c_target,
        min_tokens_per_head=args.min_tokens_per_head,
    )
    
    start_time = time.time()
    
    with torch.no_grad():
        with adaptive_press(model):
            outputs = model.generate(
                input_ids,
                max_new_tokens=args.max_new_tokens,
                past_key_values=cache,
                do_sample=False,
                output_attentions=True,
                pad_token_id=tokenizer.eos_token_id,
            )
    
    elapsed = time.time() - start_time
    output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Try to get cache size from multiple sources
    cache_size = get_cache_size(cache)
    
    # For adaptive press, use internal tracking which is more accurate
    if hasattr(adaptive_press, "get_compressed_cache_size"):
        tracked_size = adaptive_press.get_compressed_cache_size()
        if tracked_size > 0:
            cache_size = tracked_size
            print(f"Using tracked compressed cache size: {cache_size:,} elements")
        else:
            # Calculate from c_target if tracking failed
            if "baseline" in results and results["baseline"]["cache_size"] > 0:
                cache_size = int(results["baseline"]["cache_size"] * args.c_target)
                print(f"Warning: Tracking failed. Using c_target-based estimate: {cache_size:,} elements")
    elif cache_size == 0:
        # Fallback: estimate from c_target
        if "baseline" in results and results["baseline"]["cache_size"] > 0:
            cache_size = int(results["baseline"]["cache_size"] * args.c_target)
            print(f"Warning: Cache size not available. Using c_target-based estimate: {cache_size:,} elements")
    
    results["adaptive"] = {
        "output": output_text[len(prompt):],
        "cache_size": cache_size,
        "time": elapsed,
    }
    
    print(f"Output: {results['adaptive']['output']}")
    print(f"Cache size: {cache_size:,} elements")
    print(f"Time: {elapsed:.2f}s")
    
    if "baseline" in results and results["baseline"]["cache_size"] > 0:
        # Calculate baseline cache size for prefill stage only (for fair comparison)
        # Baseline cache includes prefill + decoding, but compression only happens in prefill
        prefill_seq_len = None
        if hasattr(adaptive_press, "get_prefill_seq_len"):
            prefill_seq_len = adaptive_press.get_prefill_seq_len()
        
        if prefill_seq_len is not None:
            # Calculate baseline prefill cache size
            baseline_prefill_size = (
                model.config.num_hidden_layers * 2 *  # K + V
                model.config.num_key_value_heads *
                prefill_seq_len *
                (model.config.hidden_size // model.config.num_attention_heads)
            )
            
            # Compare with prefill baseline
            reduction = (1 - cache_size / baseline_prefill_size) * 100
            actual_retention = (cache_size / baseline_prefill_size) * 100
            print(f"Cache reduction vs baseline (prefill only): {reduction:.1f}%")
            print(f"Actual retention ratio: {actual_retention:.1f}% (target: {args.c_target*100:.0f}%)")
            print(f"  Baseline prefill cache: {baseline_prefill_size:,} elements")
            print(f"  Compressed cache: {cache_size:,} elements")
            if abs(actual_retention - args.c_target*100) > 2.0:
                print(f"Warning: Actual retention ({actual_retention:.1f}%) differs from target ({args.c_target*100:.0f}%)")
        else:
            # Fallback: compare with full baseline (includes decoding)
            reduction = (1 - cache_size / results["baseline"]["cache_size"]) * 100
            actual_retention = (cache_size / results["baseline"]["cache_size"]) * 100
            print(f"Cache reduction vs baseline (full): {reduction:.1f}%")
            print(f"Actual retention ratio: {actual_retention:.1f}% (target: {args.c_target*100:.0f}%)")
            print(f"Note: Baseline includes decoding cache, so ratio may differ from target")
    elif "baseline" in results:
        print("Cache reduction vs baseline: N/A (baseline cache size is 0)")
    
    # Show per-head k statistics
    print("\nPer-head token budget statistics:")
    per_head_stats = adaptive_press.get_per_head_k_stats()
    for layer_name, k_values in list(per_head_stats.items())[:5]:  # Show first 5 layers
        print(
            f"  {layer_name}: mean={k_values.mean():.1f}, "
            f"std={k_values.std():.1f}, "
            f"min={k_values.min():.0f}, max={k_values.max():.0f}"
        )
    if len(per_head_stats) > 5:
        print(f"  ... ({len(per_head_stats) - 5} more layers)")
    
    # Summary comparison
    if len(results) > 1:
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        
        # Calculate baseline prefill cache size for accurate comparison
        baseline_prefill_size = None
        if "baseline" in results and results["baseline"]["cache_size"] > 0:
            prefill_seq_len = None
            # Try to get from adaptive press first
            if "adaptive" in results and hasattr(adaptive_press, "get_prefill_seq_len"):
                prefill_seq_len = adaptive_press.get_prefill_seq_len()
            # Fallback: use input length
            if prefill_seq_len is None:
                prefill_seq_len = input_ids.shape[1]
            
            if prefill_seq_len is not None:
                baseline_prefill_size = (
                    model.config.num_hidden_layers * 2 *  # K + V
                    model.config.num_key_value_heads *
                    prefill_seq_len *
                    (model.config.hidden_size // model.config.num_attention_heads)
                )
        
        for name, result in results.items():
            print(f"\n{name.upper()}:")
            print(f"  Cache size: {result['cache_size']:,} elements")
            print(f"  Time: {result['time']:.2f}s")
            if name != "baseline":
                if baseline_prefill_size is not None:
                    # Use prefill baseline for accurate compression ratio
                    reduction = (1 - result["cache_size"] / baseline_prefill_size) * 100
                    retention = (result["cache_size"] / baseline_prefill_size) * 100
                    print(f"  Cache reduction vs prefill baseline: {reduction:.1f}%")
                    if name in ["adaptive", "fixed"]:
                        print(f"  Retention ratio: {retention:.1f}% (target: {args.c_target*100:.0f}%)")
                        if abs(retention - args.c_target*100) > 2.0:
                            print(f"  Warning: Actual retention ({retention:.1f}%) differs from target ({args.c_target*100:.0f}%)")
                elif "baseline" in results and results["baseline"]["cache_size"] > 0:
                    # Fallback: use full baseline
                    reduction = (
                        1 - result["cache_size"] / results["baseline"]["cache_size"]
                    ) * 100
                    print(f"  Cache reduction vs full baseline: {reduction:.1f}%")
                    print(f"  (Note: Baseline includes decoding cache)")
                elif "baseline" in results:
                    print(f"  Cache reduction: N/A (baseline cache size is 0)")
            print(f"  Output: {result['output'][:100]}...")


if __name__ == "__main__":
    main()

