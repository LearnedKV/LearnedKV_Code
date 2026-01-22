# LearnedKV

## Quick Start

This feature allows you to apply **learned per-layer per-head compression strategies** to any KV cache compression algorithm in kvpress.

### 1. Install Dependencies

```bash
pip install -e .
```

### 2. Train a Policy Network

#### Step 2.1: Collect KV Cache Data

```bash
python teachers/kv_dump.py \
  --model-name meta-llama/Llama-3.2-1B-Instruct \
  --dataset-name pg19 \
  --output-dir kv_data/llama-3.2-1B \
  --num-samples 1000 \
  --max-length 4096
```

#### Step 2.2: Train Policy

```bash
python policy/train.py \
  --kv-root kv_data/llama-3.2-1B \
  --batch-size 4 \
  --epochs 10 \
  --lr 1e-4 \
  --c-min 0.2 \
  --c-max 0.8 \
  --save-dir ckpts/policy/llama-3.2-1B
```

### 3. Use Adaptive Compression

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
from kvpress import ObservedAttentionPress, AdaptivePerHeadPress
import torch

# Load model
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-1B-Instruct",
    torch_dtype=torch.float16,
    device_map="cuda",
    attn_implementation="eager",
)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")

# Create adaptive press
base_press = ObservedAttentionPress()
adaptive_press = AdaptivePerHeadPress(
    press=base_press,
    policy_checkpoint="ckpts/policy/llama-3.2-1B/policy_epoch10.pt",
    c_target=0.5,  # Keep 50% of tokens
    min_tokens_per_head=1.0,
)

# Generate with adaptive compression
input_ids = tokenizer("Your prompt here", return_tensors="pt").input_ids.cuda()
cache = DynamicCache()

with adaptive_press(model):
    outputs = model.generate(
        input_ids,
        max_new_tokens=100,
        past_key_values=cache,
        output_attentions=True,
    )
```

## Run Example

```bash
python examples/adaptive_per_head_example.py \
  --model-name meta-llama/Llama-3.2-1B-Instruct \
  --policy-checkpoint ckpts/policy/llama-3.2-1B/policy_epoch10.pt \
  --compression-method observed \
  --c-target 0.5 \
  --compare-baseline \
  --compare-fixed
```

## Key Features

✅ **Works with all ScorerPress-based compression algorithms**
- ObservedAttentionPress
- SnapKVPress
- ExpectedAttentionPress
- KnormPress
- TOVAPress
- And more...

✅ **Learned from data**
- Policy network learns optimal compression strategies from real attention patterns
- Trains on self-supervised attention-mass loss

✅ **Per-layer per-head adaptation**
- Each attention head gets a different token budget
- More important heads keep more tokens
- Less important heads are compressed more aggressively

✅ **Minimal overhead**
- Policy network runs only once during prefill
- Negligible latency impact compared to fixed compression

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   Input Sequence                         │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
         ┌───────────────────────┐
         │  LLM Forward Pass     │
         │  (Full KV Cache)      │
         └───────────┬───────────┘
                     │
        ┌────────────┴────────────┐
        │                         │
        ▼                         ▼
┌───────────────┐         ┌──────────────┐
│  K, V Cache   │         │  Attention   │
│  [L,H,T,d]    │         │  Weights     │
└───────┬───────┘         └──────┬───────┘
        │                        │
        │                        ▼
        │              ┌─────────────────┐
        │              │ Key Importance  │
        │              │  u = Σ_q A_qt   │
        │              └────────┬────────┘
        │                       │
        ▼                       │
┌──────────────────────────────┴────┐
│   HeadFeatureExtractor            │
│   - Mean/variance statistics       │
│   - Positional segment features    │
│   - Structural features (l, h)     │
└──────────────┬────────────────────┘
               │
               ▼
      ┌────────────────┐
      │  Policy network    │
      │  s → alpha     │
      └────────┬───────┘
               │
               ▼
      ┌────────────────┐
      │  Softmax +     │
      │  Budget Alloc  │
      │  alpha → k_h   │
      └────────┬───────┘
               │
               ▼
   ┌───────────────────────┐
   │  Base Compression     │
   │  (Score + Top-k_h)    │
   └──────────┬────────────┘
              │
              ▼
    ┌─────────────────┐
    │ Compressed KV   │
    │ Per-head k_h    │
    └─────────────────┘
```



