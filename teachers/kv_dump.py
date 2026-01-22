# teachers/kv_dump.py
"""
Dump KV cache (K, V) and key-level attention importance u from a teacher LLM
(meta-llama/Llama-2-7b-hf, Qwen2, etc.) on long-context English data.

Each output file is a single sample:
  {
    "K": Tensor[L, H, T_kv, d_head],
    "V": Tensor[L, H, T_kv, d_head],
    "u": Tensor[L, H, T_kv],
    "input_ids": Tensor[T_kv],
  }

This format matches policy/train.py (KVSampleDataset).
"""

from __future__ import annotations
import os
import argparse
from pathlib import Path
from typing import List, Dict, Any

import torch
from torch import Tensor
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model_and_tokenizer(
    model_name: str,
    device: str = "cuda",
    dtype: str = "auto",
) -> tuple[torch.nn.Module, AutoTokenizer]:
    """
    Load a HF CausalLM and tokenizer.

    For meta-llama/Llama-2-7b-hf and Qwen2 HF models, this should work as-is.
    """
    if dtype == "auto":
        torch_dtype = torch.float16 if "cuda" in device else torch.float32
    else:
        torch_dtype = getattr(torch, dtype)

    print(f"Loading model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        device_map="auto" if "cuda" in device else None,
        trust_remote_code=True,
    )
    if "cuda" not in device:
        model = model.to(device)

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.eval()
    return model, tokenizer


def load_texts(
    dataset_name: str,
    split: str,
    text_column: str,
    num_samples: int,
) -> List[str]:
    print(f"Loading {num_samples} samples from {dataset_name}:{split} ({text_column})")
    ds = load_dataset(dataset_name, split=f"{split}[:{num_samples}]")
    texts: List[str] = []
    for ex in ds:
        text = ex[text_column]
        if isinstance(text, list):
            text = " ".join(text)
        texts.append(text)
    return texts


@torch.no_grad()
def run_one_sample(
    model: torch.nn.Module,
    tokenizer,
    text: str,
    max_length: int,
    device: str,
    t_q: int | None = 256,
) -> Dict[str, Tensor]:
    """
    对单个样本跑一次 prefill，返回:
      - K: [L,H,T,d_head]
      - V: [L,H,T,d_head]
      - u: [L,H,T]
      - input_ids: [T]
    """
    encoded = tokenizer(
        text,
        max_length=max_length,
        truncation=True,
        padding="max_length",  # 固定 T_kv，方便训练
        return_tensors="pt",
    )
    encoded = {k: v.to(device) for k, v in encoded.items()}
    input_ids: Tensor = encoded["input_ids"]  # [1,T]
    attention_mask: Tensor = encoded["attention_mask"]  # [1,T]
    seq_len = input_ids.size(1)

    # 关键：同时要拿到 attentions 和 past_key_values
    outputs = model(
        **encoded,
        output_attentions=True,
        use_cache=True,
    )
    # attentions: list[L] of [1,H,T_q,T_kv] (here T_q=T_kv=seq_len for full prefill)
    attentions = outputs.attentions
    past_key_values = outputs.past_key_values  # list[L], each: (K,V,...) or similar

    L = len(past_key_values)
    # 支持不同实现的 past kv 形状，一般是 [1,H,T,d] 或 [1,H,d,T]
    sample_k, sample_v = past_key_values[0][0], past_key_values[0][1]
    if sample_k.dim() != 4:
        raise RuntimeError(f"Unexpected key shape: {sample_k.shape}")
    B, H, _, d_head_or_T = sample_k.shape
    if B != 1:
        raise RuntimeError("run_one_sample expects batch_size=1")

    # 判断形状排列
    if sample_k.size(2) == seq_len:
        # [1,H,T,d]
        head_dim = sample_k.size(3)
        layout = "bhtd"
    elif sample_k.size(3) == seq_len:
        # [1,H,d,T]，需要 transpose
        head_dim = sample_k.size(2)
        layout = "bhdT"
    else:
        raise RuntimeError(f"Cannot infer seq_len from key shape: {sample_k.shape}")

    # 收集 K, V: [L,H,T,d_head]
    K_list: List[Tensor] = []
    V_list: List[Tensor] = []
    for layer_idx, (k, v, *_) in enumerate(past_key_values):
        # k, v: [1,H,T,d] or [1,H,d,T]
        if layout == "bhtd":
            k_l = k[0]  # [H,T,d]
            v_l = v[0]
        else:
            # [1,H,d,T] -> [H,T,d]
            k_l = k[0].transpose(2, 3)
            v_l = v[0].transpose(2, 3)
        K_list.append(k_l)  # [H,T,d_head]
        V_list.append(v_l)

    # [L,H,T,d_head]
    K = torch.stack(K_list, dim=0)
    V = torch.stack(V_list, dim=0)

    # 计算 u: [L,H,T]
    u_list: List[Tensor] = []
    for layer_idx, attn in enumerate(attentions):
        # attn: [1,H,T_q,T_kv]
        A_l: Tensor = attn[0]  # [H,T_q,T_kv]
        T_q, T_kv = A_l.size(1), A_l.size(2)
        if t_q is not None and T_q > t_q:
            A_l = A_l[:, -t_q:, :]  # 只取最后 t_q 个 query
        # sum over query dim -> [H,T_kv]
        u_l = A_l.sum(dim=1)
        u_list.append(u_l)
    u = torch.stack(u_list, dim=0)  # [L,H,T_kv]

    # 一致性检查
    assert K.shape[0] == L and V.shape[0] == L and u.shape[0] == L
    assert K.shape[1] == H and V.shape[1] == H and u.shape[1] == H
    assert K.shape[2] == seq_len and V.shape[2] == seq_len and u.shape[2] == seq_len

    return {
        "K": K.cpu(),  # [L,H,T_kv,d_head]
        "V": V.cpu(),
        "u": u.cpu(),  # [L,H,T_kv]
        "input_ids": input_ids[0].cpu(),  # [T_kv]
        "attention_mask": attention_mask[0].cpu(),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dump KV (K,V,u) for policy training.")
    parser.add_argument(
        "--model-name",
        type=str,
        required=True,
        help="HF model name, e.g. meta-llama/Llama-2-7b-hf or Qwen/Qwen2-7B",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="c4",
        help="HF dataset name, e.g. c4, wikitext, etc.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Dataset split, e.g. train, validation, test.",
    )
    parser.add_argument(
        "--text-column",
        type=str,
        default="text",
        help="Text column name in the dataset.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=100,
        help="Number of samples to dump.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=2048,
        help="Max sequence length (tokens). Will pad/truncate to this.",
    )
    parser.add_argument(
        "--t-q",
        type=int,
        default=256,
        help="Number of latest queries to use when aggregating attention (u). "
             "If <=0, use all queries.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/kv_samples",
        help="Directory to save .pt sample files.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run model on.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="auto",
        help="Torch dtype: auto / float16 / bfloat16 / float32.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    out_dir = Path(args.output_dir) / args.model_name.replace("/", "_")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving KV samples to: {out_dir}")

    model, tokenizer = load_model_and_tokenizer(
        model_name=args.model_name,
        device=args.device,
        dtype=args.dtype,
    )

    texts = load_texts(
        dataset_name=args.dataset,
        split=args.split,
        text_column=args.text_column,
        num_samples=args.num_samples,
    )

    t_q = args.t_q if args.t_q > 0 else None

    for idx, text in enumerate(texts):
        try:
            sample = run_one_sample(
                model=model,
                tokenizer=tokenizer,
                text=text,
                max_length=args.max_length,
                device=args.device,
                t_q=t_q,
            )
        except Exception as e:
            print(f"[WARN] sample {idx}: error {e}, skip.")
            continue

        out_path = out_dir / f"sample_{idx:06d}.pt"
        torch.save(
            {
                "K": sample["K"],  # [L,H,T_kv,d_head]
                "V": sample["V"],
                "u": sample["u"],  # [L,H,T_kv]
                "input_ids": sample["input_ids"],
                "attention_mask": sample["attention_mask"],
            },
            out_path,
        )
        if (idx + 1) % 10 == 0:
            print(f"Saved {idx + 1}/{len(texts)} samples")

    print(f"Done. Total saved samples: {len(texts)}")


if __name__ == "__main__":
    main()