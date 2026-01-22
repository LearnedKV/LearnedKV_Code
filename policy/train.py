# # policy/train.py

# from __future__ import annotations
# import os
# import glob
# import argparse
# from typing import Dict, Any

# import torch
# from torch.utils.data import Dataset, DataLoader
# from torch import Tensor
# from tqdm import tqdm

# # from .features import HeadFeatureExtractor
# # from .model import PolicyMLP
# # from .loss import attention_mass_loss
# from policy.features import HeadFeatureExtractor
# from policy.model import PolicyMLP
# from policy.loss import attention_mass_loss


# class KVSampleDataset(Dataset):
#     """
#     简单的离线 KV 样本数据集。

#     假定 kv_dump 生成的每个 .pt 文件是一个 dict:
#       {
#         "K": Tensor[L,H,T_kv,d_head],
#         "V": Tensor[L,H,T_kv,d_head],
#         "u": Tensor[L,H,T_kv]
#       }
#     """

#     def __init__(self, root: str) -> None:
#         super().__init__()
#         self.root = root
#         self.paths = sorted(glob.glob(os.path.join(root, "*.pt")))
#         if not self.paths:
#             raise RuntimeError(f"No .pt files found in kv-root: {root}")

#     def __len__(self) -> int:
#         return len(self.paths)

#     def __getitem__(self, idx: int) -> Dict[str, Tensor]:
#         path = self.paths[idx]
#         sample: Dict[str, Any] = torch.load(path, map_location="cpu")
#         K: Tensor = sample["K"]
#         V: Tensor = sample["V"]
#         u: Tensor = sample["u"]
#         return {"K": K, "V": V, "u": u}


# def build_head_features(
#     extractor: HeadFeatureExtractor,
#     K: Tensor,
#     V: Tensor,
# ) -> Tensor:
#     """
#     对一个 batch 构造所有 head 的特征。

#     输入:
#       - K: [B,L,H,T_kv,d_head]
#       - V: [B,L,H,T_kv,d_head]

#     输出:
#       - head_features: [B,L,H,d_feat]
#     """
#     device = next(extractor.parameters()).device
#     B, L, H, T_kv, d_head = K.shape
#     feats = []
#     for b in range(B):
#         for l in range(L):
#             for h in range(H):
#                 K_blh = K[b, l, h].to(device)  # [T_kv,d_head]
#                 V_blh = V[b, l, h].to(device)
#                 f = extractor(K_blh, V_blh, layer_idx=l, head_idx=h, num_layers=L, num_heads=H)
#                 feats.append(f)
#     feats_tensor = torch.stack(feats, dim=0)  # [B*L*H,d_feat]
#     d_feat = feats_tensor.size(-1)
#     head_features = feats_tensor.view(B, L, H, d_feat)
#     return head_features


# def parse_args() -> argparse.Namespace:
#     parser = argparse.ArgumentParser(description="Train KV policy MLP with attention-mass loss.")
#     parser.add_argument("--kv-root", type=str, required=True, help="Directory containing offline KV .pt files.")
#     parser.add_argument("--batch-size", type=int, default=1, help="Samples per batch (each sample is a full long sequence).")
#     parser.add_argument("--epochs", type=int, default=1)
#     parser.add_argument("--lr", type=float, default=1e-4)
#     parser.add_argument("--hidden-dim", type=int, default=128)
#     parser.add_argument("--num-workers", type=int, default=0)
#     parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

#     parser.add_argument("--c-min", type=float, default=0.2, help="Min global compression ratio.")
#     parser.add_argument("--c-max", type=float, default=0.8, help="Max global compression ratio.")
#     parser.add_argument("--min-tokens-per-head", type=float, default=0.0, help="Minimum tokens per head (continuous capacity).")

#     parser.add_argument("--save-dir", type=str, default="ckpts/policy")
#     parser.add_argument("--save-every", type=int, default=1, help="Save checkpoint every N epochs.")

#     return parser.parse_args()


# def main() -> None:
#     args = parse_args()
#     os.makedirs(args.save-dir if hasattr(args, "save-dir") else args.save_dir, exist_ok=True)  # just in case

#     device = torch.device(args.device)
#     print(f"Using device: {device}")

#     dataset = KVSampleDataset(args.kv_root)
#     # 探测一个样本的形状
#     probe = dataset[0]
#     K0: Tensor = probe["K"]
#     L, H, T_kv, d_head = K0.shape
#     print(f"Detected shapes: L={L}, H={H}, T_kv={T_kv}, d_head={d_head}")

#     dataloader = DataLoader(
#         dataset,
#         batch_size=args.batch_size,
#         shuffle=True,
#         num_workers=args.num_workers,
#         pin_memory=True,
#     )

#     # 构造特征抽取器和 policy MLP
#     extractor = HeadFeatureExtractor(d_head=d_head).to(device)
#     d_feat = extractor.feature_dim
#     policy = PolicyMLP(d_feat=d_feat, d_hid=args.hidden_dim).to(device)

#     optimizer = torch.optim.AdamW(
#         list(extractor.parameters()) + list(policy.parameters()),
#         lr=args.lr,
#     )

#     global_step = 0
#     for epoch in range(1, args.epochs + 1):
#         extractor.train()
#         policy.train()
#         pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{args.epochs}")
#         for batch in pbar:
#             K: Tensor = batch["K"]  # [B,L,H,T_kv,d_head]
#             V: Tensor = batch["V"]
#             u: Tensor = batch["u"]  # [B,L,H,T_kv]

#             # 简单假设 batch 内所有样本形状一致
#             K = K.to(device)
#             V = V.to(device)
#             u = u.to(device)

#             # 构造每个 head 的特征
#             head_features = build_head_features(extractor, K, V)  # [B,L,H,d_feat]

#             # policy logits
#             alpha = policy(head_features)  # [B,L,H]

#             B = K.size(0)
#             # 随机采样每个样本的全局压缩比
#             c_target = torch.empty(B, device=device).uniform_(args.c_min, args.c_max)

#             loss, info = attention_mass_loss(
#                 alpha=alpha,
#                 u=u,
#                 c_target=c_target,
#                 min_tokens_per_head=args.min_tokens_per_head,
#             )

#             optimizer.zero_grad(set_to_none=True)
#             loss.backward()
#             optimizer.step()

#             global_step += 1
#             pbar.set_postfix(
#                 loss=float(loss.item()),
#                 cov=f"{info['mean_cov']:.3f}",
#                 k_star=f"{info['mean_k_star']:.1f}",
#             )

#         # 保存 checkpoint
#         if epoch % args.save_every == 0:
#             ckpt_path = os.path.join(args.save_dir, f"policy_epoch{epoch}.pt")
#             os.makedirs(args.save_dir, exist_ok=True)
#             torch.save(
#                 {
#                     "extractor": extractor.state_dict(),
#                     "policy": policy.state_dict(),
#                     "config": {
#                         "L": L,
#                         "H": H,
#                         "T_kv": T_kv,
#                         "d_head": d_head,
#                         "d_feat": d_feat,
#                         "hidden_dim": args.hidden_dim,
#                     },
#                     "epoch": epoch,
#                     "global_step": global_step,
#                 },
#                 ckpt_path,
#             )
#             print(f"Saved checkpoint to {ckpt_path}")


# if __name__ == "__main__":
#     main()
# policy/train.py

from __future__ import annotations
import os
import glob
import argparse
from typing import Dict, Any

import torch
from torch.utils.data import Dataset, DataLoader
from torch import Tensor
from tqdm import tqdm

# from .features import HeadFeatureExtractor
# from .model import PolicyMLP
# from .loss import attention_mass_loss
from policy.features import HeadFeatureExtractor
from policy.model import PolicyMLP
from policy.loss import attention_mass_loss

class KVSampleDataset(Dataset):
    """
    简单的离线 KV 样本数据集。

    假定 kv_dump 生成的每个 .pt 文件是一个 dict:
      {
        "K": Tensor[L,H,T_kv,d_head],
        "V": Tensor[L,H,T_kv,d_head],
        "u": Tensor[L,H,T_kv]
      }
    """

    def __init__(self, root: str) -> None:
        super().__init__()
        self.root = root
        self.paths = sorted(glob.glob(os.path.join(root, "*.pt")))
        if not self.paths:
            raise RuntimeError(f"No .pt files found in kv-root: {root}")

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> Dict[str, Tensor]:
        path = self.paths[idx]
        sample: Dict[str, Any] = torch.load(path, map_location="cpu")
        K: Tensor = sample["K"]
        V: Tensor = sample["V"]
        u: Tensor = sample["u"]
        return {"K": K, "V": V, "u": u}


def build_head_features(
    extractor: HeadFeatureExtractor,
    K: Tensor,
    V: Tensor,
) -> Tensor:
    """
    对一个 batch 构造所有 head 的特征。

    输入:
      - K: [B,L,H,T_kv,d_head]
      - V: [B,L,H,T_kv,d_head]

    输出:
      - head_features: [B,L,H,d_feat]
    """
    device = next(extractor.parameters()).device
    B, L, H, T_kv, d_head = K.shape
    feats = []
    for b in range(B):
        for l in range(L):
            for h in range(H):
                K_blh = K[b, l, h].to(device)  # [T_kv,d_head]
                V_blh = V[b, l, h].to(device)
                f = extractor(
                    K_blh,
                    V_blh,
                    layer_idx=l,
                    head_idx=h,
                    num_layers=L,
                    num_heads=H,
                )
                feats.append(f)
    feats_tensor = torch.stack(feats, dim=0)  # [B*L*H,d_feat]
    d_feat = feats_tensor.size(-1)
    head_features = feats_tensor.view(B, L, H, d_feat)
    return head_features


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train KV policy MLP with attention-mass loss."
    )
    parser.add_argument(
        "--kv-root",
        type=str,
        required=True,
        help="Directory containing offline KV .pt files.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Samples per batch (each sample is a full long sequence).",
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )

    parser.add_argument(
        "--c-min", type=float, default=0.2, help="Min global compression ratio."
    )
    parser.add_argument(
        "--c-max", type=float, default=0.8, help="Max global compression ratio."
    )
    parser.add_argument(
        "--min-tokens-per-head",
        type=float,
        default=0.0,
        help="Minimum tokens per head (continuous capacity).",
    )

    parser.add_argument("--save-dir", type=str, default="ckpts/policy")
    parser.add_argument(
        "--save-every", type=int, default=1, help="Save checkpoint every N epochs."
    )

    # 你命令里用到的参数：可选，用来做一致性检查
    parser.add_argument(
        "--model-dim",
        type=int,
        default=None,
        help="(optional) model hidden size, e.g. 4096 for Llama-2-7B.",
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=None,
        help="(optional) number of transformer layers; checked against KV shape.",
    )
    parser.add_argument(
        "--num-heads",
        type=int,
        default=None,
        help="(optional) number of attention heads; checked against KV shape.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    device = torch.device(args.device)
    print(f"Using device: {device}")

    dataset = KVSampleDataset(args.kv_root)
    # 探测一个样本的形状
    probe = dataset[0]
    K0: Tensor = probe["K"]
    L, H, T_kv, d_head = K0.shape
    d_model_inferred = d_head * H
    print(f"Detected shapes: L={L}, H={H}, T_kv={T_kv}, d_head={d_head}")
    print(f"Inferred model hidden size from KV: d_model={d_model_inferred}")

    # 与命令行参数做一致性检查（如果提供了的话）
    if args.num_layers is not None and args.num_layers != L:
        raise ValueError(
            f"--num-layers={args.num_layers} but KV dump has L={L}. "
            "Please check that kv_root matches the model config."
        )
    if args.num_heads is not None and args.num_heads != H:
        raise ValueError(
            f"--num-heads={args.num_heads} but KV dump has H={H}. "
            "Please check that kv_root matches the model config."
        )
    if args.model_dim is not None and args.model_dim != d_model_inferred:
        raise ValueError(
            f"--model-dim={args.model_dim} but inferred hidden size from KV is "
            f"{d_model_inferred} (= d_head * num_heads)."
        )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # 构造特征抽取器和 policy MLP
    extractor = HeadFeatureExtractor(d_head=d_head).to(device)
    d_feat = extractor.feature_dim
    policy = PolicyMLP(d_feat=d_feat, d_hid=args.hidden_dim).to(device)

    optimizer = torch.optim.AdamW(
        list(extractor.parameters()) + list(policy.parameters()),
        lr=args.lr,
    )

    global_step = 0
    for epoch in range(1, args.epochs + 1):
        extractor.train()
        policy.train()
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{args.epochs}")
        for batch in pbar:
            K: Tensor = batch["K"]  # [B,L,H,T_kv,d_head]
            V: Tensor = batch["V"]
            u: Tensor = batch["u"]  # [B,L,H,T_kv]

            # 简单假设 batch 内所有样本形状一致
            K = K.to(device)
            V = V.to(device)
            u = u.to(device)

            # 构造每个 head 的特征
            head_features = build_head_features(
                extractor, K, V
            )  # [B,L,H,d_feat]

            # policy logits
            alpha = policy(head_features)  # [B,L,H]

            B = K.size(0)
            # 随机采样每个样本的全局压缩比
            c_target = torch.empty(B, device=device).uniform_(
                args.c_min, args.c_max
            )

            loss, info = attention_mass_loss(
                alpha=alpha,
                u=u,
                c_target=c_target,
                min_tokens_per_head=args.min_tokens_per_head,
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            global_step += 1
            pbar.set_postfix(
                loss=float(loss.item()),
                cov=f"{info['mean_cov']:.3f}",
                k_star=f"{info['mean_k_star']:.1f}",
            )

        # 保存 checkpoint
        if epoch % args.save_every == 0:
            ckpt_path = os.path.join(args.save_dir, f"policy_epoch{epoch}.pt")
            os.makedirs(args.save_dir, exist_ok=True)
            torch.save(
                {
                    "extractor": extractor.state_dict(),
                    "policy": policy.state_dict(),
                    "config": {
                        "L": L,
                        "H": H,
                        "T_kv": T_kv,
                        "d_head": d_head,
                        "d_feat": d_feat,
                        "hidden_dim": args.hidden_dim,
                    },
                    "epoch": epoch,
                    "global_step": global_step,
                },
                ckpt_path,
            )
            print(f"Saved checkpoint to {ckpt_path}")


if __name__ == "__main__":
    main()
