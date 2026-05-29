"""
GRN vs LayerScale Feature Activation 시각화
- V1 (LayerScale) vs V2 (GRN) 채널 활성화 다양성 비교
실행: python visualize_grn.py --gpu 7
"""
import os
import argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch
from PIL import Image
import torchvision.transforms as transforms

from models.convnextv2 import convnextv2_large
from models.convnextv1 import convnext_large

os.makedirs("figures", exist_ok=True)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--v1_checkpoint", default="checkpoints/convnext_large_1k_224.pt")
    parser.add_argument("--v2_checkpoint", default="checkpoints/convnextv2_large_1k_224.pt")
    parser.add_argument("--image", default=None, help="이미지 경로 (없으면 랜덤 노이즈)")
    parser.add_argument("--gpu", default=7, type=int)
    return parser.parse_args()


def load_image(image_path, size=224):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    if image_path and os.path.exists(image_path):
        img = Image.open(image_path).convert("RGB")
    else:
        # ImageNet val에서 샘플 이미지 사용
        sample = "/nas/datahub/imagenet/val/n02099601/ILSVRC2012_val_00001112.JPEG"
        img = Image.open(sample).convert("RGB")
    return transform(img).unsqueeze(0)


def hook_activations(model, layer_idx=2):
    """3번째 stage의 첫 번째 블록 MLP 출력 hooking."""
    activations = {}

    def make_hook(name):
        def hook(module, input, output):
            activations[name] = output.detach().cpu()
        return hook

    # stage 2의 첫 블록 pwconv2 (MLP 최종 출력) 전/후 hooking
    block = model.stages[layer_idx][0]
    block.pwconv2.register_forward_hook(make_hook("after_mlp"))

    return activations


def compute_channel_diversity(feat):
    """채널별 활성화 분포의 다양성 측정 (각 채널의 L2 norm)."""
    # feat: (B, H, W, C) or (B, C, H, W)
    if feat.dim() == 4 and feat.shape[1] != feat.shape[-1]:
        feat = feat.permute(0, 2, 3, 1)  # → (B, H, W, C)
    B, H, W, C = feat.shape
    feat_flat = feat.reshape(-1, C)  # (B*H*W, C)
    channel_norms = feat_flat.norm(dim=0)  # (C,)
    return channel_norms.numpy()


@torch.no_grad()
def get_features(model, x, device):
    x = x.to(device)
    acts = hook_activations(model, layer_idx=2)
    _ = model(x)
    return acts["after_mlp"]


def fig_channel_diversity(v1_model, v2_model, x, device):
    """V1 vs V2 채널 활성화 분포 비교."""
    v1_feat = get_features(v1_model, x, device)
    v2_feat = get_features(v2_model, x, device)

    v1_norms = compute_channel_diversity(v1_feat)
    v2_norms = compute_channel_diversity(v2_feat)

    # 정규화
    v1_norms = v1_norms / v1_norms.max()
    v2_norms = v2_norms / v2_norms.max()

    # 상위/하위 채널 비율 (활성화 불균형 지표)
    v1_dead = (v1_norms < 0.1).mean() * 100
    v2_dead = (v2_norms < 0.1).mean() * 100

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    fig.suptitle("Channel Activation Analysis: V1 (LayerScale) vs V2 (GRN)", fontsize=13)

    C_V1, C_V2 = "#5B8DB8", "#E07B54"

    for ax, norms, label, color, dead in [
        (axes[0], v1_norms, "ConvNeXt V1 (LayerScale)", C_V1, v1_dead),
        (axes[1], v2_norms, "ConvNeXt V2 (GRN)",        C_V2, v2_dead),
    ]:
        sorted_norms = np.sort(norms)[::-1]
        ax.bar(range(len(sorted_norms)), sorted_norms, color=color, alpha=0.8, width=1.0)
        ax.axhline(0.1, color="red", linestyle="--", alpha=0.6, label=f"threshold (dead: {dead:.1f}%)")
        ax.set_title(label, fontsize=11)
        ax.set_xlabel("Channel (sorted by activation)")
        ax.set_ylabel("Normalized Activation Norm")
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    path = "figures/grn_channel_diversity.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  저장: {path}")
    print(f"  V1 dead channels: {v1_dead:.1f}%")
    print(f"  V2 dead channels: {v2_dead:.1f}%")


def fig_grn_mechanism():
    """GRN 동작 원리 다이어그램."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("V1 vs V2 Block Architecture", fontsize=13)

    C_V1, C_V2 = "#5B8DB8", "#E07B54"
    box_kw = dict(ha="center", va="center", fontsize=10,
                  bbox=dict(boxstyle="round,pad=0.4", edgecolor="gray"))

    # V1 block
    ax = axes[0]
    ax.set_xlim(0, 4); ax.set_ylim(0, 10); ax.axis("off")
    ax.set_title("ConvNeXt V1 Block", fontsize=11, color=C_V1)
    steps_v1 = [
        (2, 9.0, "Input",         "white"),
        (2, 7.5, "DWConv 7×7",    "#D6E8F5"),
        (2, 6.0, "LayerNorm",     "#D6E8F5"),
        (2, 4.5, "Linear (×4)",   "#D6E8F5"),
        (2, 3.0, "GELU",          "#D6E8F5"),
        (2, 1.5, "Linear (÷4)",   "#D6E8F5"),
        (2, 0.2, "Layer Scale ✗", "#FADADD"),
    ]
    for x, y, text, color in steps_v1:
        box_kw["bbox"]["facecolor"] = color
        ax.text(x, y, text, **box_kw)
        if y > 0.5:
            ax.annotate("", xy=(x, y - 0.6), xytext=(x, y - 0.1),
                        arrowprops=dict(arrowstyle="->", color="gray"))

    # V2 block
    ax = axes[1]
    ax.set_xlim(0, 4); ax.set_ylim(0, 10); ax.axis("off")
    ax.set_title("ConvNeXt V2 Block", fontsize=11, color=C_V2)
    steps_v2 = [
        (2, 9.0, "Input",         "white"),
        (2, 7.5, "DWConv 7×7",    "#FDE8D8"),
        (2, 6.0, "LayerNorm",     "#FDE8D8"),
        (2, 4.5, "Linear (×4)",   "#FDE8D8"),
        (2, 3.0, "GELU",          "#FDE8D8"),
        (2, 1.5, "GRN ★",         "#F4A460"),
        (2, 0.2, "Linear (÷4)",   "#FDE8D8"),
    ]
    for x, y, text, color in steps_v2:
        box_kw["bbox"]["facecolor"] = color
        ax.text(x, y, text, **box_kw)
        if y > 0.5:
            ax.annotate("", xy=(x, y - 0.6), xytext=(x, y - 0.1),
                        arrowprops=dict(arrowstyle="->", color="gray"))

    fig.tight_layout()
    path = "figures/grn_mechanism.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  저장: {path}")


def main():
    args = get_args()
    device = torch.device(f"cuda:{args.gpu}")
    print(f"[디바이스] {device}")

    print("[모델 로딩]")
    v1_model = convnext_large(num_classes=1000)
    ckpt = torch.load(args.v1_checkpoint, map_location="cpu")
    v1_model.load_state_dict(ckpt.get("model", ckpt), strict=False)
    v1_model = v1_model.to(device).eval()

    v2_model = convnextv2_large(num_classes=1000)
    ckpt = torch.load(args.v2_checkpoint, map_location="cpu")
    v2_model.load_state_dict(ckpt.get("model", ckpt), strict=False)
    v2_model = v2_model.to(device).eval()

    x = load_image(args.image)
    print(f"[이미지] shape: {x.shape}")

    print("\n[1] 채널 활성화 다양성 비교")
    fig_channel_diversity(v1_model, v2_model, x, device)

    print("\n[2] Block 구조 다이어그램")
    fig_grn_mechanism()

    print("\n완료!")


if __name__ == "__main__":
    main()
