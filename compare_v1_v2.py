"""
ConvNeXt V1 vs V2 성능 비교 스크립트
- 동일 크기 모델의 Top-1/5 accuracy, 파라미터 수, 처리 속도 비교

실행 예시:
  python compare_v1_v2.py \\
    --data_path /nas/datahub/imagenet \\
    --gpu 4 \\
    --size huge
"""
import os
import time
import argparse

import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from models.convnextv1 import MODEL_CONFIGS as V1_CONFIGS
from models.convnextv2 import MODEL_CONFIGS as V2_CONFIGS

# 모델 크기별 체크포인트 및 논문 기재 성능
COMPARISON_TABLE = {
    "tiny": {
        "v1": {"ckpt": "checkpoints/convnext_tiny_1k_224.pt",   "paper_top1": 82.1},
        "v2": {"ckpt": "checkpoints/convnextv2_tiny_1k_224.pt", "paper_top1": 82.7},
    },
    "base": {
        "v1": {"ckpt": "checkpoints/convnext_base_1k_224.pt",   "paper_top1": 83.8},
        "v2": {"ckpt": "checkpoints/convnextv2_base_1k_224.pt", "paper_top1": 84.9},
    },
    "large": {
        "v1": {"ckpt": "checkpoints/convnext_large_1k_224.pt",  "paper_top1": 84.3},
        "v2": {"ckpt": "checkpoints/convnextv2_large_1k_224.pt","paper_top1": 85.5},
    },
    "huge": {
        "v1": {"ckpt": "checkpoints/convnext_huge_1k_224.pt",   "paper_top1": 84.7},
        "v2": {"ckpt": "checkpoints/convnextv2_huge_1k_224.pt", "paper_top1": 86.3},
    },
}


def get_args():
    parser = argparse.ArgumentParser("ConvNeXt V1 vs V2 Comparison")
    parser.add_argument("--data_path", default="/nas/datahub/imagenet", type=str)
    parser.add_argument("--size", default="huge", type=str,
                        choices=list(COMPARISON_TABLE.keys()),
                        help="비교할 모델 크기")
    parser.add_argument("--input_size", default=224, type=int)
    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--gpu", default=4, type=int)
    parser.add_argument("--throughput_only", action="store_true",
                        help="처리 속도만 측정 (ImageNet eval 생략)")
    return parser.parse_args()


def load_model(model_fn, ckpt_path, device):
    model = model_fn(num_classes=1000)
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    state_dict = checkpoint.get("model", checkpoint)
    state_dict = {k: v for k, v in state_dict.items()
                  if not k.startswith("decoder") and not k.startswith("mask_token")}
    model.load_state_dict(state_dict, strict=False)
    return model.to(device)


def build_dataloader(args):
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    val_transform = transforms.Compose([
        transforms.Resize(int(args.input_size * 256 / 224)),
        transforms.CenterCrop(args.input_size),
        transforms.ToTensor(),
        normalize,
    ])
    val_dir = os.path.join(args.data_path, "val")
    dataset = datasets.ImageFolder(val_dir, transform=val_transform)
    return torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    top1_correct = top5_correct = total = 0
    start = time.time()

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        outputs = model(images)

        _, pred_top5 = outputs.topk(5, dim=1, largest=True, sorted=True)
        targets_expanded = targets.view(-1, 1).expand_as(pred_top5)
        top1_correct += pred_top5[:, :1].eq(targets_expanded[:, :1]).sum().item()
        top5_correct += pred_top5.eq(targets_expanded).sum().item()
        total += targets.size(0)

    elapsed = time.time() - start
    return (top1_correct / total * 100,
            top5_correct / total * 100,
            total / elapsed)


@torch.no_grad()
def measure_throughput(model, device, input_size=224, batch_size=128, n_iters=100):
    model.eval()
    dummy = torch.randn(batch_size, 3, input_size, input_size).to(device)

    # warmup
    for _ in range(10):
        model(dummy)
    torch.cuda.synchronize()

    start = time.time()
    for _ in range(n_iters):
        model(dummy)
    torch.cuda.synchronize()

    elapsed = time.time() - start
    return batch_size * n_iters / elapsed


def print_table(results):
    print("\n" + "=" * 70)
    print(f"{'':20} {'V1':>20} {'V2':>20} {'개선':>8}")
    print("=" * 70)

    metrics = [
        ("Top-1 Accuracy (%)", "top1", ".2f"),
        ("Top-5 Accuracy (%)", "top5", ".2f"),
        ("파라미터 수 (M)",     "params", ".1f"),
        ("처리 속도 (img/s)",  "throughput", ".0f"),
        ("논문 Top-1 (%)",     "paper_top1", ".1f"),
    ]

    for label, key, fmt in metrics:
        v1_val = results["v1"].get(key)
        v2_val = results["v2"].get(key)
        if v1_val is None or v2_val is None:
            continue
        diff = v2_val - v1_val
        sign = "+" if diff > 0 else ""
        print(f"{label:20} {v1_val:>20{fmt}} {v2_val:>20{fmt}} {sign}{diff:>7{fmt}}")

    print("=" * 70)


def main():
    args = get_args()
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"[디바이스] {device}")
    print(f"[비교 크기] {args.size.upper()}")

    cfg = COMPARISON_TABLE[args.size]
    v1_name = f"convnext_{args.size}"
    v2_name = f"convnextv2_{args.size}"

    results = {}

    for version, model_name, configs, info in [
        ("v1", v1_name, V1_CONFIGS, cfg["v1"]),
        ("v2", v2_name, V2_CONFIGS, cfg["v2"]),
    ]:
        print(f"\n[{version.upper()}] {model_name} 로딩...")
        model = load_model(configs[model_name], info["ckpt"], device)

        n_params = model.get_num_params() / 1e6
        throughput = measure_throughput(model, device, args.input_size, args.batch_size)
        print(f"  파라미터: {n_params:.1f}M  |  처리 속도: {throughput:.0f} img/s")

        entry = {
            "params": n_params,
            "throughput": throughput,
            "paper_top1": info["paper_top1"],
        }

        if not args.throughput_only:
            loader = build_dataloader(args)
            print(f"  ImageNet eval 중...")
            top1, top5, _ = evaluate(model, loader, device)
            entry["top1"] = top1
            entry["top5"] = top5
            print(f"  Top-1: {top1:.2f}%  Top-5: {top5:.2f}%")

        results[version] = entry

    print_table(results)

    # 결과 저장
    os.makedirs("results", exist_ok=True)
    result_file = f"results/compare_v1_v2_{args.size}.txt"
    with open(result_file, "w") as f:
        f.write(f"ConvNeXt V1 vs V2 비교 ({args.size.upper()})\n")
        f.write("=" * 50 + "\n")
        for version, entry in results.items():
            f.write(f"\n[{version.upper()}]\n")
            for k, v in entry.items():
                f.write(f"  {k}: {v}\n")
    print(f"\n결과 저장: {result_file}")


if __name__ == "__main__":
    main()
