"""
체크포인트 다운로드 스크립트
실행: python download_checkpoints.py
"""
import os
import urllib.request

CHECKPOINTS_DIR = os.path.join(os.path.dirname(__file__), "checkpoints")
os.makedirs(CHECKPOINTS_DIR, exist_ok=True)

# ConvNeXt V2 ImageNet-1K fine-tuned checkpoints
V2_CHECKPOINTS = {
    "convnextv2_tiny_1k_224":   "https://dl.fbaipublicfiles.com/convnext/convnextv2/im1k/convnextv2_tiny_1k_224_ema.pt",
    "convnextv2_base_1k_224":   "https://dl.fbaipublicfiles.com/convnext/convnextv2/im1k/convnextv2_base_1k_224_ema.pt",
    "convnextv2_large_1k_224":  "https://dl.fbaipublicfiles.com/convnext/convnextv2/im1k/convnextv2_large_1k_224_ema.pt",
    "convnextv2_huge_1k_224":   "https://dl.fbaipublicfiles.com/convnext/convnextv2/im1k/convnextv2_huge_1k_224_ema.pt",
}

# ConvNeXt V1 ImageNet-1K fine-tuned checkpoints (비교용)
V1_CHECKPOINTS = {
    "convnext_tiny_1k_224":     "https://dl.fbaipublicfiles.com/convnext/convnext_tiny_1k_224_ema.pth",
    "convnext_base_1k_224":     "https://dl.fbaipublicfiles.com/convnext/convnext_base_1k_224_ema.pth",
    "convnext_large_1k_224":    "https://dl.fbaipublicfiles.com/convnext/convnext_large_1k_224_ema.pth",
    "convnext_huge_1k_224":     "https://dl.fbaipublicfiles.com/convnext/convnext_huge_1k_224_ema.pth",
}


def download(name, url):
    filename = os.path.join(CHECKPOINTS_DIR, f"{name}.pt")
    if os.path.exists(filename):
        print(f"[skip] {name} already exists")
        return filename

    print(f"[download] {name}")
    print(f"  URL: {url}")

    def progress(count, block_size, total_size):
        pct = count * block_size / total_size * 100
        mb = count * block_size / 1024 / 1024
        total_mb = total_size / 1024 / 1024
        print(f"\r  {pct:.1f}%  {mb:.1f}/{total_mb:.1f} MB", end="", flush=True)

    urllib.request.urlretrieve(url, filename, reporthook=progress)
    print(f"\n  saved: {filename}")
    return filename


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="all",
                        help="다운로드할 모델 (예: convnextv2_huge_1k_224 또는 all)")
    parser.add_argument("--version", default="both", choices=["v1", "v2", "both"])
    args = parser.parse_args()

    all_ckpts = {}
    if args.version in ("v2", "both"):
        all_ckpts.update(V2_CHECKPOINTS)
    if args.version in ("v1", "both"):
        all_ckpts.update(V1_CHECKPOINTS)

    if args.model == "all":
        targets = all_ckpts
    else:
        if args.model not in all_ckpts:
            print(f"Unknown model: {args.model}")
            print(f"Available: {list(all_ckpts.keys())}")
            exit(1)
        targets = {args.model: all_ckpts[args.model]}

    for name, url in targets.items():
        download(name, url)

    print("\n완료! 다운로드된 체크포인트:")
    for f in os.listdir(CHECKPOINTS_DIR):
        path = os.path.join(CHECKPOINTS_DIR, f)
        size_mb = os.path.getsize(path) / 1024 / 1024
        print(f"  {f}  ({size_mb:.1f} MB)")
