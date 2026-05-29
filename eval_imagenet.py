"""
ImageNet Validation 평가 스크립트
실행 예시:
  python eval_imagenet.py \\
    --model convnextv2_huge \\
    --checkpoint checkpoints/convnextv2_huge_1k_224.pt \\
    --data_path /nas/datahub/imagenet \\
    --gpu 4
"""
import os
import time
import argparse

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from models.convnextv2 import MODEL_CONFIGS as V2_CONFIGS


def get_args():
    parser = argparse.ArgumentParser("ConvNeXt V2 ImageNet Evaluation")
    parser.add_argument("--model", default="convnextv2_huge", type=str,
                        choices=list(V2_CONFIGS.keys()))
    parser.add_argument("--checkpoint", required=True, type=str)
    parser.add_argument("--data_path", default="/nas/datahub/imagenet", type=str)
    parser.add_argument("--input_size", default=224, type=int)
    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--gpu", default=4, type=int)
    return parser.parse_args()


def load_model(args):
    print(f"[모델 로딩] {args.model}")
    model = V2_CONFIGS[args.model](num_classes=1000)

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    state_dict = checkpoint.get("model", checkpoint)

    # FCMAE pretrained weight의 decoder key 제거
    state_dict = {k: v for k, v in state_dict.items()
                  if not k.startswith("decoder") and not k.startswith("mask_token")}

    msg = model.load_state_dict(state_dict, strict=False)
    print(f"  missing keys: {msg.missing_keys}")
    print(f"  unexpected keys: {msg.unexpected_keys}")
    print(f"  파라미터 수: {model.get_num_params() / 1e6:.1f}M")
    return model


def build_dataloader(args):
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    val_transform = transforms.Compose([
        transforms.Resize(int(args.input_size * 256 / 224)),
        transforms.CenterCrop(args.input_size),
        transforms.ToTensor(),
        normalize,
    ])

    val_dir = os.path.join(args.data_path, "val")
    dataset = datasets.ImageFolder(val_dir, transform=val_transform)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    print(f"[데이터] ImageNet val: {len(dataset):,}장")
    return loader


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    top1_correct = top5_correct = total = 0
    start = time.time()

    for i, (images, targets) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        outputs = model(images)

        _, pred_top5 = outputs.topk(5, dim=1, largest=True, sorted=True)
        targets_expanded = targets.view(-1, 1).expand_as(pred_top5)

        top1_correct += pred_top5[:, :1].eq(targets_expanded[:, :1]).sum().item()
        top5_correct += pred_top5.eq(targets_expanded).sum().item()
        total += targets.size(0)

        if (i + 1) % 50 == 0:
            elapsed = time.time() - start
            imgs_per_sec = total / elapsed
            print(f"  [{i+1}/{len(loader)}]  "
                  f"Top-1: {top1_correct/total*100:.2f}%  "
                  f"Top-5: {top5_correct/total*100:.2f}%  "
                  f"({imgs_per_sec:.0f} img/s)")

    elapsed = time.time() - start
    top1_acc = top1_correct / total * 100
    top5_acc = top5_correct / total * 100
    return top1_acc, top5_acc, elapsed


def main():
    args = get_args()
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"[디바이스] {device}")

    model = load_model(args)
    model = model.to(device)

    loader = build_dataloader(args)

    print(f"\n[평가 시작] {args.model}")
    print("-" * 60)
    top1, top5, elapsed = evaluate(model, loader, device)
    print("-" * 60)
    print(f"결과:")
    print(f"  Top-1 Accuracy: {top1:.2f}%")
    print(f"  Top-5 Accuracy: {top5:.2f}%")
    print(f"  총 소요 시간:   {elapsed/60:.1f}분")

    # 결과 저장
    os.makedirs("results", exist_ok=True)
    result_file = f"results/{args.model}_eval.txt"
    with open(result_file, "w") as f:
        f.write(f"Model: {args.model}\n")
        f.write(f"Checkpoint: {args.checkpoint}\n")
        f.write(f"Input size: {args.input_size}\n")
        f.write(f"Top-1: {top1:.2f}%\n")
        f.write(f"Top-5: {top5:.2f}%\n")
    print(f"  결과 저장: {result_file}")


if __name__ == "__main__":
    main()
