"""
커스텀 이미지 추론 데모
실행 예시:
  # 단일 이미지
  python demo.py --image path/to/image.jpg --checkpoint checkpoints/convnextv2_huge_1k_224.pt

  # 폴더 내 전체 이미지
  python demo.py --image_dir path/to/folder --checkpoint checkpoints/convnextv2_huge_1k_224.pt

  # V1 vs V2 나란히 비교
  python demo.py --image path/to/image.jpg \\
    --checkpoint checkpoints/convnextv2_huge_1k_224.pt \\
    --v1_checkpoint checkpoints/convnext_huge_1k_224.pt \\
    --compare
"""
import os
import argparse
import json
import urllib.request

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image

from models.convnextv2 import MODEL_CONFIGS as V2_CONFIGS
from models.convnextv1 import MODEL_CONFIGS as V1_CONFIGS


IMAGENET_LABELS_URL = (
    "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels"
    "/master/imagenet-simple-labels.json"
)
LABELS_FILE = "checkpoints/imagenet_labels.json"


def get_imagenet_labels():
    if os.path.exists(LABELS_FILE):
        with open(LABELS_FILE) as f:
            return json.load(f)
    print("[레이블 다운로드] ImageNet class names...")
    urllib.request.urlretrieve(IMAGENET_LABELS_URL, LABELS_FILE)
    with open(LABELS_FILE) as f:
        return json.load(f)


def get_args():
    parser = argparse.ArgumentParser("ConvNeXt V2 Demo")
    parser.add_argument("--image", type=str, help="단일 이미지 경로")
    parser.add_argument("--image_dir", type=str, help="이미지 폴더 경로")
    parser.add_argument("--model", default="convnextv2_huge", type=str,
                        choices=list(V2_CONFIGS.keys()))
    parser.add_argument("--checkpoint", required=True, type=str,
                        help="ConvNeXt V2 체크포인트 경로")
    parser.add_argument("--v1_checkpoint", type=str, default=None,
                        help="ConvNeXt V1 체크포인트 (--compare 사용 시)")
    parser.add_argument("--compare", action="store_true",
                        help="V1 vs V2 나란히 비교")
    parser.add_argument("--input_size", default=224, type=int)
    parser.add_argument("--topk", default=5, type=int)
    parser.add_argument("--gpu", default=4, type=int)
    return parser.parse_args()


def build_transform(input_size):
    return transforms.Compose([
        transforms.Resize(int(input_size * 256 / 224)),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])


def load_v2_model(model_name, ckpt_path, device):
    model = V2_CONFIGS[model_name](num_classes=1000)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state_dict = ckpt.get("model", ckpt)
    state_dict = {k: v for k, v in state_dict.items()
                  if not k.startswith("decoder") and not k.startswith("mask_token")}
    model.load_state_dict(state_dict, strict=False)
    return model.to(device).eval()


def load_v1_model(model_name, ckpt_path, device):
    model = V1_CONFIGS[model_name](num_classes=1000)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state_dict = ckpt.get("model", ckpt)
    model.load_state_dict(state_dict, strict=False)
    return model.to(device).eval()


@torch.no_grad()
def predict(model, image_tensor, device, labels, topk=5):
    image_tensor = image_tensor.to(device)
    logits = model(image_tensor.unsqueeze(0))
    probs = F.softmax(logits, dim=1)[0]
    top_probs, top_indices = probs.topk(topk)
    return [(labels[idx.item()], prob.item()) for idx, prob in zip(top_indices, top_probs)]


def print_predictions(name, preds):
    print(f"\n  [{name}]")
    for rank, (label, prob) in enumerate(preds, 1):
        bar = "█" * int(prob * 30)
        print(f"  {rank}. {label:<30} {prob*100:5.1f}%  {bar}")


def process_image(image_path, v2_model, transform, labels, args, device, v1_model=None):
    try:
        img = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"[오류] {image_path}: {e}")
        return

    tensor = transform(img)

    print(f"\n{'='*60}")
    print(f"이미지: {os.path.basename(image_path)}")
    print(f"크기:   {img.size[0]}x{img.size[1]}")
    print("=" * 60)

    v2_preds = predict(v2_model, tensor, device, labels, args.topk)
    print_predictions("ConvNeXt V2", v2_preds)

    if args.compare and v1_model is not None:
        v1_preds = predict(v1_model, tensor, device, labels, args.topk)
        print_predictions("ConvNeXt V1", v1_preds)

        print(f"\n  [V1 vs V2 Top-1 비교]")
        v1_top = v1_preds[0]
        v2_top = v2_preds[0]
        match = "일치" if v1_top[0] == v2_top[0] else "불일치"
        print(f"  V1: {v1_top[0]:<30} ({v1_top[1]*100:.1f}%)")
        print(f"  V2: {v2_top[0]:<30} ({v2_top[1]*100:.1f}%)")
        print(f"  결과: {match}")


def main():
    args = get_args()
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"[디바이스] {device}")

    labels = get_imagenet_labels()
    transform = build_transform(args.input_size)

    print(f"[V2 모델 로딩] {args.model}")
    v2_model = load_v2_model(args.model, args.checkpoint, device)

    v1_model = None
    if args.compare and args.v1_checkpoint:
        v1_name = args.model.replace("v2_", "_")  # convnextv2_huge -> convnext_huge
        print(f"[V1 모델 로딩] {v1_name}")
        v1_model = load_v1_model(v1_name, args.v1_checkpoint, device)

    # 이미지 목록 수집
    image_paths = []
    if args.image:
        image_paths.append(args.image)
    if args.image_dir:
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        for fname in sorted(os.listdir(args.image_dir)):
            if os.path.splitext(fname)[1].lower() in exts:
                image_paths.append(os.path.join(args.image_dir, fname))

    if not image_paths:
        print("[오류] --image 또는 --image_dir 를 지정하세요.")
        return

    print(f"\n총 {len(image_paths)}장 처리 시작\n")
    for path in image_paths:
        process_image(path, v2_model, transform, labels, args, device, v1_model)


if __name__ == "__main__":
    main()
