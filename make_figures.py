"""
실험 결과 시각화 스크립트
실행: python make_figures.py
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

os.makedirs("figures", exist_ok=True)

# ── 색상 ──────────────────────────────────────────────
C_V1 = "#5B8DB8"
C_V2 = "#E07B54"


def save(fig, name):
    path = f"figures/{name}"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  저장: {path}")


# ── 1. V1 vs V2 Top-1 Accuracy 비교 (모델 크기별) ──────
def fig_accuracy():
    sizes   = ["Tiny", "Base", "Large", "Huge"]
    v1_top1 = [82.1,   83.8,   84.3,   84.7]
    v2_top1 = [82.7,   84.9,   85.5,   86.3]   # 논문 수치 (Huge는 V2만)

    x = np.arange(len(sizes))
    w = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    b1 = ax.bar(x - w/2, v1_top1, w, label="ConvNeXt V1", color=C_V1, edgecolor="white")
    b2 = ax.bar(x + w/2, v2_top1, w, label="ConvNeXt V2", color=C_V2, edgecolor="white")

    # 값 레이블
    for bar in b1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                f"{bar.get_height():.1f}", ha="center", va="bottom", fontsize=9)
    for bar in b2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                f"{bar.get_height():.1f}", ha="center", va="bottom", fontsize=9,
                color=C_V2, fontweight="bold")

    ax.set_xlabel("Model Size")
    ax.set_ylabel("Top-1 Accuracy (%)")
    ax.set_title("ConvNeXt V1 vs V2 — ImageNet-1K Top-1 Accuracy")
    ax.set_xticks(x)
    ax.set_xticklabels(sizes)
    ax.set_ylim(80, 88)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    save(fig, "accuracy_comparison.png")


# ── 2. V2-Huge 재현 결과 vs 논문 수치 ──────────────────
def fig_reproduction():
    labels  = ["Top-1 (Paper)", "Top-1 (Ours)", "Top-5 (Ours)"]
    values  = [86.3,           86.13,          97.76]
    colors  = ["#aaa",         C_V2,           "#6DBF87"]

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(labels, values, color=colors, edgecolor="white", width=0.5)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f"{val:.2f}%", ha="center", va="bottom", fontsize=11, fontweight="bold")

    ax.set_ylabel("Accuracy (%)")
    ax.set_title("ConvNeXt V2-Huge — Paper vs Reproduced")
    ax.set_ylim(84, 100)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    save(fig, "reproduction.png")


# ── 3. V1 vs V2 처리 속도 vs 정확도 scatter ────────────
def fig_tradeoff():
    data = {
        "V1-Tiny":  (82.1, 311, C_V1),
        "V1-Base":  (83.8, 238, C_V1),
        "V1-Large": (84.3, 203, C_V1),
        "V2-Tiny":  (82.7, 290, C_V2),
        "V2-Base":  (84.9, 201, C_V2),
        "V2-Large": (85.5, 154, C_V2),
        "V2-Huge":  (86.3,  56, C_V2),
    }

    fig, ax = plt.subplots(figsize=(8, 5))
    for name, (acc, speed, color) in data.items():
        ax.scatter(speed, acc, color=color, s=120, zorder=3)
        ax.annotate(name, (speed, acc), textcoords="offset points",
                    xytext=(5, 4), fontsize=8)

    v1_patch = mpatches.Patch(color=C_V1, label="ConvNeXt V1")
    v2_patch = mpatches.Patch(color=C_V2, label="ConvNeXt V2")
    ax.legend(handles=[v1_patch, v2_patch])
    ax.set_xlabel("Throughput (img/s)")
    ax.set_ylabel("Top-1 Accuracy (%)")
    ax.set_title("Accuracy vs Throughput Trade-off")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    save(fig, "tradeoff.png")


# ── 4. Demo 결과 시각화 ────────────────────────────────
def fig_demo():
    demo_results = {
        "golden_retriever": {
            "V2": [("Golden Retriever", 0.898), ("Labrador Retriever", 0.007),
                   ("Kuvasz", 0.003), ("Irish Setter", 0.001)],
            "V1": [("Golden Retriever", 0.944), ("Kuvasz", 0.003),
                   ("Labrador Retriever", 0.002), ("Pyrenean Mountain Dog", 0.001)],
        },
        "saluki_dog": {
            "V2": [("Saluki", 0.517), ("Whippet", 0.354),
                   ("Ibizan Hound", 0.006), ("Scottish Deerhound", 0.002)],
            "V1": [("Saluki", 0.827), ("Whippet", 0.094),
                   ("Great Dane", 0.003), ("Ibizan Hound", 0.002)],
        },
    }

    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    fig.suptitle("Demo: ConvNeXt V1 vs V2 Predictions (Top-4)", fontsize=13)

    for row, (img_name, preds) in enumerate(demo_results.items()):
        for col, (version, results) in enumerate(preds.items()):
            ax = axes[row][col]
            labels = [r[0] for r in results]
            probs  = [r[1] * 100 for r in results]
            color  = C_V2 if version == "V2" else C_V1

            bars = ax.barh(labels[::-1], probs[::-1], color=color, edgecolor="white")
            for bar, prob in zip(bars, probs[::-1]):
                ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                        f"{prob:.1f}%", va="center", fontsize=8)

            ax.set_xlim(0, 110)
            ax.set_title(f"{img_name.replace('_', ' ').title()} — {version}", fontsize=10)
            ax.grid(axis="x", alpha=0.3)

    fig.tight_layout()
    save(fig, "demo_results.png")


if __name__ == "__main__":
    print("그래프 생성 중...")
    fig_accuracy()
    fig_reproduction()
    fig_tradeoff()
    fig_demo()
    print("완료!")
