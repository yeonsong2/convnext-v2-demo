"""
Ablation 결과 시각화 (논문 Table 3, 4 수치 기반)
실행: python ablation_figures.py
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

os.makedirs("figures", exist_ok=True)

C_V1  = "#5B8DB8"
C_V2  = "#E07B54"
C_GRN = "#6DBF87"
C_BAR = "#9B7FD4"


def save(fig, name):
    path = f"figures/{name}"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  저장: {path}")


def fig_ablation_grn():
    """논문 Table 3: GRN 구성 요소 ablation."""
    configs = [
        "Baseline\n(no norm)",
        "LayerNorm",
        "LayerScale\n(V1)",
        "BatchNorm",
        "GRN\n(V2, ours)",
    ]
    top1 = [81.9, 82.1, 82.3, 82.1, 82.7]

    colors = [C_V1]*4 + [C_V2]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.bar(configs, top1, color=colors, edgecolor="white", width=0.55)

    for bar, val in zip(bars, top1):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=10,
                fontweight="bold" if val == max(top1) else "normal")

    ax.set_ylabel("Top-1 Accuracy (%)")
    ax.set_title("Ablation: Normalization Strategy (ConvNeXt-T, ImageNet-1K)\n[Paper Table 3]",
                 fontsize=11)
    ax.set_ylim(81, 83.3)
    ax.grid(axis="y", alpha=0.3)

    # GRN 강조 화살표
    ax.annotate("Best: GRN (+0.8%\nvs Baseline)",
                xy=(4, 82.7), xytext=(3.0, 83.1),
                arrowprops=dict(arrowstyle="->", color=C_V2),
                color=C_V2, fontsize=9)

    fig.tight_layout()
    save(fig, "ablation_grn.png")


def fig_ablation_fcmae():
    """논문 Table 4: FCMAE 사전학습 효과."""
    settings = [
        "Scratch\n(1K only)",
        "FCMAE\nPretrain",
        "FCMAE +\nGRN (Full V2)",
    ]
    top1_tiny  = [82.1, 82.4, 82.7]
    top1_base  = [83.8, 84.4, 84.9]
    top1_large = [84.3, 85.0, 85.5]

    x = np.arange(len(settings))
    w = 0.25
    fig, ax = plt.subplots(figsize=(9, 5))

    for i, (vals, label, color) in enumerate([
        (top1_tiny,  "Tiny",  "#A8C9E4"),
        (top1_base,  "Base",  C_V2),
        (top1_large, "Large", "#C0392B"),
    ]):
        offset = (i - 1) * w
        bars = ax.bar(x + offset, vals, w, label=label, color=color, edgecolor="white")
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f"{val:.1f}", ha="center", va="bottom", fontsize=8)

    ax.set_ylabel("Top-1 Accuracy (%)")
    ax.set_title("Ablation: Effect of FCMAE Pre-training + GRN [Paper Table 4]", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(settings)
    ax.set_ylim(81.5, 86.5)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    save(fig, "ablation_fcmae.png")


def fig_grn_formula():
    """GRN 수식 설명 다이어그램."""
    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.axis("off")
    ax.set_facecolor("#F8F9FA")
    fig.patch.set_facecolor("#F8F9FA")

    ax.text(0.5, 0.92, "Global Response Normalization (GRN)", ha="center",
            fontsize=14, fontweight="bold", transform=ax.transAxes)

    formula = (
        r"$\mathbf{GRN}(X) = \gamma \cdot \left(X \cdot N(X)\right) + \beta + X$"
        "\n\n"
        r"where  $G(X) = \|X\|_2$,   $N(X) = G(X) \;/\; \mathrm{mean}(G(X))$"
    )
    ax.text(0.5, 0.55, formula, ha="center", va="center", fontsize=12,
            transform=ax.transAxes,
            bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="#E07B54", lw=2))

    steps = [
        ("① Aggregation",  r"$G(X)_c = \|X_c\|_2$",         "Compute L2 norm per channel"),
        ("② Normalization", r"$N(X)_c = G(X)_c / mean_c$",   "Normalize across channels"),
        ("③ Calibration",  r"$X_c \;\times\; N(X)_c$",        "Scale features by relative importance"),
    ]
    for i, (title, eq, desc) in enumerate(steps):
        x_pos = 0.15 + i * 0.35
        ax.text(x_pos, 0.15, f"{title}\n{eq}\n{desc}",
                ha="center", va="center", fontsize=9, transform=ax.transAxes,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#FDE8D8", edgecolor=C_V2))

    fig.tight_layout()
    save(fig, "grn_formula.png")


if __name__ == "__main__":
    print("Ablation 그래프 생성 중...")
    fig_ablation_grn()
    fig_ablation_fcmae()
    fig_grn_formula()
    print("완료!")
