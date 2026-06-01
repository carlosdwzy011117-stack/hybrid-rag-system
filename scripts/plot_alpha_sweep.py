"""Plot alpha sweep results for Weighted Fusion retriever.

Reads alpha sweep data from Day 7 experiment (hardcoded constants below) and
produces a publication-style figure showing Recall@10 / MRR / NDCG@10 vs alpha,
with peak annotations and baseline endpoint markers.

Output:
    docs/alpha_sweep.png

Usage:
    python scripts/plot_alpha_sweep.py
"""

import os
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick


# ============================================================
# Data from Day 7 alpha sweep experiment (BEIR/SciFact)
# ============================================================
ALPHAS    = [0.0,    0.1,    0.2,    0.3,    0.5,    0.7,    1.0]
RECALL_10 = [0.8452, 0.8486, 0.8502, 0.8562, 0.8246, 0.7585, 0.6862]
MRR       = [0.6845, 0.6913, 0.6934, 0.6850, 0.6590, 0.6019, 0.5242]
NDCG_10   = [0.7200, 0.7252, 0.7269, 0.7239, 0.6952, 0.6352, 0.5597]

# Peak markers
NDCG_PEAK_ALPHA = 0.2  # also MRR peak
RECALL_PEAK_ALPHA = 0.3


# ============================================================
# Plotting
# ============================================================
def plot_alpha_sweep(save_path: str = "docs/alpha_sweep.png") -> None:
    """Plot alpha sweep with peak annotations and baseline labels."""
    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=120)

    # --- Three metric curves ---
    ax.plot(ALPHAS, RECALL_10, marker="o", linewidth=2.2, label="Recall@10", color="#1f77b4")
    ax.plot(ALPHAS, MRR,       marker="s", linewidth=2.2, label="MRR",       color="#ff7f0e")
    ax.plot(ALPHAS, NDCG_10,   marker="^", linewidth=2.2, label="NDCG@10",   color="#2ca02c")

    # --- Peak annotations ---
    # Recall peak at α=0.3
    recall_peak_idx = ALPHAS.index(RECALL_PEAK_ALPHA)
    ax.annotate(
        f"Recall peak\n({RECALL_PEAK_ALPHA}, {RECALL_10[recall_peak_idx]:.4f})",
        xy=(RECALL_PEAK_ALPHA, RECALL_10[recall_peak_idx]),
        xytext=(0.42, 0.88),
        fontsize=9,
        ha="center",
        color="#1f77b4",
        arrowprops=dict(arrowstyle="->", color="#1f77b4", lw=1.2),
    )

    # NDCG/MRR peak at α=0.2
    ndcg_peak_idx = ALPHAS.index(NDCG_PEAK_ALPHA)
    ax.annotate(
        f"NDCG & MRR peak\n({NDCG_PEAK_ALPHA}, {NDCG_10[ndcg_peak_idx]:.4f} / {MRR[ndcg_peak_idx]:.4f})",
        xy=(NDCG_PEAK_ALPHA, NDCG_10[ndcg_peak_idx]),
        xytext=(0.05, 0.62),
        fontsize=9,
        ha="left",
        color="#2ca02c",
        arrowprops=dict(arrowstyle="->", color="#2ca02c", lw=1.2),
    )

    # --- Baseline endpoint annotations ---
    ax.axvline(x=0.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.axvline(x=1.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.text(0.0, 0.54, "α=0\n(Dense baseline)",
            ha="center", va="top", fontsize=8.5, color="gray",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="gray", alpha=0.85))
    ax.text(1.0, 0.54, "α=1\n(BM25 baseline)",
            ha="center", va="top", fontsize=8.5, color="gray",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="gray", alpha=0.85))

    # --- Axes / labels / styling ---
    ax.set_xlabel(r"$\alpha$  (BM25 weight; $1-\alpha$ = Dense weight)", fontsize=11)
    ax.set_ylabel("Metric value", fontsize=11)
    ax.set_title(
        "Weighted Fusion: α Sweep on BEIR/SciFact\n"
        "Score-based fusion (min-max + weighted sum) outperforms both single-path baselines",
        fontsize=11.5,
        pad=12,
    )
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(0.50, 0.92)
    ax.set_xticks(ALPHAS)
    ax.yaxis.set_major_formatter(mtick.FormatStrFormatter("%.2f"))
    ax.grid(True, alpha=0.3, linestyle=":")
    ax.legend(loc="upper right", fontsize=10, framealpha=0.92)

    # --- Tight layout + save ---
    plt.tight_layout()

    # Make sure docs/ exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved figure to: {save_path}")

    plt.show()


if __name__ == "__main__":
    plot_alpha_sweep()