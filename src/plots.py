#!/usr/bin/env python3
# plot_figure_3_4_windows.py (tidied, no overlaps)

import os
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D

# ────────────────────────── Config ──────────────────────────
L_ENC  = 96
L_LAB  = 48
L_PRED = 96
STRIDE = 24
OUT_PATH = "figs/figure_3_4.png"

W0_START = 0
W0_END   = L_ENC + L_PRED
W1_START = STRIDE
W1_END   = W1_START + L_ENC + L_PRED
AX_MAX   = max(W0_END, W1_END) + 10

mpl.rcParams.update({
    "figure.dpi": 300,
    "savefig.dpi": 600,
    "font.family": "serif",
    "font.size": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

def add_span(ax, x0, x1, y0, h, fc, ec, alpha=1.0):
    rect = Rectangle((x0, y0), x1 - x0, h, facecolor=fc,
                     edgecolor=ec, linewidth=1.0, alpha=alpha)
    ax.add_patch(rect)

def add_len_bracket(ax, x0, x1, y, text, dy=-0.5):
    """Draw a length bracket with label slightly below (dy)."""
    ax.annotate("", xy=(x0, y), xytext=(x1, y),
                arrowprops=dict(arrowstyle="<->", linewidth=0.8))
    ax.text((x0 + x1)/2, y + dy, text, ha="center", va="top")

def main():
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.2, 3.0))

    ax.set_xlim(-2, AX_MAX)     # give extra left margin
    ax.set_ylim(-4, 12)
    ax.set_yticks([])
    ax.set_xlabel("Time index (t)")
    ax.set_title("Windowed encoder–decoder construction")

    y_w0, y_w1, h = 3.0, 7.0, 1.4
    c_enc, c_lab, c_pred = "#6baed6", "#2171b5", "#fa9fb5"

    # Window 1 (bottom)
    add_span(ax, 0, L_ENC, y_w0, h, c_enc, "black")
    add_span(ax, L_ENC - L_LAB, L_ENC, y_w0, h, c_lab, "black")
    add_span(ax, L_ENC, L_ENC + L_PRED, y_w0, h, c_pred, "black")
    ax.text(-3.5, y_w0 + h/2, "Window 1", va="center", ha="right")

    # Brackets (staggered to avoid overlap)
    add_len_bracket(ax, 0, L_ENC, y_w0 - 1.0, r"$L_{\mathrm{enc}}$")
    add_len_bracket(ax, L_ENC, L_ENC + L_PRED, y_w0 - 2.0, r"$L_{\mathrm{pred}}$")
    add_len_bracket(ax, L_ENC - L_LAB, L_ENC, y_w0 - 3.0, r"$L_{\mathrm{lab}}$", dy=-0.8)

    # Window 2 (top, faded)
    alpha_g = 0.35
    add_span(ax, W1_START, W1_START + L_ENC, y_w1, h, c_enc, "black", alpha=alpha_g)
    add_span(ax, W1_START + L_ENC - L_LAB, W1_START + L_ENC, y_w1, h, c_lab, "black", alpha=alpha_g)
    add_span(ax, W1_START + L_ENC, W1_START + L_ENC + L_PRED, y_w1, h, c_pred, "black", alpha=alpha_g)
    ax.text(-3.5, y_w1 + h/2, "Window 2", va="center", ha="right")

    # Stride arrow
    y_stride = (y_w0 + y_w1) / 2
    ax.annotate("", xy=(W0_START, y_stride), xytext=(W1_START, y_stride),
                arrowprops=dict(arrowstyle="<->", linewidth=0.8))
    ax.text((W0_START + W1_START)/2, y_stride + 0.8, r"stride $s$", ha="center", va="bottom")

    # Legend
    legend_elems = [
        Rectangle((0,0),1,1, facecolor=c_enc,  edgecolor="black", label="Encoder context"),
        Rectangle((0,0),1,1, facecolor=c_lab,  edgecolor="black", label="Decoder history (labels)"),
        Rectangle((0,0),1,1, facecolor=c_pred, edgecolor="black", label="Prediction horizon"),
    ]
    ax.legend(handles=legend_elems, loc="upper right", frameon=False, ncol=2)

    fig.tight_layout()
    plt.subplots_adjust(left=0.12)  # extra space for y-axis labels
    fig.savefig(OUT_PATH, bbox_inches="tight", format="png")
    plt.close(fig)
    print(f"Saved Figure 3.4 → {OUT_PATH}")


if __name__ == "__main__":
    main()
