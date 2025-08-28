# simple_patchtst_diagram.py
# Minimal, clean PatchTST schematic (PNG + SVG)
# pip install matplotlib

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrow

def box(ax, x, y, w, h, text, fc="#F5F5F5", ec="#111111", fontsize=10, weight="regular"):
    ax.add_patch(Rectangle((x, y), w, h, facecolor=fc, edgecolor=ec, linewidth=1.5))
    ax.text(x + w/2, y + h/2, text, ha="center", va="center", fontsize=fontsize, color="#111111", weight=weight)

def arrow(ax, x1, y1, x2, y2):
    ax.add_patch(FancyArrow(x1, y1, x2 - x1, y2 - y1, width=0.0, head_width=2.2, head_length=6, length_includes_head=True, color="#111111", linewidth=1.8))

def main(out_png="patchtst.png", out_svg="patchtst.svg"):
    plt.rcParams.update({"figure.dpi": 200, "font.family": "DejaVu Sans", "font.size": 10})
    fig, ax = plt.subplots(figsize=(10, 2.8))
    ax.set_xlim(0, 300)
    ax.set_ylim(0, 60)
    ax.axis("off")

    # layout
    y = 20; H = 22
    W_data = 70; W = 58; gap = 8

    x0 = 6
    x1 = x0 + W_data + gap
    x2 = x1 + W + gap
    x3 = x2 + W + gap
    x4 = x3 + W + gap
    x5 = x4 + W + gap
    x6 = x5 + W + gap

    # boxes
    box(ax, x0, y, W_data, H,
        "Exogenous seq\n[E1..E6, θ, LP1..LP4]\n11 ch, T=200", fc="#EAF2FF")
    box(ax, x1, y, W, H, "Patchify\nlength p → T/p tokens")
    box(ax, x2, y, W, H, "InstanceNorm")
    box(ax, x3, y, W, H, "Patch Embedding")
    box(ax, x4, y, W, H, "Transformer\nEncoder (L)")
    box(ax, x5, y, W, H, "Linear Head\n(c_out=6)")
    box(ax, x6, y, W_data, H, "Predicted stress\n[S1..S6]", fc="#EAF2FF")

    # arrows (center-to-center on box edges)
    arrow(ax, x0 + W_data, y + H/2, x1, y + H/2)
    arrow(ax, x1 + W,      y + H/2, x2, y + H/2)
    arrow(ax, x2 + W,      y + H/2, x3, y + H/2)
    arrow(ax, x3 + W,      y + H/2, x4, y + H/2)
    arrow(ax, x4 + W,      y + H/2, x5, y + H/2)
    arrow(ax, x5 + W,      y + H/2, x6, y + H/2)

    # tiny notes
    ax.text(x0 + W_data/2, y + H + 8, "11 channels", ha="center", va="center", color="#444444")
    ax.text(x6 + W_data/2, y - 8,     "6 channels",  ha="center", va="center", color="#444444")

    fig.tight_layout(pad=0.2)
    fig.savefig(out_png, bbox_inches="tight", dpi=300)
    fig.savefig(out_svg, bbox_inches="tight")
    print(f"Saved: {out_png}, {out_svg}")

if __name__ == "__main__":
    main()
