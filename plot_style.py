import os
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

def set_oa_style():
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "#E6E6E6",
        "axes.labelcolor": "#444444",
        "axes.titlecolor": "#2E2E2E",
        "xtick.color": "#555555",
        "ytick.color": "#555555",
        "grid.color": "#DDDDDD",
        "grid.linestyle": "-",
        "grid.linewidth": 0.8,
        "axes.grid": True,
        "axes.spines.top": False,
        # "axes.spines.right": False,
        "savefig.bbox": "tight",
        "savefig.dpi": 300,
        "font.size": 10,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
    })

def plot_metrics(psnr_hist, loss_hist, epoch, out_dir, total_epochs, psnr_ymax=40.0, loss_ymax=0.5):
    psnr_color = "#10A37F"
    psnr_fill  = "#DDF5EC"
    loss_color = "#A88AE8"
    dot_edge   = "white"

    psnr_xs = [e for e, _ in psnr_hist]
    psnr_ys = [p for _, p in psnr_hist]
    loss_xs = [e for e, _ in loss_hist]
    loss_ys = [l for _, l in loss_hist]

    fig, ax1 = plt.subplots(figsize=(8.0, 4.6), dpi=300)
    ax2 = ax1.twinx()

    ax1.set_xlim(0, total_epochs)
    ax1.set_ylim(0, psnr_ymax)
    ax2.set_ylim(0, loss_ymax)

    ax1.grid(True, alpha=0.28)
    ax2.grid(False)
    for sp in ["top"]:
        ax1.spines[sp].set_visible(False)
        ax2.spines[sp].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax2.spines["right"].set_visible(True)
    ax2.spines["right"].set_color("#E6E6E6")

    ax1.set_title("Optimization Metrics", pad=10, weight="semibold")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("PSNR (dB)")
    ax2.set_ylabel("Loss")

    ax1.tick_params(axis="both", labelsize=9, colors="#444444")
    ax2.tick_params(axis="y", labelsize=9, colors="#444444")

    # === PSNR 主线 + 填充 + 当前圆点 ===
    if len(psnr_xs) > 1:
        ax1.plot(psnr_xs, psnr_ys, color=psnr_color, linewidth=2.6, solid_capstyle="round")
        ax1.fill_between(psnr_xs, psnr_ys, color=psnr_fill, alpha=0.45)
    elif len(psnr_xs) == 1:
        ax1.scatter(psnr_xs, psnr_ys, color=psnr_color, s=36, zorder=3, linewidths=0.8, edgecolors=dot_edge)

    if len(psnr_xs) > 0:
        ax1.scatter(psnr_xs[-1], psnr_ys[-1],
                    color=psnr_color, s=64, zorder=4, linewidths=1.0, edgecolors=dot_edge)

    # === Loss 虚线（更密、更细、更淡） + 当前圆点 ===
    if len(loss_xs) > 1:
        ax2.plot(
            loss_xs, loss_ys,
            color=loss_color,
            linewidth=1.6,
            linestyle=(0, (2.4, 2.4)),
            alpha=0.7,
            solid_capstyle="round",
        )
    elif len(loss_xs) == 1:
        ax2.scatter(loss_xs, loss_ys, color=loss_color, s=30, zorder=3, linewidths=0.8, edgecolors=dot_edge)

    if len(loss_xs) > 0:
        ax2.scatter(loss_xs[-1], loss_ys[-1],
                    color=loss_color, s=54, zorder=4, linewidths=0.9, edgecolors=dot_edge)

    # === 图例：左上角 ===
    handles = [
        Line2D([0], [0], color=psnr_color, lw=2.4, label="PSNR"),
        Line2D([0], [0], color=loss_color, lw=1.2, linestyle=(0, (2, 2)), label="Loss")
    ]
    ax1.legend(handles=handles, frameon=False, loc="upper right", fontsize=10)

    plt.tight_layout()
    out_path = os.path.join(out_dir, f"metrics_epoch_{epoch:04d}.png")
    plt.savefig(out_path, facecolor="white")
    plt.close()