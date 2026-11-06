import argparse

import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle

from rust_qldpc import get_window_borders

from util import build_bb_code, build_circuit_and_matrices


def paint_windows(axis, H, window_borders):
    axis.spy(H, markersize=0.5, color="black", aspect="auto")

    colors = plt.get_cmap("tab10")
    legend_items = []
    for index, ((row_begin, col_begin), (row_end, col_end)) in enumerate(
        window_borders, start=1
    ):
        color = colors((index - 1) % colors.N)
        rectangle = Rectangle(
            (col_begin - 0.5, row_begin - 0.5),
            col_end - col_begin + 1,
            row_end - row_begin + 1,
            facecolor=color,
            edgecolor=color,
            alpha=0.16,
            linewidth=1.5,
        )
        axis.add_patch(rectangle)
        legend_items.append(Patch(facecolor=color, alpha=0.4, label=f"Window {index}"))

    axis.set_xlim(-0.5, H.shape[1] - 0.5)
    axis.set_ylim(H.shape[0] - 0.5, -0.5)
    axis.set_xlabel("PCM column")
    axis.set_ylabel("PCM row")
    axis.legend(handles=legend_items, loc="upper right", fontsize="small")


def plot_windows(H, forward_borders, reverse_borders, W, F):
    figure, axes = plt.subplots(1, 2, figsize=(24, 7), sharex=True, sharey=True)
    paint_windows(axes[0], H, forward_borders)
    paint_windows(axes[1], H, list(reversed(reverse_borders)))
    axes[0].set_title("Forward: top to bottom")
    axes[1].set_title("Backward: bottom to top")
    figure.suptitle(f"Detector-error PCM and sliding windows (W={W}, F={F})")
    figure.tight_layout()
    return figure


def main():
    parser = argparse.ArgumentParser(
        description="Plot the BB-72 detector-error PCM and its Rust-generated windows."
    )
    parser.add_argument("--W", type=int, default=3, help="Window size in rounds")
    parser.add_argument("--F", type=int, default=1, help="Window shift in rounds")
    parser.add_argument("--num-rounds", type=int, default=6)
    args = parser.parse_args()

    code, _, H, _, _ = build_circuit_and_matrices(
        0.001, args.num_rounds, build_bb_code(72)
    )
    forward_borders = get_window_borders(
        H, code.hz.shape[0], args.num_rounds, args.W, args.F
    )
    reverse_borders = get_window_borders(
        H, code.hz.shape[0], args.num_rounds, args.W, args.F, True
    )
    plot_windows(H, forward_borders, reverse_borders, args.W, args.F)
    plt.show()


if __name__ == "__main__":
    main()
