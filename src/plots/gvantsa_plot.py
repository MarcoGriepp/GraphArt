"""Gvantsa's warp plot - artistic visualization with warped sine waves."""

import numpy as np
import matplotlib.pyplot as plt
from src.utils import create_multi_cmap, LANDSCAPE_DIMENSIONS, gvantsa_colors


def warp_plot(num_lines, x_start, x_end, **kwargs):
    # Set up canvas with specified size and DPI and green background
    fig, ax = plt.subplots(figsize=LANDSCAPE_DIMENSIONS, dpi=300, facecolor="green")
    ax.set_facecolor("green")
    ax.axis("off")

    # Parameters
    x = np.linspace(x_start, x_end, 1000)

    cmap = create_multi_cmap(colors=list(gvantsa_colors.values()), length=num_lines)

    for i in range(num_lines):
        stretch = 1 + 0.5 * np.sin(i * 3)
        y = (
            np.tanh(x * 0.5) * (i / 10)
            + np.sin(x * stretch) * 0.1 * stretch
            + np.cos(x)
            + np.arctan(x * 2)
        )

        ax.plot(x, y, color=cmap[i], **kwargs)
        ax.plot(x, np.flip(y), color=cmap[i], **kwargs)
        ax.plot(np.flip(x), np.flip(y), color=cmap[i], **kwargs)
        ax.plot(y, x, color=cmap[i], **kwargs)
        ax.plot(y, np.flip(x), color=cmap[i], **kwargs)
        ax.plot(np.flip(y), np.flip(x), color=cmap[i], **kwargs)

    fig.patch.set_facecolor("#1E1B2E")
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)  # Remove all padding/margins
    plt.margins(0)  # Remove axis margins
    plt.tight_layout(pad=-1)
    plt.savefig("outputs/gvantsa_plot.png", transparent=True)

    plt.show()


if __name__ == "__main__":
    # Example usage
    warp_plot(num_lines=20, x_start=-8, x_end=8, linewidth=17.3)
