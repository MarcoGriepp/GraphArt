"""Victoria's butterfly plot - parametric butterfly curve visualization."""

import numpy as np
import matplotlib.pyplot as plt
from src.utils import create_multi_cmap, LANDSCAPE_DIMENSIONS, victoria_colors


def butterfly_plot(
    t_len,
    color_len,
    phi_step,
    step=0.01,
    line_thickness=1,
):
    t = np.arange(stop=t_len, step=step)

    cmap = create_multi_cmap(colors=list(victoria_colors.values()), length=color_len)
    fig, ax = plt.subplots(1, 1, figsize=LANDSCAPE_DIMENSIONS, dpi=300)
    ax.axis("off")

    for color in range(color_len):
        x = np.sin(t) * (np.exp(np.cos(t)) - 2 * np.cos(4 * t) - np.sin(t / 12) ** 5)
        y = np.cos(t) * (np.exp(np.cos(t)) - 2 * np.cos(4 * t) - np.sin(t / 12) ** 5)
        ax.plot(
            x * (color + color**phi_step),
            y * (color + color**phi_step),
            color=cmap[color],
            linewidth=line_thickness,
        )

    plt.margins(0.3)  # Remove axis margins
    plt.tight_layout(pad=0)
    # ax.set_facecolor("#F8F6F9")

    plt.savefig(
        fname="outputs/victoria_plot",
        transparent=True,
    )

    plt.show()


if __name__ == "__main__":
    # Example usage
    butterfly_plot(
        t_len=int(12*np.pi),
        step=0.001,
        color_len=30,
        phi_step=0,
        line_thickness=2.5
    )
