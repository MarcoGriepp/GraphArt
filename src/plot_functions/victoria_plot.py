import warnings
from matplotlib.collections import LineCollection
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Define dimensions if not imported
landscape_dimensions = (4961 / 300, 3508 / 300)


victoria_colors = {
    "Soft Pink": "#F4C2C2",
    "Mauve": "#E0B0FF",
    "Lilac": "#C8A2C8",
    "Matcha Green": "#A0C25A",
    "Amethyst": "#9966CC",
    "Plum": "#8E4585",
    "Cobalt Blue": "#0047AB",
}


def create_multi_cmap(colors_list, length):
    """
    Creates a colormap from a list of colors and samples 'length' colors from it.
    """
    if not colors_list:  # Handle empty colors_list
        return ["#000000"] * length  # Default to black
    if length == 0:
        return []
    cmap = mcolors.LinearSegmentedColormap.from_list("custom_multi_cmap", colors_list)
    gradient = [
        mcolors.to_hex(cmap(i / (length - 1)))
        if length > 1
        else mcolors.to_hex(cmap(0.0))
        for i in range(length)
    ]
    return gradient


def butterfly_plot(
    t_len,
    color_len,
    phi_step,
    step=0.01,
    line_thickness=1,
):
    t = np.arange(stop=t_len, step=step)

    cmap = create_multi_cmap(colors=list(victoria_colors.values()), length=color_len)
    fig, ax = plt.subplots(1, 1, figsize=landscape_dimensions, dpi=300)
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


butterfly_plot()
