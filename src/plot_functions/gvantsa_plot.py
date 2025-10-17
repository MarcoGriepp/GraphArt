import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Define dimensions if not imported
landscape_dimensions = (4961 / 300, 3508 / 300)


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


gvantsa_colors = {
    "Yellow": "#FFFF00",
    "Raspberry Pink": "#E30B5C",
    "Lilac Purple": "#C8A2C8",
}


def warp_plot(num_lines, x_start, x_end, **kwargs):
    # Set up canvas with specified size and DPI and green background
    fig, ax = plt.subplots(figsize=landscape_dimensions, dpi=300, facecolor="green")
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


warp_plot()
