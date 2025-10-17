import warnings
from matplotlib.collections import LineCollection
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


def colored_line(x, y, c, ax, **lc_kwargs):
    if "array" in lc_kwargs:
        warnings.warn('The provided "array" keyword argument will be overridden')

    # Default the capstyle to butt so that the line segments smoothly line up
    default_kwargs = {"capstyle": "butt"}
    default_kwargs.update(lc_kwargs)

    # Compute the midpoints of the line segments. Include the first and last points
    # twice so we don't need any special syntax later to handle them.
    x = np.asarray(x)
    y = np.asarray(y)
    x_midpts = np.hstack((x[0], 0.5 * (x[1:] + x[:-1]), x[-1]))
    y_midpts = np.hstack((y[0], 0.5 * (y[1:] + y[:-1]), y[-1]))

    coord_start = np.column_stack((x_midpts[:-1], y_midpts[:-1]))[:, np.newaxis, :]
    coord_mid = np.column_stack((x, y))[:, np.newaxis, :]
    coord_end = np.column_stack((x_midpts[1:], y_midpts[1:]))[:, np.newaxis, :]
    segments = np.concatenate((coord_start, coord_mid, coord_end), axis=1)

    lc = LineCollection(segments, **default_kwargs)
    lc.set_array(c)  # set the colors of each segment

    return ax.add_collection(lc)


def taylor_expansion_plot(
    n_terms=5,  # Number of Taylor series terms to display
    x_start=-10,
    x_end=10,
    num_points=1000,
    line_thickness=2,
    bg_color="white",
    show_actual=True,
    y_lim=[5, 5],
    a_b=[-0.5, 10],
):
    """
    Plot Taylor series expansion of sine wave using different numbers of terms.
    """
    from scipy.special import factorial

    # Pink, turquoise, lime green, beige colors
    water_colors = np.flip(
        [
            "#E7d4BF",
            "#ebe7cd",
            "#d4f1ef",
            "#b6e3e4",
            "#a2d2dc",
            "#91c0d4",
        ]
    )
    colors = water_colors

    x = np.linspace(x_start, x_end, num_points)

    # Calculate the actual sine function for reference
    linear_line = a_b[0] * x + a_b[1]
    y_actual = np.sin(x)

    # Create the figure
    fig, ax = plt.subplots(figsize=landscape_dimensions)
    ax.axis("off")

    # Create a color map for the approximations
    cmap = create_multi_cmap(colors, n_terms)

    # Plot Taylor series approximations with increasing terms
    for i in range(2, n_terms + 1):
        y_approx = np.zeros_like(x)
        # Calculate the Taylor series up to i terms
        for n in range(i):
            y_approx += ((-1) ** n * x ** (2 * n + 1)) / factorial(2 * n + 1)

            # y_approx += ((-1)**n * x**(2*n+1)) / np.math.factorial(2*n+1)

        ax.plot(
            x,
            y_approx + linear_line,
            linewidth=line_thickness,
            color=cmap[i - 1],
            label=f"Taylor series ({i} terms)",
        )

        # Plot the actual sine function if requested
    if show_actual:
        halfway_point = int(len(x) / 2)
        cmap = create_multi_cmap(colors, halfway_point)

        left_line = colored_line(
            x=np.flip(x[: halfway_point + 1]),
            y=np.flip(y_actual[: halfway_point + 1] + linear_line[: halfway_point + 1]),
            c=np.linspace(0, 2, halfway_point),
            ax=ax,
            colors=cmap,
            linewidth=line_thickness,
        )

        right_line = colored_line(
            x=x[halfway_point:],
            y=y_actual[halfway_point:] + linear_line[halfway_point:],
            c=np.linspace(0, 2, halfway_point),
            ax=ax,
            colors=cmap,
            linewidth=line_thickness,
        )

    ax.set_facecolor(bg_color)
    ax.set_ylim(y_lim[0], y_lim[1])
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)  # Remove all padding/margins
    plt.margins(0)  # Remove axis margins
    plt.tight_layout(pad=0)

    plt.savefig("outputs/jil_plot.png", transparent=True, dpi=300)

    plt.show()

    taylor_expansion_plot()
