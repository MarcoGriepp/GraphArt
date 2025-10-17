"""Jil's Taylor expansion plot - visualizing Taylor series approximations."""

import warnings
from matplotlib.collections import LineCollection
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import factorial
from src.utils import create_multi_cmap, colored_line, LANDSCAPE_DIMENSIONS


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
    fig, ax = plt.subplots(figsize=LANDSCAPE_DIMENSIONS)
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


if __name__ == "__main__":
    # Example usage
    taylor_expansion_plot(
        n_terms=30,
        x_start=-25,
        x_end=25,
        bg_color="#E7D7BE",
        line_thickness=20,
        show_actual=True,
        a_b=[-0.5, 10],
        y_lim=[-5, 25]
    )
