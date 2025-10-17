"""Taro's noisewave plot - symmetric wave patterns with noise."""

import numpy as np
import matplotlib.pyplot as plt
from src.utils import create_multi_cmap, LANDSCAPE_DIMENSIONS


def noisewave_plot(
    n_lines,
    dist,
    decay_rate,
    sigma_initialisation,
    x_length,
    step_size,
    line_thickness,
    sigma_error,
    one_sided=False,
):
    index = np.arange(start=0, step=step_size, stop=x_length)

    x_right = index**decay_rate

    phi_adjustment = 2 if one_sided else 1

    if dist == "normal":
        phi_samples = (
            np.random.normal(loc=0, scale=sigma_initialisation, size=n_lines)
            ** phi_adjustment
        )
    elif dist == "chisquare":
        phi_samples = np.random.chisquare(df=5, size=n_lines) ** phi_adjustment
    elif dist == "uniform":
        phi_samples = np.random.uniform(low=0, high=5, size=n_lines) ** phi_adjustment

    base_line = np.cos(index)

    fig, axs = plt.subplots(nrows=1, ncols=1, figsize=LANDSCAPE_DIMENSIONS, dpi=300)
    axs.axis("off")

    gradient_1 = create_multi_cmap(
        ["#6F6092", "#01796F", "#93DFB8", "#4F7942", "#00587A"], n_lines
    )

    for i, phi in enumerate(np.sort(phi_samples)):
        y = base_line * phi
        x_line_right = x_right
        x_line_left = -x_right

        error = np.zeros(len(x_line_left))

        for t in range(len(x_line_left) - 1, 0, -1):
            error[t] = np.random.normal(
                loc=0, scale=(np.abs(x_line_left[t]) ** 1) * sigma_error, size=1
            )

        # Plot lines
        x = np.append(np.flip(x_line_left), x_line_right)
        y = np.append(np.flip(y + error), y)

        axs.plot(x, y, lw=line_thickness, color=gradient_1[i])
        axs.get_yaxis().set_visible(False)
        axs.get_xaxis().set_visible(False)
        axs.set_facecolor("#000000")
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)  # Remove all padding/margins
    plt.margins(0.02)  # Remove axis margins
    # plt.axis('image')
    plt.savefig("outputs/taro_plot.png", transparent=True)
    plt.show()


if __name__ == "__main__":
    # Example usage
    noisewave_plot(
        n_lines=40,
        dist="uniform",
        decay_rate=1.8,
        sigma_initialisation=5,
        x_length=20,
        step_size=0.1,
        one_sided=True,
        line_thickness=4,
        sigma_error=0.05,
    )
