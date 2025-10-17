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

    fig, axs = plt.subplots(nrows=1, ncols=1, figsize=landscape_dimensions, dpi=300)
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


noisewave_plot()
