import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Define dimensions if not imported
portrait_dimensions = (3508 / 300, 4961 / 300)


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


def step_sine_plot(
    n_lines,
    dist,
    decay_rate,
    sigma_initialisation,
    x_length,
    x_start,
    step_size,
    line_thickness,
    sigma_error,
    block_step_size,
    chance_0,
    phi_coef=0.8,
    midsection_proportion=0,
    one_sided=False,
    vertical=False,
):
    index = np.arange(start=0, step=step_size, stop=x_length)

    index_n = len(index)
    midsection_n = int(index_n * midsection_proportion)
    right_n = index_n - midsection_n

    # Implement and slice the index for the midsection
    x_right = index**decay_rate
    x_midsection_right = x_right[: -right_n + 1]
    x_right = x_right[-right_n:]
    x_midsection = np.append(-np.flip(x_midsection_right), x_midsection_right)

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

    base_line = np.cos(
        np.arange(
            start=x_start,
            step=step_size,
            stop=x_length + x_start,
        )
    )[:-midsection_n]

    fig, axs = plt.subplots(nrows=1, ncols=1, figsize=portrait_dimensions)
    axs.axis("off")

    gradient_1 = create_multi_cmap(
        ["#1E3A6A", "#A3B8ED", "#D1BDA4", "#ECE3DA"], n_lines
    )
    max_y = 0  # initialize

    for i, phi in enumerate(np.sort(phi_samples)):
        # Right side
        y_right = base_line * phi
        x_line_right = x_right

        # Left side
        y_left = np.full(2, 0)

        x_line_left = -x_right
        x_line_left_new = 0
        walk_step_old = 1

        for step in range(
            1,
            len(x_line_left),
        ):
            walk_step_new = np.random.choice(
                a=(block_step_size, 0, -block_step_size),
                size=1,
                p=((1 - chance_0) / 2, chance_0, (1 - chance_0) / 2),
            )

            if walk_step_new != 0 and walk_step_old == 0:
                # first point:
                y_left = np.append(y_left, y_left[-1])
                x_line_left_new = np.append(x_line_left_new, x_line_left[step])

                # second point:
                y_left = np.append(y_left, y_left[-1] + walk_step_new)
                x_line_left_new = np.append(x_line_left_new, x_line_left[step])

            else:
                y_left = np.append(y_left, y_left[-1])
                x_line_left_new = np.append(x_line_left_new, x_line_left[step])

            walk_step_old = walk_step_new

        # midsection:
        midsection_len = int(len(x_line_left_new) * 2 * midsection_proportion)
        y_midsection_left = np.zeros(len(x_midsection_right))
        y_midsection_right = np.zeros(len(x_midsection_right))
        y_midsection_right[len(x_midsection_right) - 1] = y_right[
            0
        ]  # Initialise connection point
        for step in range(len(x_midsection_right) - 2, -1, -1):
            y_midsection_right[step] = phi_coef * y_midsection_right[step + 1]

        y_midsection = np.append(y_midsection_left, y_midsection_right)

        max_y = max(
            max_y,
            np.max(np.abs(y_left)),
            np.max(np.abs(y_midsection)),
            np.max(np.abs(y_right)),
        )

        # Plot lines
        if vertical:
            axs.plot(y_right, x_line_right, lw=line_thickness, color=gradient_1[i])
            if i % 3 == 0:  # Check if i is divisible by 3
                axs.plot(
                    y_left[1:], x_line_left_new, lw=line_thickness, color=gradient_1[i]
                )
            axs.plot(y_midsection, x_midsection, lw=line_thickness, color=gradient_1[i])
        else:
            axs.plot(x_line_right, y_right, lw=line_thickness, color=gradient_1[i])
            axs.plot(
                x_line_left_new, y_left[1:], lw=line_thickness, color=gradient_1[i]
            )

        axs.set_facecolor("black")
        axs.set_xlim(-max_y * 1.2, max_y * 1.2)

    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)  # Remove all padding/margins
    plt.margins(0.1)  # Remove axis margins
    plt.tight_layout(pad=0)
    plt.savefig("outputs/victor_plot.png", transparent=True, dpi=300)

    plt.show()


step_sine_plot()
