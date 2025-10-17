"""Utility functions for GraphArt plotting."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.collections import LineCollection
from scipy.optimize import minimize, Bounds
from scipy.special import factorial
import subprocess
import sys
import os
import warnings
from typing import (
    List,
    Dict,
    Tuple,
    Set,
    Optional,
    Union,
    Any,
    Callable,
    Iterator,
    Sequence,
)

try:
    import imageio
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "imageio"])
    import imageio

warnings.simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

# Canvas dimensions for consistent output
LANDSCAPE_DIMENSIONS = (4961 / 300, 3508 / 300)
PORTRAIT_DIMENSIONS = (3508 / 300, 4961 / 300)


def create_cmap(color_1: str, color_2: str, length: int) -> list:
    """
    Create a color gradient between two colors.
    
    Args:
        color_1: Starting color (hex format)
        color_2: Ending color (hex format)
        length: Number of colors in the gradient
        
    Returns:
        List of hex color codes
    """
    # Convert hex colors to RGB tuples (0-1 scale)
    rgb1 = np.array(mcolors.to_rgb(color_1))
    rgb2 = np.array(mcolors.to_rgb(color_2))

    # Generate the gradient
    gradient = [
        mcolors.to_hex(rgb1 * (1 - t) + rgb2 * t) for t in np.linspace(0, 1, length)
    ]

    return gradient


def create_multi_cmap(colors: list, length: int) -> list:
    """
    Create a colormap from a list of colors and sample 'length' colors from it.
    
    Args:
        colors: List of color hex codes
        length: Number of colors to sample from the gradient
        
    Returns:
        List of sampled hex color codes
    """
    if not colors:  # Handle empty colors list
        return ["#000000"] * length  # Default to black
    if length == 0:
        return []
    
    cmap = mcolors.LinearSegmentedColormap.from_list("custom", colors)
    gradient = [
        mcolors.to_hex(cmap(i / (length - 1))) if length > 1 else mcolors.to_hex(cmap(0.0))
        for i in range(length)
    ]
    return gradient


def show_colors(colors: Dict, title):
    color_names, color_hexcodes = colors.keys(), colors.values()

    fig = plt.figure(figsize=(12, 4))
    plt.bar(color_names, np.ones(len(color_names)), color=color_hexcodes)
    plt.title(title)

    plt.show()


def plot_1(n_lines, index_length, phi, sigma, return_percent_point):
    return_point = return_percent_point / 100 * index_length - 1
    print(return_point)

    # Initialize and sort starting points
    starting_points = np.sort(
        np.random.normal(
            loc=0,
            scale=sigma,
            size=n_lines,  # Fixed: changed from size=1 to size=n_lines
        )
    )

    fig = plt.figure(figsize=(18, 10), dpi=150)

    # Create color gradients
    for i, point in enumerate(starting_points):
        line_data = np.zeros(index_length)
        line_data[0] = point

        for t in range(1, index_length):
            if t <= return_point:
                line_data[t] = phi * line_data[t - 1]
            else:
                line_data[t] = (1 / phi) * line_data[t - 1]

        # Calculate RGB values based on line index
        r = i / (n_lines - 1)
        g = 0.7
        b = 1 - (i / (n_lines - 1))

        plt.plot(line_data, color=(r, g, b))

    plt.show()


def plot_2(
    n_lines,
    index_length,
    sigma,
    step,
    r_range=[0, 1],
    g_range=[0, 1],
    b_range=[0, 1],
    plot_vertical=False,
    slope=0,
    sigma_error=0,
    sigma_dispersion=0,
):
    index = np.arange(stop=index_length, step=step)

    def calc_rgb_value(range, i, n_lines):
        range_start = range[0]
        range_end = range[1]
        range_area = range_end - range_start
        return range_start + (i / n_lines * range_area)

    def decreasing_range(start, step, length=100):
        return np.arange(start, start - step * length, -step)

    base_line = np.cos(index)
    print(len(index))
    phi_samples = np.sort(np.random.normal(0, sigma, n_lines))

    actual_index_length = len(index)

    fig = plt.figure(figsize=(18, 10), dpi=150)

    # Create color gradients
    for i, point in enumerate(phi_samples):
        error = np.random.normal(0, sigma_error, actual_index_length)
        dispersion = np.random.normal(0, sigma_dispersion, 1)
        line_data = base_line * phi_samples[i] + error + dispersion

        # Calculate RGB values based on line index
        r = calc_rgb_value(r_range, i, n_lines)
        # print(f"i: {i}")
        # print(f"r: {r}")
        # print()
        g = calc_rgb_value(g_range, i, n_lines)
        b = calc_rgb_value(b_range, i, n_lines)

        if plot_vertical:
            plt.plot(
                line_data + 0.5 * actual_index_length,
                range(actual_index_length),
                color=(r, g, b),
            )

        if slope:
            slope_addition = decreasing_range(0, slope, actual_index_length)
            plt.plot(line_data + slope_addition, color=(r, g, b))
        else:
            plt.plot(line_data, color=(r, g, b))  # x, y

    plt.show()


def plot_3_test(
    n_lines,
    index_length,
    sigma,
    step,
    r_range=[0, 1],
    g_range=[0, 1],
    b_range=[0, 1],
    plot_vertical=False,
    slope=0,
    sigma_error=0,
    sigma_dispersion=0,
    power_range=[0, 1],
    num_frames=100,
):
    index = np.arange(stop=index_length, step=step)

    def calc_rgb_value(range, i, n_lines):
        range_start = range[0]
        range_end = range[1]
        range_area = range_end - range_start
        return range_start + (i / n_lines * range_area)

    def decreasing_range(start, step, length=100):
        return np.arange(start, start - step * length, -step)

    base_line = np.cos(index)
    print(len(index))
    phi_samples = np.sort(np.random.normal(0, sigma, n_lines))

    actual_index_length = len(index)

    fig = plt.figure(figsize=(18, 10), dpi=150)
    power_step = (power_range[1] - power_range[0]) / num_frames
    filenames = []

    # Create color gradients
    for f in range(num_frames):
        plt.figure(figsize=(4, 3), dpi=50)  # Smaller figure size
        power = power_range[0] + power_step * f

        for i, point in enumerate(phi_samples):
            error = np.random.normal(0, sigma_error, actual_index_length)
            dispersion = np.random.normal(0, sigma_dispersion, 1)
            line_data = base_line * phi_samples[i] + error + dispersion

            # Calculate RGB values based on line index
            r = calc_rgb_value(r_range, i, n_lines)
            # print(f"i: {i}")
            # print(f"r: {r}")
            # print()
            g = calc_rgb_value(g_range, i, n_lines)
            b = calc_rgb_value(b_range, i, n_lines)

            if plot_vertical:
                plt.plot(
                    line_data + 0.5 * actual_index_length,
                    range(actual_index_length),
                    color=(r, g, b),
                )

            if slope:
                slope_addition = decreasing_range(0, slope, actual_index_length)
                plt.plot(line_data + slope_addition, color=(r, g, b))
            else:
                plt.plot(
                    line_data, np.sort(line_data) ** power, color=(r, g, b)
                )  # x, y

        filename = f"animation/frame_{f}.png"
        plt.savefig(filename)
        filenames.append(filename)
        plt.close()

        # Create GIF
        gif_filename = "animation.gif"
        with imageio.get_writer(gif_filename, mode="I", duration=5) as writer:
            for filename in filenames:
                image = imageio.imread(filename)
                writer.append_data(image)

    plt.show()


def plot_heart(
    points,
):
    # Make uneven points even
    while (points % 4) != 0:  # must be divisible by 4
        points += 1

    print(points)

    def create_left_x(points):
        # First half:
        x = np.zeros(1)  # set first value to 0
        step_size = 2  # initialize step size
        step_size_factor = 1 - (1 / points * 10)
        x_value = 0
        print((points / 2) - 1)

        for step in range(int(points / 2 - 1)):
            x_value += step_size
            x = np.append(x, x_value)
            step_size *= step_size_factor

        # append second half
        return np.append(x, np.flip(x))

    plt.plot(create_left_x(points))
    plt.show()
    print(create_left_x(points))

    # Third quarter:


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


dist_functions_mapping = {
    "normal": np.random.normal,
    "uniform": np.random.uniform,
    "chisquare": np.random.chisquare,
}


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

    fig, axs = plt.subplots(nrows=1, ncols=1, figsize=PORTRAIT_DIMENSIONS)
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


def colored_line(x, y, c, ax, **lc_kwargs):
    """
    Plot a line with a color specified along the line by a third value.

    It does this by creating a collection of line segments. Each line segment is
    made up of two straight lines each connecting the current (x, y) point to the
    midpoints of the lines connecting the current point with its two neighbors.
    This creates a smooth line with no gaps between the line segments.

    Parameters
    ----------
    x, y : array-like
        The horizontal and vertical coordinates of the data points.
    c : array-like
        The color values, which should be the same size as x and y.
    ax : Axes
        Axis object on which to plot the colored line.
    **lc_kwargs
        Any additional arguments to pass to matplotlib.collections.LineCollection
        constructor. This should not include the array keyword argument because
        that is set to the color argument. If provided, it will be overridden.

    Returns
    -------
    matplotlib.collections.LineCollection
        The generated line collection representing the colored line.
    """
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

    # Determine the start, middle, and end coordinate pair of each line segment.
    # Use the reshape to add an extra dimension so each pair of points is in its
    # own list. Then concatenate them to create:
    # [
    #   [(x1_start, y1_start), (x1_mid, y1_mid), (x1_end, y1_end)],
    #   [(x2_start, y2_start), (x2_mid, y2_mid), (x2_end, y2_end)],
    #   ...
    # ]
    coord_start = np.column_stack((x_midpts[:-1], y_midpts[:-1]))[:, np.newaxis, :]
    coord_mid = np.column_stack((x, y))[:, np.newaxis, :]
    coord_end = np.column_stack((x_midpts[1:], y_midpts[1:]))[:, np.newaxis, :]
    segments = np.concatenate((coord_start, coord_mid, coord_end), axis=1)

    lc = LineCollection(segments, **default_kwargs)
    lc.set_array(c)  # set the colors of each segment

    return ax.add_collection(lc)


def flower_test(cycles, step_size, n_cycles, x_start, x_end):
    total_points = cycles * step_size
    half_total_points = int(total_points / 2)

    # X-axis: time or angle
    x = np.linspace(x_start, x_end * np.pi * cycles, total_points)

    # Amplitude envelope: ramp up and down
    # For example: triangular ramp
    ramp_up = np.linspace(x_start, x_end * 2, total_points // 2)
    ramp_down = np.linspace(x_end * 2, x_start, total_points // 2)
    amplitude_envelope = np.concatenate([ramp_up, ramp_down])

    # Sine wave with varying amplitude

    y = np.tile(amplitude_envelope * np.sin(x), n_cycles)
    x = y[:-half_total_points]
    y = y[half_total_points:]

    # Plot
    plt.figure(figsize=(12, 10))
    plt.plot(x, y)
    plt.show()


def tree_plot(
    n_lines,
    x_length,
    sigma_dispersion=0,
    step_size=0.1,
    proportions=[0.1, 0.25, 0.65],
    phi_base=1.05,
    base_sd=1,
    branch_segment_length=10,
    branch_angle=np.pi / 6,  # angle for branching
    split_probability=0.9,  # Probability of splitting at each branch
    **kwargs,
):
    x_count = x_length / step_size

    # Initializing base/branches:
    len_base = int(proportions[0] * x_count)
    y_base = np.zeros(len_base)

    len_branches = int(proportions[2] * x_count)
    n_branch_segments = int(len_branches / branch_segment_length)
    len_branches = n_branch_segments * branch_segment_length

    x_count = len_base + len_branches

    x = np.arange(step=step_size, stop=x_length)[
        : int(x_count)
    ]  # Ensure x_count is int for slicing

    # Tree base:
    phi_samples = np.sort(np.random.normal(0, base_sd, n_lines))

    phi_samples_abs_sorted = np.sort(np.abs(phi_samples))
    value_at_70th_percentile = np.percentile(phi_samples_abs_sorted, 50)

    dispersions = np.random.normal(0, sigma_dispersion, n_lines)

    fig, ax = plt.subplots(figsize=LANDSCAPE_DIMENSIONS)
    ax.axis("off")

    gradient_1 = create_multi_cmap(["#FF69B4", "#FFFFFF", "#CC7722"], n_lines)

    # --- Branching logic ---
    # In this plot, x is vertical (height), y is horizontal (spread).
    def draw_branch(
        x0, y0, angle, length, depth, max_depth, current_line_color, linewidth=2
    ):
        if depth > max_depth or length < 1:
            return

        branch_x_coords = [x0]
        branch_y_coords = [y0]
        segs = 2  # Number of points to define the curve of this branch segment

        # A single random factor for the curvature magnitude of this specific branch segment
        curvature_random_mag_factor = np.random.uniform(0.8, 1.2)

        for i in range(1, segs + 1):
            frac = i / segs  # Fractional distance along the branch segment's chord

            # Calculate the point on the straight line (chord) of the branch segment
            current_x_on_chord = x0 + length * frac * np.cos(angle)
            current_y_on_chord = y0 + length * frac * np.sin(angle)

            # Add perpendicular sinusoidal displacement for curvature
            curvature_displacement = (
                np.sin(frac * np.pi) * (length * 0.2) * curvature_random_mag_factor
            )
            current_y_on_chord += curvature_displacement * np.random.uniform(
                -0.3, 0.3
            )  # Randomize curvature direction

            branch_x_coords.append(current_x_on_chord)
            branch_y_coords.append(current_y_on_chord)

        ax.plot(
            branch_y_coords,
            branch_x_coords,
            color=current_line_color,
            linewidth=linewidth,
            **kwargs,
        )

        # Recursive calls for sub-branches
        if depth < max_depth and np.random.rand() < split_probability:
            end_x = branch_x_coords[-1]
            end_y = branch_y_coords[-1]

            decay_factor = 0.65 + 0.1 * np.random.rand()
            new_length = length * decay_factor
            new_linewidth = linewidth * 0.7

            angle_jitter = np.random.normal(0, 0.05 * (depth + 1))

            draw_branch(
                end_x,
                end_y,
                angle + branch_angle + angle_jitter,
                new_length,
                depth + 1,
                max_depth,
                current_line_color,
                new_linewidth,
            )
            draw_branch(
                end_x,
                end_y,
                angle - branch_angle + angle_jitter,
                new_length,
                depth + 1,
                max_depth,
                current_line_color,
                new_linewidth,
            )

    for line_idx in range(n_lines):
        current_color = gradient_1[line_idx]

        # The base:
        y_base_current = np.zeros(len_base)  # Initialize for current line
        for t in np.arange(len_base - 1, -1, -1):
            if t == len_base - 1:
                y_base_current[t] = phi_samples[line_idx]
            else:
                y_base_current[t] = y_base_current[t + 1] * phi_base

        # Draw base as a single line (y is horizontal, x is vertical)
        ax.plot(
            y_base_current, x[:len_base], color=current_color, linewidth=3, **kwargs
        )

        # Start branches from the top of the base
        if (
            len(x[:len_base]) > 0 and len(y_base_current) > 1
        ):  # Check if base has points
            base_top_x = x[len_base - 1]
            base_top_y = y_base_current[-1]

            # First branch follows the same angle as the last part of the base
            initial_branch_angle = np.arctan2(
                base_top_y - y_base_current[-2], step_size
            )

            if (line_idx % 4 == 0) or (
                abs(phi_samples[line_idx]) > value_at_70th_percentile
            ):
                max_depth = np.random.randint(1, 3)
                draw_branch(
                    base_top_x,
                    base_top_y,
                    initial_branch_angle,
                    branch_segment_length * 1.2,
                    0,
                    max_depth,
                    current_color,
                    linewidth=2,
                )

                # Additional branches
                for _ in range(2):  # Draw 2 more main branches
                    random_branch_angle = np.random.uniform(-np.pi / 8, np.pi / 8)
                    draw_branch(
                        base_top_x,
                        base_top_y,
                        initial_branch_angle + random_branch_angle,
                        branch_segment_length * 1.2,
                        0,
                        4,
                        current_color,
                        linewidth=2,
                    )

    ax.set_facecolor("#FFE5B4")
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.margins(0.05)
    plt.tight_layout(pad=0.2)
    plt.savefig("outputs/max_plot.png", transparent=True, dpi=300)
    plt.show()


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
    from src.utils.colors import victoria_colors

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

    # # Animation
    # filenames = []

    # # Create directory for animation frames if it doesn't exist
    # anim_dir = "taylor_animation"
    # if not os.path.exists(anim_dir):
    #     os.makedirs(anim_dir)

    # # Create zoom levels
    # zoom_levels = np.linspace(zoom_range[0], zoom_range[1], frames)

    # for f in range(frames):
    #     # Current zoom level
    #     current_zoom = zoom_levels[f]
    #     x = np.linspace(-current_zoom, current_zoom, num_points)
    #     y_actual = np.sin(x)

    #     plt.figure(figsize=(10, 6), dpi=100)
    #     ax = plt.gca()

    #     # Plot the actual sine curve if requested
    #     if show_actual:
    #         ax.plot(x, y_actual, 'k--', linewidth=line_thickness/2, alpha=0.5)

    #     # Create a color map for the approximations
    #     cmap = create_multi_cmap(colors, n_terms)

    #     # Plot Taylor series approximations
    #     for i in range(1, n_terms + 1):
    #         y_approx = np.zeros_like(x)

    #     # Calculate the Taylor series up to i terms
    #     for n in range(i):
    #         y_approx += ((-1)**n * x**(2*n+1)) / np.math.factorial(2*n+1)

    #     ax.plot(x, y_approx, linewidth=line_thickness, color=cmap[i-1])

    #     ax.set_title(f'Taylor Series of Sine Wave (Zoom: {current_zoom:.2f})')
    #     ax.set_ylim(-2, 2)

    #     ax.set_facecolor(bg_color)

    #     # Save frame
    #     filename = os.path.join(anim_dir, f"frame_{f}.png")
    #     plt.savefig(filename)
    #     filenames.append(filename)
    #     plt.close()

    # # Create GIF
    # gif_filename = "taylor_animation.gif"
    # with imageio.get_writer(gif_filename, mode="I", duration=0.1) as writer:
    #     for filename in filenames:
    #         image = imageio.imread(filename)
    #         writer.append_data(image)

    # print(f"Animation saved as {gif_filename}")


def butterfly_plot(
    t_len,
    color_len,
    phi_step,
    step=0.01,
    line_thickness=1,
):
    from src.utils.colors import victoria_colors
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
        # dpi = 300,
    )

    plt.show()


def warp_plot(num_lines, x_start, x_end, **kwargs):
    from src.utils.colors import gvantsa_colors
    
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


def save_for_later(
    num_lines,
    x_start,
    x_end,
):
    # Set up canvas
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_aspect("equal")
    ax.axis("off")

    # Parameters
    x = np.linspace(x_start, x_end, 1000)
    # x_first = x[:int(len(x)/2)]
    # x_second = x[int(len(x)/2):]

    for i in range(num_lines):
        # First part

        shift = i * 0.1
        stretch = 1 + 0.5 * np.sin(i * 3)

        # Warped wave-like line
        y = (
            np.tanh(x * 0.5) * (i / 10)
            + np.sin(x * stretch) * 0.1 * stretch
            + np.cos(x)
            + np.arctan(x * 2) ** i
        )

        plt.plot(x, y, color="black")
        plt.plot(x, np.flip(y), color="black")
        # plt.plot(np.flip(x), y, color = 'black')
        plt.plot(np.flip(x), np.flip(y), color="black")

        plt.plot(y, x, color="black")
        plt.plot(y, np.flip(x), color="black")
        # plt.plot(np.flip(y), x, color = 'black')
        plt.plot(np.flip(y), np.flip(x), color="black")

        # Second part

    plt.tight_layout()
    plt.show()

