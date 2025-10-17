"""Max's tree plot - fractal tree visualization."""

import numpy as np
import matplotlib.pyplot as plt
from src.utils import create_multi_cmap, LANDSCAPE_DIMENSIONS


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


if __name__ == "__main__":
    # Example usage
    tree_plot(
        n_lines=80,
        x_length=15,
        proportions=[0.36, 0, 0.64],
        phi_base=1.05,
        branch_segment_length=10,
        branch_angle=np.pi / 12,
    )
