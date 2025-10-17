"""Utility functions and color palettes for GraphArt."""

from .functions import (
    create_cmap,
    create_multi_cmap,
    show_colors,
    colored_line,
    LANDSCAPE_DIMENSIONS,
    PORTRAIT_DIMENSIONS,
)

from .colors import (
    taro_colors,
    gvantsa_colors,
    victoria_colors,
    victor_colors,
    victor_colors_2,
    mehdi_colors,
    max_colors,
    max_colors_2,
    jil_colors,
    colormaps,
    sort_colors_by_shade,
)

__all__ = [
    # Functions
    'create_cmap',
    'create_multi_cmap',
    'show_colors',
    'colored_line',
    # Constants
    'LANDSCAPE_DIMENSIONS',
    'PORTRAIT_DIMENSIONS',
    # Color palettes
    'taro_colors',
    'gvantsa_colors',
    'victoria_colors',
    'victor_colors',
    'victor_colors_2',
    'mehdi_colors',
    'max_colors',
    'max_colors_2',
    'jil_colors',
    'colormaps',
    'sort_colors_by_shade',
]
