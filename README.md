# GraphArt

A small personal project to create art from graphs in Matplotlib! This project contains a collection of artistic visualizations created using mathematical functions and custom plotting techniques.

## Project Structure

```
GraphArt/
├── src/
│   ├── __init__.py
│   ├── plots/                  # Plot generation functions
│   │   ├── __init__.py
│   │   ├── gvantsa_plot.py    # Warped sine wave visualization
│   │   ├── jil_plot.py        # Taylor series expansion plot
│   │   ├── max_plot.py        # Fractal tree visualization
│   │   ├── taro_plot.py       # Symmetric noisewave patterns
│   │   ├── victor_plot.py     # Step sine with random walk
│   │   └── victoria_plot.py   # Parametric butterfly curve
│   └── utils/                  # Shared utilities
│       ├── __init__.py
│       ├── colors.py          # Color palettes
│       └── functions.py       # Utility functions
├── notebooks/
│   └── main.ipynb             # Main exploration notebook
├── outputs/                    # Generated artwork
│   ├── images/
│   └── source_files/          # GIMP source files (.xcf)
├── requirements.txt
└── README.md

### Available Plot Functions

- **`warp_plot`**: Creates warped sine wave patterns with artistic transformations
- **`taylor_expansion_plot`**: Visualizes Taylor series approximations of sine waves
- **`tree_plot`**: Generates fractal tree structures with branching patterns
- **`noisewave_plot`**: Creates symmetric wave patterns with controlled noise
- **`step_sine_plot`**: Combines sine waves with stepped random walks
- **`butterfly_plot`**: Renders parametric butterfly curves

## Utilities

The `src.utils` module provides:

- **`create_cmap(color_1, color_2, length)`**: Create a gradient between two colors
- **`create_multi_cmap(colors, length)`**: Create a gradient from multiple colors
- **`show_colors(colors, title)`**: Display a color palette
- **`colored_line(...)`**: Plot lines with color gradients
- **`LANDSCAPE_DIMENSIONS`**: Standard landscape canvas size (4961/300 × 3508/300)
- **`PORTRAIT_DIMENSIONS`**: Standard portrait canvas size (3508/300 × 4961/300)

## Output

All generated artwork is saved to the `outputs/` directory:
- PNG exports go to `outputs/images/`
- GIMP source files (.xcf) are stored in `outputs/source_files/`

## License

Personal project - feel free to use and modify as you like!
 
