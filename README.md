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
```

## Installation

1. Clone this repository:
```bash
git clone https://github.com/MarcoGriepp/GraphArt.git
cd GraphArt
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Using the Jupyter Notebook

Open and run `notebooks/main.ipynb` to explore all the different plot functions interactively.

### Using Individual Plot Functions

Each plot function can be imported and used directly:

```python
from src.plots import warp_plot, taylor_expansion_plot, tree_plot

# Generate a warped sine wave visualization
warp_plot(num_lines=20, x_start=-8, x_end=8, linewidth=17.3)

# Create a Taylor series expansion plot
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

# Generate a fractal tree
tree_plot(
    n_lines=80,
    x_length=15,
    proportions=[0.36, 0, 0.64],
    phi_base=1.05,
    branch_segment_length=10,
    branch_angle=np.pi/12,
)
```

### Available Plot Functions

- **`warp_plot`**: Creates warped sine wave patterns with artistic transformations
- **`taylor_expansion_plot`**: Visualizes Taylor series approximations of sine waves
- **`tree_plot`**: Generates fractal tree structures with branching patterns
- **`noisewave_plot`**: Creates symmetric wave patterns with controlled noise
- **`step_sine_plot`**: Combines sine waves with stepped random walks
- **`butterfly_plot`**: Renders parametric butterfly curves

### Color Palettes

The project includes several curated color palettes accessible via `src.utils.colors`:

- `taro_colors`
- `gvantsa_colors`
- `victoria_colors`
- `victor_colors` / `victor_colors_2`
- `max_colors` / `max_colors_2`
- `jil_colors`
- `mehdi_colors`

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
 
