import colorsys


def sort_colors_by_shade(color_dict):
    """Sorts colors based on hue, saturation, and value (HSV)."""
    return dict(
        sorted(
            color_dict.items(),
            key=lambda item: colorsys.rgb_to_hsv(
                int(item[1][1:3], 16),  # R
                int(item[1][3:5], 16),  # G
                int(item[1][5:7], 16),  # B
            ),
        )
    )


taro_colors = {
    "Cosmic Purple": "#6F6092",
    "Pine Green": "#01796F",
    "Fern Green": "#4F7942",
    "Algae Green": "#93DFB8",
    "Deep Ocean Blue": "#00587A",
}

gvantsa_colors = sort_colors_by_shade(
    {"Yellow": "#FFFF00", "Raspberry Pink": "#E30B5C", "Lilac Purple": "#C8A2C8"}
)

victoria_colors = {
    "Soft Pink": "#F4C2C2",
    "Mauve": "#E0B0FF",
    "Lilac": "#C8A2C8",
    "Matcha Green": "#A0C25A",
    "Amethyst": "#9966CC",
    "Plum": "#8E4585",
    "Cobalt Blue": "#0047AB",
}

victor_colors = sort_colors_by_shade(
    {"Black": "#000000", "Beige": "#F5F5DC", "Sky Blue": "#87CEEB"}
)

victor_colors_2 = {
    "Onyx": "#1B1B1B",
    "Night": "#1E3A6A",
    "Sky blue": "#A3B8ED",
    # "Clay":"#8B7F71",
    "Sand": "#D1BDA4",
    "Ivory": "#ECE3DA",
}

mehdi_colors = sort_colors_by_shade(
    {"Black": "#000000", "Beige": "#F5F5DC", "Blue": "#0000FF", "Turquoise": "#40E0D0"}
)

max_colors = sort_colors_by_shade(
    {
        "Bordeaux Red": "#7D1B19",
        "Dark Teal": "#014D4E",
        "Pastel Blue": "#A7C7E7",
        "Pink (Juicy)": "#FF69B4",
        "Ocher Yellow": "#CC7722",
    }
)

max_colors_2 = {
    "Dark Teal": "#014D4E",
    "Bordeaux Red": "#7D1B19",  # Rich wine red
    "Ocher Yellow": "#CC7722",  # Earthy golden yellow
    "Burnt Orange": "#E25822",  # Warm, vivid orange
    "Clay Rose": "#C97C5D",  # Muted pinkish clay
    "Peach Cream": "#FFE5B4",  # Soft pastel for contrast
}  # Light warm pastel, balances intensity

jil_colors = sort_colors_by_shade(
    {
        "Pink": "#FFC0CB",
        "Turquoise": "#40E0D0",
        "Lime Bright Green": "#32CD32",
        "Beige": "#F5F5DC",
    }
)
colormaps = [
    ("taro_colors", taro_colors),
    ("gvantsa_colors", gvantsa_colors),
    ("victoria_colors", victoria_colors),
    ("victor_colors", victor_colors),
    ("victor_colors_2", victor_colors_2),
    ("mehdi_colors", mehdi_colors),
    ("jil_colors", jil_colors),
    ("max_colors", max_colors),
    ("max_colors_2", max_colors_2),
    ]
