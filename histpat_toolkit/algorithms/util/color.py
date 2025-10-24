from dataclasses import dataclass

import numpy as np


@dataclass
class Color:
    key: int
    value: tuple[int, int, int, int]


def get_colormap_lut(colors: list[Color], save_classes: list[int] | None = None):
    lut = np.zeros((256, 4), dtype="uint8")
    for col in colors:
        if save_classes is None or col.key in save_classes:
            lut[col.key] = col.value

    return lut
