import numpy as np

from ...geom import Rectangle
from .base_rectangularizer import BaseRectangularizer


class TrivialRectangularizer(BaseRectangularizer):
    def __init__(
        self,
        overlap: float = 0.1,
    ) -> None:
        super().__init__(overlap=overlap)

    def rectangularize_tissue(
        self,
        contours: list[np.ndarray],
        scale: float = 1,
    ) -> list[Rectangle]:
        """
        For each contour in contours, we create its bounding box
        """

        def bbox(contour):
            max_x, max_y = contour.max(axis=0)
            min_x, min_y = contour.min(axis=0)
            return Rectangle(
                min_x,
                min_y,
                max_x - min_x,
                max_y - min_y,
            ).scale_around_center(self.overlap + 1)

        return [bbox(contour.squeeze()) for contour in contours]
