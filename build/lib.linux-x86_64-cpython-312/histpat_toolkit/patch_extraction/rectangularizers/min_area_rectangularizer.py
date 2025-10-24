from .base_rectangularizer import BaseRectangularizer
import cv2
from ...geom import Rectangle, rotate_vector
import numpy as np

class MinAreaRectangularizer(BaseRectangularizer):

    def __init__(self,
                 overlap: float = 0.1,
                 ) -> None:
        super().__init__(overlap=overlap)

    def rectangularize_tissue(self,
                              contours: list[np.ndarray],
                              scale: float = 1,
                              ) -> list[Rectangle]:
        """
            Using cv2.minAreaRect to create rectangles
        """

        def create_rectangle(contour):
            box = cv2.boxPoints(cv2.minAreaRect(contour))
            rect = Rectangle.from_points(box)

            return rect.scale_around_center(self.overlap + 1)

        return [create_rectangle(contour) for contour in contours]
