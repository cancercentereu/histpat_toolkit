from abc import ABC, abstractmethod

import numpy as np

from ...geom import Rectangle


class BaseRectangularizer(ABC):
    def __init__(
        self,
        overlap: float = 0.1,
    ) -> None:
        """
        This is a base class used by patchers for covering tissue
        areas with rectangles

        Parameters:
            overlap: overlap between the patches (fraction of the patch size)
        """
        self.overlap = overlap

    @abstractmethod
    def rectangularize_tissue(
        self,
        contours: list[np.ndarray],
        scale: float = 1,
    ) -> list[Rectangle]:
        """
        Given contours, this function should return a list of rectangles

        Parameters:
            contours: list of contours (each contour is array of points)
                Each contour is of the shape (n, 1, 2)
            scale: the scale that the image was resized with before patching
        """
        ...
