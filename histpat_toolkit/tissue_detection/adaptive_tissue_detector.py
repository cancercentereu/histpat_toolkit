from .base_tissue_detector import BaseTissueDetector
from abc import ABC
import cv2 as cv
import numpy as np


class AdaptiveTissueDetector(BaseTissueDetector, ABC):
    def __init__(self, mode: int, block_size: int = 11, C: int = 2, **kwargs) -> None:
        super().__init__(**kwargs)

        assert mode in [cv.ADAPTIVE_THRESH_MEAN_C,
                        cv.ADAPTIVE_THRESH_GAUSSIAN_C]
        self.mode = mode

        self.block_size = block_size
        self.C = C

    def detect_tissue(self, img: np.ndarray) -> np.ndarray:
        """ This function returns the binary mask of the tissue in the image """
        img = super().prepare_grayscale_image(img)
        return cv.adaptiveThreshold(img, 255, self.mode, cv.THRESH_BINARY_INV, self.block_size, self.C)
