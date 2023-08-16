from .base_tissue_detector import BaseTissueDetector
import cv2 as cv
import numpy as np


class ThresholdTissueDetector(BaseTissueDetector):
    def __init__(self, threshold: int, mode: int = 0, **kwargs) -> None:
        super().__init__(**kwargs)

        self.threshold = threshold
        self.mode = mode

    def detect_tissue(self, img: np.ndarray) -> np.ndarray:
        """ This function returns the binary mask of the tissue in the image """
        img = self.prepare_grayscale_image(img)
        return cv.threshold(img, self.threshold, 255, cv.THRESH_BINARY_INV + self.mode)[1]
