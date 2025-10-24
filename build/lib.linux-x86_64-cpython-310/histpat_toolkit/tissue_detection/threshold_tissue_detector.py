import cv2 as cv
import numpy as np

from .base_tissue_detector import BaseTissueDetector


class ThresholdTissueDetector(BaseTissueDetector):
    def __init__(
        self,
        threshold: int,
        mode: int = cv.THRESH_BINARY_INV,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        self.threshold = threshold
        self.mode = mode

    def _detect_tissue(
        self,
        img: np.ndarray[np.uint8],
    ) -> np.ndarray[np.uint8]:
        """This function returns the binary mask of the tissue in the image"""
        img = self.prepare_grayscale_mask(img)
        return cv.threshold(
            img,
            self.threshold,
            255,
            self.mode,
        )[1]
