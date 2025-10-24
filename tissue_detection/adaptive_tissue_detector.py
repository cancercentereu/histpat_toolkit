from .base_tissue_detector import BaseTissueDetector
from abc import ABC
import cv2 as cv
import numpy as np


class AdaptiveTissueDetector(BaseTissueDetector, ABC):
    def __init__(self,
                 mode: int,
                 block_size: int = 11,
                 C: int = 2,
                 **kwargs,
                 ) -> None:
        super().__init__(**kwargs)

        assert mode in [cv.ADAPTIVE_THRESH_MEAN_C,
                        cv.ADAPTIVE_THRESH_GAUSSIAN_C
                        ]
        self.mode = mode

        self.block_size = block_size
        self.C = C

    def _detect_tissue(self,
                       img: np.ndarray[np.uint8],
                       ) -> np.ndarray[np.uint8]:
        """ This function returns the binary mask of the tissue in the image """
        img = super().prepare_grayscale_mask(img)
        return cv.adaptiveThreshold(img,
                                    maxValue=255,
                                    adaptiveMethod=self.mode,
                                    thresholdType=cv.THRESH_BINARY_INV,
                                    blockSize=self.block_size,
                                    C=self.C,
                                    )
