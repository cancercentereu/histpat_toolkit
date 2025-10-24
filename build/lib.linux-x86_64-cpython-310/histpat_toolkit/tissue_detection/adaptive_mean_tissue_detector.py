from .adaptive_tissue_detector import AdaptiveTissueDetector
import cv2 as cv


class AdaptiveMeanDetector(AdaptiveTissueDetector):
    def __init__(self,
                 block_size: int = 11,
                 C: int = 2,
                 **kwargs,
                 ) -> None:
        super().__init__(mode=cv.ADAPTIVE_THRESH_MEAN_C,
                         block_size=block_size,
                         C=C,
                         **kwargs,
                         )
