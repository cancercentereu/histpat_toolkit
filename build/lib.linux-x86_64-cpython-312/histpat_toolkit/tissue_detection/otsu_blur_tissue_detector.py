from .otsu_tissue_detector import OtsuTissueDetector
import cv2 as cv
import numpy as np

class OtsuBlurTissueDetector(OtsuTissueDetector):
    def __init__(self,
                 blur_range: int = 2,
                 **kwargs,
                 ) -> None:
        super().__init__(**kwargs)
        self.blur = blur_range

    def prepare_grayscale_mask(self,
                                img: np.ndarray[np.uint8],
                                ) -> np.ndarray[np.uint8]:
        img = super().prepare_grayscale_mask(img)
        return cv.GaussianBlur(img,
                               (self.blur * 2 + 1,
                                self.blur * 2 + 1),
                                0,
                                ).astype(np.uint8)