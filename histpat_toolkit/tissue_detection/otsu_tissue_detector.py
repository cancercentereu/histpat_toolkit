from .threshold_tissue_detector import ThresholdTissueDetector
import cv2 as cv

class OtsuTissueDetector(ThresholdTissueDetector):
    def __init__(self, **kwargs) -> None:
        super().__init__(threshold = 0, mode = cv.THRESH_OTSU, **kwargs)