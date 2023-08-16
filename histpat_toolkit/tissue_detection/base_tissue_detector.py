import numpy as np
from abc import ABC, abstractmethod
import cv2


class BaseTissueDetector(ABC):
    def __init__(self, erosion_ksize: int = None, morph_ksize: int = None) -> None:
        self.erosion_ksize = erosion_ksize
        self.morph_ksize = morph_ksize

    @abstractmethod
    def detect_tissue(self, img: np.ndarray) -> np.ndarray:
        """ This function returns the binary mask of the tissue in the image """
        pass

    def prepare_grayscale_image(self, img: np.ndarray) -> np.ndarray:
        img = img[:, :, :min(img.shape[-1], 3)]
        img = img.mean(axis=-1)

        if self.erosion_ksize is not None:
            eroded_img = cv2.erode(img, np.ones(
                (self.erosion_ksize,)*2, np.uint8), iterations=1)
            img[eroded_img == 0] = 255

        if self.morph_ksize is not None:
            # We open image instread of closing since later we use invert threshold
            img = cv2.morphologyEx(img.astype(np.uint8), cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.morph_ksize,)*2))

        return img.astype(np.uint8)
