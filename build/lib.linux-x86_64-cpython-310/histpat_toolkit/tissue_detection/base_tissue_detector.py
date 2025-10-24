import numpy as np
from abc import ABC, abstractmethod
import cv2


class BaseTissueDetector(ABC):
    def __init__(self,
                 opening_ksize: int = None,
                 closing_ksize: int = None,
                 mpp: float = None,
                 ) -> None:
        """
            This is a base class for tissue detectors.

            Parameters:
                opening_ksize (int): default kernel size for cv2 opening
                    operations while detecting tissue 
                closing_ksize (int): default kernel size for cv2 closing
                    operations while detecting tissue
                mpp (float): microns per pixel for the image (for example 10).
                    If provided, image will be resized to have this mpp before
                    detecting tissue. If not provided, image will not be resized
        """
        self.opening_ksize = opening_ksize
        self.closing_ksize = closing_ksize
        self.mpp = mpp

    @abstractmethod
    def _detect_tissue(self,
                       img: np.ndarray[np.uint8],
                       ) -> np.ndarray[np.uint8]:
        """
            This abstract function should extract the mask of
            the tissue given the image 

            Parameters:
                img (np.ndarray): image to detect tissue on

            Returns:
                np.ndarray: mask of the tissue in the image
                (1 - tissue, 0 - background)
        """
        ...

    def detect_tissue(self,
                      img: np.ndarray[np.uint8],
                      img_mpp: float = None,
                      ) -> np.ndarray[np.uint8]:
        """
            This is a wrapper function for _detect_tissue that also
            resizes image if needed (if mpp parameter is provided 
            both to the detector class and to the function)

            Parameters:
                img (np.ndarray): image to detect tissue on
                img_mpp (float): microns per pixel of the img
        """
        needs_resizing = self.mpp is not None \
            and img_mpp is not None \
            and self.mpp != img_mpp

        if needs_resizing:
            # We resize image to mpp we would like to use while detecting tissue
            scale = img_mpp / self.mpp
            orig_size = (img.shape[1],
                         img.shape[0],
                         )
            new_size = (round(img.shape[1] * scale),
                        round(img.shape[0] * scale),
                        )
            img = cv2.resize(img,
                             new_size,
                             interpolation=cv2.INTER_LINEAR,
                             ).astype(np.uint8)

        mask = self._detect_tissue(img)

        if needs_resizing:
            mask = cv2.resize(mask,
                              orig_size,
                              interpolation=cv2.INTER_NEAREST,
                              )

        return mask.astype(np.uint8)

    def smooth_mask(self,
                    mask: np.ndarray[np.uint8],
                    opening_ksize: int = None,
                    closing_ksize: int = None,
                    ) -> np.ndarray[np.uint8]:
        """
            This function performs closing or opening of the image/mask.

            Opening the image is good for removing small objects
            from outsite of the tissue

            Closing the image is good for removing small holes
            from within the tissue

            Parameters:
                mask (np.ndarray): mask to smooth
                opening_ksize (int): kernel size for opening
                closing_ksize (int): kernel size for closing

            Returns:
                np.ndarray: smoothed mask
        """

        operations = []
        if opening_ksize is not None:
            operations.append((cv2.MORPH_OPEN, (opening_ksize,)*2))

        if closing_ksize is not None:
            operations.append((cv2.MORPH_CLOSE, (closing_ksize,)*2))

        for (operation, ksize) in operations:
            mask = cv2.morphologyEx(mask,
                                    operation,
                                    cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                                              ksize,
                                                              ),
                                    iterations=1,
                                    )

        return mask.astype(np.uint8)

    def prepare_grayscale_mask(self,
                               img: np.ndarray[np.uint8],
                               ) -> np.ndarray[np.uint8]:

        if len(img.shape) == 2:
            # We assume that the image is already grayscale
            return img

        # We remove alpha channel if it exists
        img = img[:, :, :min(img.shape[-1], 3)]

        # Gray value is the mean of values of RGB channels
        mask = img.mean(axis=-1)

        return self.smooth_mask(mask,
                                self.opening_ksize,
                                self.closing_ksize,
                                ).astype(np.uint8)
