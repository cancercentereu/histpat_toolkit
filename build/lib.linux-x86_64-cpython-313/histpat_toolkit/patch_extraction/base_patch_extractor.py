from abc import ABC, abstractmethod

import cv2
import numpy as np

from ..geom import Rectangle
from ..tissue_detection.base_tissue_detector import BaseTissueDetector


class BasePatchExtractor(ABC):
    def __init__(
        self,
        tissue_detector: BaseTissueDetector,
        max_area: int | None = None,
        discard_components_under_area: float | None = 0.15,
    ) -> None:
        self.tissue_detector = tissue_detector
        self.max_area = max_area
        self.discard_components_under_area = discard_components_under_area

    @abstractmethod
    def _extract_patches(
        self,
        mask: np.ndarray[np.uint8],
        scale: float,
        **kwargs,
    ) -> list[Rectangle]:
        """
        This function should return a list of patches
        Parameters:
            mask: np.ndarray
                Mask of detected tissue
            scale:
                The scale by which the original extract_patches(img) was scaled by
                this means, we may want to rescale some parameters in the anchorizer
                or other classes
            discard_components_under_area:
                The threshold under which areas are discarded (in square milimiters)
        Returns:
            List of patches. Patches should be in the scale for mask (not the original image).
            They will be scaled after returning from this function
        """
        ...

    def extract_patches(
        self,
        img: np.ndarray[np.uint8],
        img_mpp: float | None = None,
        **kwargs,
    ) -> list[Rectangle]:
        """
        Wrapper function for _extract_patches that first detects the tissue and
        then if needed resizes the result before passing to patch extractor

        Parameters:
            img: np.ndarray
                The image to extract the patches from
            img_mpp: float
                The microns per pixel of the image (this parameter may be used )
            **kwargs:
                some arguments to pass to the _extract_patches function
        Returns:
            List of patches (Instances of Rectangle class from geom.py).
        """

        tissue_mask = self.tissue_detector.detect_tissue(
            img,
            img_mpp,
        )
        self.remove_small_components(
            tissue_mask,
            img_mpp=img_mpp,
        )

        if self.max_area is not None:
            img_area = img.shape[0] * img.shape[1]
            scale = min(1, np.sqrt(self.max_area / img_area))
        else:
            scale = 1

        if scale != 1:
            downscaled_dims = (round(img.shape[1] * scale), round(img.shape[0] * scale))
            tissue_mask = cv2.resize(
                tissue_mask,
                downscaled_dims,
                interpolation=cv2.INTER_NEAREST,
            )

        patches = self._extract_patches(
            tissue_mask,
            scale,
            **kwargs,
        )
        if scale != 1:
            patches = [patch.scale(1 / scale) for patch in patches]
        return patches

    def remove_small_components(
        self,
        mask: np.ndarray[np.uint8],
        img_mpp: float | None = None,
    ) -> None:
        """
        This function removes small components from the mask

        Parameters:
            mask: mask of the tissue
            size: max size of the removed components
            img_mpp: microns per pixel of the image
        Returns:
            None (modifies mask inplace)
        """

        if img_mpp is not None and self.discard_components_under_area:
            area_in_milimeters = self.discard_components_under_area
            area_in_micrometers = area_in_milimeters * 1000 * 1000
            size = area_in_micrometers / (img_mpp**2)

            _, labels, stats, _ = cv2.connectedComponentsWithStats(
                mask.astype(np.uint8),
                connectivity=4,
            )
            mask[stats[labels, 4] <= size] = 0
