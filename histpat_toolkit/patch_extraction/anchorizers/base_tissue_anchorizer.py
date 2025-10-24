from abc import ABC, abstractmethod

import numpy as np


class BaseTissueAnchorizer(ABC):
    def __init__(sefl):
        pass

    @abstractmethod
    def anchorize_tissue(
        self,
        tissue_mask: np.ndarray[np.uint8],
        scale: float = 1,
    ) -> np.ndarray[np.uint8]:
        """
        This function returns works on the potentially modified
        mask of the original tissue. It tries to find the centroids
        of the patches in the image. It returns a mask so that
        all the centroids are marked with 1 (each connected component
        stands for one centroid)

        Parameters:
        ----------
        tissue_mask: potentially modified mask of the original tissue
        scale: The scale that original image was multiplied by before
            being passed to the anchorizer
        """
        ...
