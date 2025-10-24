from abc import ABC, abstractmethod

import numpy as np


class BaseTissueLabeler(ABC):
    def __init__(self) -> None:
        pass

    @abstractmethod
    def label_by_anchors_mask(
        self,
        anchors_mask: np.ndarray[np.int32],
        tissue_mask: np.ndarray[np.uint8],
        scale: float = 1,
    ) -> np.ndarray[np.int32]:
        """
        Given anchors_mask and tissue_mask, this function should
        assign the anchors values to the rest of the tissue pixels
        (not yet covered by the anchors). For example we could assign
        for each pixel the value of the closest anchor

        Parameters:
        -----------
        anchors_mask:
            mask of the same size as tissue_mask with values
            corresponding to the labels of the anchors (0 - no anchor)
        tissue_mask:
            mask of the tissue (0 - background, 1 - tissue)
        scale:
            scale that the image was resized by before passing to the
            patcher
        """
        ...
