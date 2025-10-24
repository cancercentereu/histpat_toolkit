import cv2 as cv
import numpy as np

from .base_tissue_labeler import BaseTissueLabeler
from .bfs_nearest_helper import cython_bfs_nearest_labeling


class BFSNearestTissueLabeler(BaseTissueLabeler):
    def __init__(self) -> None:
        super().__init__()

    def label_by_anchors_mask(
        self,
        anchors_mask: np.ndarray[np.int32],
        tissue_mask: np.ndarray[np.uint8],
        scale: float = None,
    ) -> np.ndarray[np.int32]:
        """
        Simply run algorithm to detect all connected Components
        of the tissue_mask and ignore anchors_mask.
        """

        _, markers = cv.connectedComponents(
            anchors_mask,
            connectivity=8,
        )
        labels = cython_bfs_nearest_labeling(tissue_mask, markers.astype(np.int32))
        return labels
