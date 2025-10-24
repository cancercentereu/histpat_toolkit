import cv2 as cv
import numpy as np

from .base_tissue_labeler import BaseTissueLabeler


class WatershedTissueLabeler(BaseTissueLabeler):
    def __init__(self) -> None:
        super().__init__()

    def label_by_anchors_mask(
        self,
        anchors_mask: np.ndarray[np.int32],
        tissue_mask: np.ndarray[np.uint8],
        scale: float = 1,
    ) -> np.ndarray[np.int32]:
        """
        Performs watershed with the given anchors.
        It can be thought as of flooding the tissue with water
        that is coming from the centroids of the patches.

        We remove artifacts since we want to enforce that
        each label is used only in the single connected component
        """
        _, markers = cv.connectedComponents(
            anchors_mask,
            connectivity=4,
        )

        # WARNING This may produce some artifacts which we remove later on
        markers = cv.watershed(
            cv.merge((tissue_mask,) * 3),
            markers,
        )

        markers[tissue_mask == 0] = 0
        markers[markers == -1] = 0

        # removing artifacts -- unfortunately this implementation seems to be
        # quite slow, but we ensure that only one component remains for each label
        # after the watershed algorithm

        artifacts = np.ones(markers.shape, bool)
        for label in range(1, markers.max() + 1):
            if (markers == label).any():
                label_mask = (markers == label).astype(np.uint8)
                _, comp_markers, stats, _ = cv.connectedComponentsWithStats(
                    label_mask,
                    connectivity=4,
                )
                largest_component_id = np.argmax(stats[1:, 4]) + 1

                # out of all components assigned to some marker we left only the largest one
                artifacts[comp_markers == largest_component_id] = 0

        markers[artifacts] = 0
        return markers.astype(np.uint32)
