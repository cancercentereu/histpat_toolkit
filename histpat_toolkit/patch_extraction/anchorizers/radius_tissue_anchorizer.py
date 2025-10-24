import numpy as np

from .base_tissue_anchorizer import BaseTissueAnchorizer
from .radius_tissue_helper import anchorize_tissue


class RadiusTissueAnchorizer(BaseTissueAnchorizer):
    def __init__(
        self,
        radius: int = 100,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.radius = radius

    def anchorize_tissue(
        self,
        tissue_mask: np.ndarray[np.uint8],
        scale: float = 1,
    ) -> np.ndarray[np.uint8]:
        """
        This function performs BFS with the given radius
        from points that are in the tissue_mask (and marks
        points visited so that they are no longer feasible
        candidates for the centroids).

        The code of the BFS is implemented in cython (.pyx file)
        for performance reasons
        """
        return anchorize_tissue(
            tissue_mask,
            max(
                1,
                self.radius * scale,
            ),
        )
