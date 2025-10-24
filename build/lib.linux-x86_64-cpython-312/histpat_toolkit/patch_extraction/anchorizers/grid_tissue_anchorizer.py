from .base_tissue_anchorizer import BaseTissueAnchorizer
import numpy as np


class GridTissueAnchorizer(BaseTissueAnchorizer):

    def __init__(self,
                 width: int = 100,
                 height: int = 100,
                 **kwargs,
                 ) -> None:
        """
            Parameters:
                width: width of the grid cell
                height: height of the grid cell
        """
        super().__init__(**kwargs)
        self.width = width
        self.height = height

    def anchorize_tissue(self,
                         tissue_mask: np.ndarray[np.uint8],
                         scale: float = 1,
                         ) -> np.ndarray[np.uint8]:
        """
            This function assigns a centroid to each point
            on the [self.width x self.height] grid if the point
            belongs to the tissue_mask
        """
        anchors_mask = np.zeros(tissue_mask.shape)

        for i in range(0, tissue_mask.shape[0], max(1, round(self.height * scale))):
            for j in range(0, tissue_mask.shape[1], max(1, round(self.width * scale))):
                if tissue_mask[i, j] != 0:
                    anchors_mask[i, j] = 255

        return anchors_mask.astype(np.uint8)
