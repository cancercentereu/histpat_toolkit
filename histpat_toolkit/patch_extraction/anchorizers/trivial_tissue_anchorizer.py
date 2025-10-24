import numpy as np

from .base_tissue_anchorizer import BaseTissueAnchorizer


class TrivialTissueAnchorizer(BaseTissueAnchorizer):
    def __init__(
        self,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

    def anchorize_tissue(
        self,
        tissue_mask: np.ndarray[np.uint8],
        scale: float = 1,
    ) -> np.ndarray[np.uint8]:
        """
        Trivial anchorizer returns entire tissue mask
        as anchors, which means that each connected component
        will be covered with exactly one patch
        """
        return tissue_mask.astype(np.uint8)
