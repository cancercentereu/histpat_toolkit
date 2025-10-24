import numpy as np
from cv2.ximgproc import thinning

from .base_tissue_modifier import BaseTissueModifier


class ThinningTissueModifier(BaseTissueModifier):
    def modify_tissue(
        self,
        tissue_mask: np.ndarray[np.uint8],
        scale: float = 1,
    ) -> np.ndarray[np.uint8]:
        """
        Apply cv2.ximgproc.thinning to the tissue mask.
        The result can be thought of the 'skeleton' of the original
        tissue mask

        Parameters:
        ----------
            tissue_mask: binary mask of the tissue
            scale: the scale that the image was resized by before patching
        """
        return thinning(tissue_mask).astype(np.uint8)
