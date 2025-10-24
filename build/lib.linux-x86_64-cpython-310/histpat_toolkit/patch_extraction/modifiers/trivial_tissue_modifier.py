from .base_tissue_modifier import BaseTissueModifier
import numpy as np

class TrivialTissueModifier(BaseTissueModifier):

    def modify_tissue(self,
                      tissue_mask: np.ndarray[np.uint8],
                      scale: float = 1,
                      ) -> np.ndarray[np.uint8]:
        """
            Leave the tissue mask as is
        """
        return tissue_mask.astype(np.uint8)
