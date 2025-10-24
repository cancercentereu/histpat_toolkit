from abc import ABC, abstractmethod
import numpy as np


class BaseTissueModifier(ABC):
    def __init__(self) -> None:
        pass

    @abstractmethod
    def modify_tissue(self,
                      tissue_mask: np.ndarray[np.uint8],
                      scale: float = 1,
                      ) -> np.ndarray[np.uint8]:
        """
            Before passing to the anchorizer, we may want to modify the
            tissue (for example by thinning it, to improve performance)

            Parameters:
                tissue_mask: binary mask of the tissue
                scale: the scale that the image was resized with before patching
        """
        ...
