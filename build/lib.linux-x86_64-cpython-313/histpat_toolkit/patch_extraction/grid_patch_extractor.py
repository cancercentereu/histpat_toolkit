import numpy as np

from ..geom import Rectangle
from ..tissue_detection.base_tissue_detector import BaseTissueDetector
from .base_patch_extractor import BasePatchExtractor


class GridPatchExtractor(BasePatchExtractor):
    def __init__(self, tissue_detector: BaseTissueDetector, patch_size: tuple[float, float], **kwargs):
        super().__init__(tissue_detector, **kwargs)
        self.patch_size = patch_size

    """
        This class makes patches of specified size using a grid pattern
    """

    def _extract_patches(self, mask: np.ndarray, scale: float, **kwargs) -> list[Rectangle]:
        ys, xs = mask.nonzero()
        w, h = self.patch_size
        ys = ys // h
        xs = xs // w
        items = set(zip(ys, xs))
        return [Rectangle(x * w, y * h, w, h) for y, x in items]
