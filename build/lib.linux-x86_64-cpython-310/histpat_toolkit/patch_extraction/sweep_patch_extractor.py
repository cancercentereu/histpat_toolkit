import numpy as np
from histpat_toolkit.geom import Rectangle

from ..geom import Rectangle
from .base_patch_extractor import BasePatchExtractor


class SweepPatchExtractor(BasePatchExtractor):
    """
    This class makes patches of specified size with a simple line sweep algorithm
    """

    def _extract_patches(
        self, mask: np.ndarray, scale: float, patch_size: tuple[int | float, int | float]
    ) -> list[Rectangle]:
        ys, xs = mask.nonzero()
        w, h = patch_size[0] * scale, patch_size[1] * scale
        result = []
        i = 0
        eps = 1e-6
        while i < len(ys):
            y = ys[i]
            start = i
            while i < len(ys) and ys[i] + 1 <= y + h + eps:
                i += 1
            end = i

            xb = sorted(xs[start:end])
            done_x = 0
            for x in xb:
                if x + 1 <= done_x + eps:
                    continue
                patch = Rectangle(x, y, w, h)
                result.append(patch)
                done_x = x + w

        return result
