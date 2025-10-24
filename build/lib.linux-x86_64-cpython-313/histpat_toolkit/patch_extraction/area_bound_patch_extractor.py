from math import ceil
from typing import Iterable, Iterator

import numpy as np

from ..geom import Rectangle
from ..tissue_detection.base_tissue_detector import BaseTissueDetector
from .anchorizers.base_tissue_anchorizer import BaseTissueAnchorizer
from .base_anchor_patch_extractor import BaseAnchorPatchExtractor
from .labelers.base_tissue_labeler import BaseTissueLabeler
from .modifiers.base_tissue_modifier import BaseTissueModifier
from .rectangularizers.base_rectangularizer import BaseRectangularizer


def _bound_area_of_patches(
    patches: Iterable[Rectangle],
    max_area: int | float,
) -> Iterator[Rectangle]:
    for patch in patches:
        patch_area = patch.area()
        if patch_area > max_area:
            # divide patch into two of equal sizes
            if patch.w > patch.h:
                patch.w = patch.w / 2
                diff = patch.norm_vec_x(len=patch.w)
            else:
                patch.h = patch.h / 2
                diff = patch.norm_vec_y(len=patch.h)
            yield patch
            patch = patch.translate(diff.x, diff.y)
            yield patch

        elif patch_area < max_area * 0.5:
            # extend patch so that it has the size of exactly the half of the max area
            temp_scale = ((max_area * 0.5) / patch_area) ** 0.5
            patch = patch.scale_around_center(temp_scale)
            yield patch
        else:
            yield patch


class AreaBoundPatchExtractor(BaseAnchorPatchExtractor):
    """
    This class does binary search over the area of the patches
    It assumes that increasing some parameter will not decrease
    the area of the patches
    """

    def __init__(
        self,
        tissue_detector: BaseTissueDetector,
        tissue_modifier: BaseTissueModifier,
        tissue_anchorizer: BaseTissueAnchorizer,
        tissue_labeler: BaseTissueLabeler,
        tissue_rectangularizer: BaseRectangularizer,
        max_area: int = 400 * 400,
        eps=1e-4,
    ) -> None:
        super().__init__(
            tissue_detector=tissue_detector,
            tissue_modifier=tissue_modifier,
            tissue_anchorizer=tissue_anchorizer,
            tissue_labeler=tissue_labeler,
            tissue_rectangularizer=tissue_rectangularizer,
            max_area=max_area,
        )
        self.eps = eps

    def _extract_patches(
        self,
        tissue_mask: np.ndarray,
        scale: float | None,
        max_patch_area: int | float,
        iters: int = 5,
        **kwargs,
    ) -> list[Rectangle]:
        """
        Parameters:
            tissue_mask: np.ndarray
                The image to extract the patches from
            scale: float
                The scale by which the original extract_patches(img) was scaled by
                this means, we may want to rescale some parameters in the anchorizer
                or other classes
            max_patch_area: int | float
                The maximum area of the patches (measured in pixels)
            iters: int
                The number of iterations to do the binary search
            **kwargs:
                The parameters to pass to the anchorizer, each parameter range should
                be passed either by a tuple or by a list of two elements

            WARNING:
            1) THIS FUNCTION DOES NOT GUARANTEE THAT ALL THE
            RECTANGLES WILL HAVE AREA LESS THAN max_area
            2) ALSO IT MODIFIES ANCHIRIZER PARAMETERS
        """

        parameters_ranges = dict()
        for key, (lo, hi) in kwargs.items():
            if not hasattr(self.tissue_anchorizer, key) or key.startswith("_"):
                raise ValueError(f"Invalid parameter passed to the anchorizer: {key}")
            assert lo <= hi, f"Invalid range for parameter {key}: {lo} > {hi}"
            if isinstance(lo, int) and isinstance(hi, int):
                parameters_ranges[key] = (ceil(lo * scale), round(hi * scale))
            else:
                parameters_ranges[key] = (lo * scale, hi * scale)

        if scale is not None:
            max_patch_area = max_patch_area * scale * scale

        def set_anchorizer_params(parameters_values):
            something_changed = False
            for key, new_val in parameters_values.items():
                old_val = getattr(self.tissue_anchorizer, key)
                if isinstance(old_val, int):
                    if getattr(self.tissue_anchorizer, key) != new_val:
                        something_changed = True
                else:
                    if abs(old_val - new_val) > self.eps:
                        something_changed = True
                self.tissue_anchorizer.__setattr__(key, new_val)

            return something_changed

        for iteration in range(iters):
            parameters_values = dict()
            for key, (lo, hi) in parameters_ranges.items():
                if isinstance(lo, int):
                    parameters_values[key] = (lo + hi) // 2
                else:
                    parameters_values[key] = (lo + hi) / 2
            if not set_anchorizer_params(parameters_values):
                # when no parameter changed from previous iteration,
                # we can (quite surely) finish the binary search
                break

            rectangles = super()._extract_patches(
                tissue_mask=tissue_mask,
                scale=1,
            )

            if len(rectangles) == 0:
                return list()

            good_rectangles = sum([rect.area() <= max_patch_area for rect in rectangles])
            if good_rectangles < len(rectangles) * (2 / 3):
                parameters_ranges = {key: (lo, parameters_values[key]) for key, (lo, _) in parameters_ranges.items()}
            else:
                parameters_ranges = {key: (parameters_values[key], hi) for key, (_, hi) in parameters_ranges.items()}

        set_anchorizer_params({key: lo for key, (lo, hi) in parameters_ranges.items()})

        result = super()._extract_patches(
            tissue_mask=tissue_mask,
            scale=1,
        )
        result = list(_bound_area_of_patches(result, max_patch_area))
        return result
