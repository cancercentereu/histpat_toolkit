import cv2
import numpy as np

from ...geom import Rectangle
from ...image_pyramid.base_image_pyramid import BaseImagePyramid
from ...patch_extraction import (
    AreaBoundPatchExtractor,
    SweepPatchExtractor,
)
from ...patch_extraction.anchorizers import RadiusTissueAnchorizer
from ...patch_extraction.labelers import BFSNearestTissueLabeler
from ...patch_extraction.modifiers import TrivialTissueModifier
from ...patch_extraction.rectangularizers import MinAreaRectangularizer
from ...tissue_detection.nn_tissue_detector import NNTissueDetector


def scale_if_needed(
    scale_diff: float,
    patch: Rectangle,
    patch_mask: np.ndarray | list[np.ndarray],
):
    if scale_diff == 1:
        return patch, patch_mask
    if isinstance(patch_mask, np.ndarray):
        patch_mask = cv2.resize(
            patch_mask,
            (
                int(patch_mask.shape[0] * scale_diff),
                int(patch_mask.shape[1] * scale_diff),
            ),
            interpolation=cv2.INTER_NEAREST,
        )
    else:
        patch_mask = [
            cv2.resize(
                mask,
                (
                    int(mask.shape[0] * scale_diff),
                    int(mask.shape[1] * scale_diff),
                ),
                interpolation=cv2.INTER_LINEAR,
            )
            for mask in patch_mask
        ]

    patch = patch.scale(scale_diff)
    return patch, patch_mask


def get_flexible_size_patches(
    detector: NNTissueDetector,
    detector_mpp: float,
    max_input_model_area: int,
    model_mpp: float,
    overlap_factor: float,
    dzi_pyramid: BaseImagePyramid,
    roi: Rectangle,
) -> list[Rectangle]:
    patch_extractor = AreaBoundPatchExtractor(
        detector,
        TrivialTissueModifier(),
        RadiusTissueAnchorizer(),
        BFSNearestTissueLabeler(),
        MinAreaRectangularizer(overlap=overlap_factor),
        max_area=1000 * 1000,
    )

    max_area_of_patches = max_input_model_area / (detector_mpp / model_mpp) ** 2
    sqrt_of_area = int(np.sqrt(max_area_of_patches))

    img_mpp = detector_mpp
    img_scale = dzi_pyramid.get_scale_for(mpp=img_mpp)

    patches = patch_extractor.extract_patches(
        img=dzi_pyramid.crop_rect(roi.scale(img_scale), scale=img_scale, allow_out_of_bounds=True),
        img_mpp=img_mpp,
        max_patch_area=max_area_of_patches,
        iters=10,
        radius=(
            max(
                1,
                sqrt_of_area // 4,
            ),
            sqrt_of_area * 2,
        ),
    )
    if model_mpp != img_mpp:
        # scaling back to the slide scale
        patches = [patch.scale(img_mpp / model_mpp) for patch in patches]

    return patches


def get_exact_size_patches(
    detector: NNTissueDetector,
    detector_mpp: float,
    width: int,
    height: int,
    model_mpp: float,
    img_mpp: float,
    dzi_pyramid: BaseImagePyramid,
    roi: Rectangle,
) -> list[Rectangle]:
    patch_extractor = SweepPatchExtractor(
        detector,
        max_area=10 * 1000 * 1000,
    )

    img_scale = dzi_pyramid.get_scale_for(mpp=img_mpp)

    model_patch_size = (width, height)

    patch_scale = detector_mpp / model_mpp
    patch_size = (model_patch_size[0] / patch_scale, model_patch_size[1] / patch_scale)

    patches = patch_extractor.extract_patches(
        img=dzi_pyramid.crop_rect(roi.scale(img_scale), scale=img_scale, allow_out_of_bounds=True),
        img_mpp=img_mpp,
        patch_size=patch_size,
    )

    patches = [patch.scale(patch_scale) for patch in patches]
    for patch in patches:
        # make sure that the patch is exactly the size of the model patch
        patch.w = model_patch_size[0]
        patch.h = model_patch_size[1]

    return patches
