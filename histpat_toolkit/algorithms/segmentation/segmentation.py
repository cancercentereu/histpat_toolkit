import concurrent.futures
import gc
import logging
import os

import cv2
import numpy as np

from ...geom import Rectangle, paste_rect_into_img
from ...tissue_detection import NNTissueDetector
from ..util.color import Color, get_colormap_lut
from ..util.config import ExactInputSize, FlexibleInputSize, MicroscopyConfig
from ..util.image_source import ImageSource
from ..util.modal_client import RunMode
from ..util.model_runner import SegmentationModelRunner
from ..util.patches import get_exact_size_patches, get_flexible_size_patches, scale_if_needed

DETECTOR_MPP = 10


def run_segmentation(
    image_source: ImageSource,
    roi: Rectangle,
    colors: list[Color],
    model_runner: SegmentationModelRunner,
    run_mode: str = "PREDICTION",
    config: MicroscopyConfig | None = None,
    overlap_factor: float = 0.1,
    batch_size: int = 10,
    class_weights: dict | None = None,
    save_classes: list[int] | None = None,
):
    """
    Run segmentation on an image source.
    
    Args:
        image_source: Source for image data (e.g., DZIPyramid or other ImageSource implementation)
        roi: Region of interest to segment
        colors: List of colors for the segmentation classes
        model_runner: Model runner for inference (e.g., ModalSegmentationModelRunner)
        run_mode: One of "PREDICTION", "PROBABILITIES", or "PREDICTION_MAX"
        config: Model configuration (if None, will be fetched from model_runner)
        overlap_factor: Overlap factor for flexible size patches
        batch_size: Batch size for model inference
        class_weights: Optional weights for each class (for PREDICTION_MAX mode)
        save_classes: Optional list of classes to save (if None, all classes are saved)
        
    Returns:
        Dictionary containing:
            - "scale": Scale factor used for the masks
            - "tile_size": Tile size of the image source
            - "masks": Dictionary mapping class keys to mask arrays
            - "lut": Color lookup table for visualization
            - "mask_offset": Offset of the mask relative to the original image
    """
    if class_weights is None:
        class_weights = {}
        
    run_mode_enum = RunMode[run_mode.upper()]
    logger = logging.getLogger()

    try:
        if config is None:
            config = model_runner.get_config()
        
        if not isinstance(config, MicroscopyConfig):
            config = MicroscopyConfig.model_validate(config)

        model_mpp = config.slide_mpp
        assert model_mpp is not None, "model mpp not provided"

        model_scale = image_source.get_scale_for(
            mpp=model_mpp,
            magnification=config.slide_magnification,
        )

        detector_mpp: float = DETECTOR_MPP
        detector = NNTissueDetector(
            tile_size=512,
            mpp=detector_mpp,
            opening_ksize=6,
            closing_ksize=10,
        )

        if isinstance(
            config.input_size,
            FlexibleInputSize,
        ):
            logger.info("Using flexible input size")
            max_input_model_area = int(config.input_size.max_surface)
            patches = get_flexible_size_patches(
                detector=detector,
                detector_mpp=detector_mpp,
                max_input_model_area=max_input_model_area,
                model_mpp=model_mpp,
                overlap_factor=overlap_factor,
                dzi_pyramid=image_source,
                roi=roi,
            )
        elif isinstance(
            config.input_size,
            ExactInputSize,
        ):
            logger.info("Using exact input size")
            patches = get_exact_size_patches(
                detector=detector,
                detector_mpp=detector_mpp,
                width=config.input_size.width,
                height=config.input_size.height,
                model_mpp=model_mpp,
                img_mpp=detector_mpp,
                dzi_pyramid=image_source,
                roi=roi,
            )
        else:
            raise NotImplementedError("Unsupported input size config: " + str(type(config.input_size)))

        logger.info(f"Found {len(patches)} patches")

        lut = get_colormap_lut(colors, save_classes)

        mask_surface = roi.area() * model_scale**2
        mask_max_dimension = max(roi.w, roi.h) * model_scale

        MAX_PIXELS = os.environ.get("SEGMENTATION_MAX_PIXELS")
        if MAX_PIXELS is None:
            # This code prevents us from running out of the RAM
            # Will use at most ~2GB of RAM (for the masks)
            logger.info("SEGMENTATION_MAX_PIXELS environmental variable not set (using default value to 2*10^9)")
            MAX_PIXELS = 2 * 1000 * 1000 * 1000
        else:
            MAX_PIXELS = int(MAX_PIXELS)

        if run_mode_enum == RunMode.PROBABILITIES:
            MAX_PIXELS //= len(colors)
        elif run_mode_enum == RunMode.PREDICTION_MAX:
            MAX_PIXELS //= max(color.key for color in colors) + 1

        MAX_DIMENSION = (1 << 15) - 1

        scale_diff = min(np.sqrt(MAX_PIXELS / mask_surface), MAX_DIMENSION / mask_max_dimension)

        if scale_diff < 1:
            logger.info(f"Masks too large to store in RAM - needs resizing {scale_diff=}")
        else:
            scale_diff = 1

        mask_roi = roi.scale(model_scale * scale_diff).integer_boundary()
        mask_size = [int(mask_roi.w), int(mask_roi.h)]
        mask_offset = [int(mask_roi.x), int(mask_roi.y)]

        def get_empty_mask():
            return np.zeros((mask_size[1], mask_size[0]), dtype="uint8")

        def crop_patch(patch):
            patch = patch.translate(roi.x * model_scale, roi.y * model_scale)
            return image_source.crop_rect(
                patch,
                scale=model_scale,
                allow_out_of_bounds=True,
            )

        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            input_images = executor.map(crop_patch, patches)

            invocation_run_mode = RunMode.PROBABILITIES if run_mode_enum == RunMode.PREDICTION_MAX else run_mode_enum

            output = model_runner.invoke(
                input_images,
                invocation_run_mode,
                batch_size=batch_size,
                executor=executor,
                logger=logger,
            )

            output_generator = (
                scale_if_needed(scale_diff, patch, patch_mask) for patch, patch_mask in zip(patches, output)
            )

            match run_mode_enum:
                case RunMode.PREDICTION:
                    mask = get_empty_mask()
                    for patch, patch_mask in output_generator:
                        assert isinstance(patch_mask, np.ndarray)
                        paste_rect_into_img(mask, patch_mask, patch, interpolation=cv2.INTER_NEAREST)
                    masks = {0: mask}
                case RunMode.PROBABILITIES:
                    segmentation_keys = [color.key for color in colors]
                    if save_classes is not None:
                        segmentation_keys = [key for key in segmentation_keys if key in save_classes]
                    logger.info(
                        f"Combining {len(patches)} patches into mask " f"for {len(segmentation_keys)} segmentations"
                    )
                    masks = {color_key: get_empty_mask() for color_key in segmentation_keys}
                    for patch, patches_masks in output_generator:
                        for color_key in segmentation_keys:
                            paste_rect_into_img(
                                masks[color_key],
                                patches_masks[color_key],
                                patch,
                                interpolation=cv2.INTER_NEAREST,
                                maximum=True,
                            )
                case RunMode.PREDICTION_MAX:
                    num_keys = max(color.key for color in colors) + 1
                    segmentation_keys = list(range(num_keys))
                    logger.info(
                        f"Combining {len(patches)} patches " f"into mask for {len(segmentation_keys)} segmentations"
                    )
                    masks = np.zeros((num_keys, mask_size[1], mask_size[0]), dtype="uint8")
                    for patch, patches_masks in output_generator:
                        for color_key in range(num_keys):
                            patch_mask = patches_masks[color_key]
                            if str(color_key) in class_weights:
                                patch_mask = (
                                    (patch_mask.astype("float32") * class_weights[str(color_key)])
                                    .clip(0, 255)
                                    .astype("uint8")
                                )
                            paste_rect_into_img(
                                masks[color_key],
                                patch_mask,
                                patch,
                                interpolation=cv2.INTER_NEAREST,
                                maximum=True,
                            )

                    mask = np.argmax(masks, axis=0)
                    del masks

                    gc.collect()

                    masks = {0: mask}

            logger.info("Done first part!")
            return {
                "scale": 1 / (model_scale * scale_diff),
                "tile_size": image_source.tile_size,
                "masks": masks,
                "lut": lut,
                "mask_offset": mask_offset,
            }

    except Exception as e:
        raise e
