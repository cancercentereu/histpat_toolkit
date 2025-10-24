"""
Helper functions for local segmentation workflows.

This module provides utility functions for saving segmentation results to disk.
These are designed to be used after running the segmentation pipeline.
"""

from pathlib import Path

import numpy as np
from PIL import Image

from ...geom import Rectangle
from ..util.image_source import ImageSource


def save_segmentation_results(
    result: dict,
    image_source: ImageSource,
    roi: Rectangle,
    output_path: Path,
    transparency_level: float = 0.5,
    save_classes: list[int] | None = None,
) -> dict:
    """
    Save segmentation results to disk as PNG files.
    
    Creates two types of images for each class:
    1. segmentation_mask_{key}.png - The colored mask itself
    2. segmentation_overlay_{key}.png - The mask overlaid on the original image
    
    Args:
        result: Segmentation result dictionary from run_segmentation (contains "masks", "lut", etc.)
        image_source: Image source used for segmentation (e.g., DZIPyramid)
        roi: Region of interest that was segmented
        output_path: Directory to save output images
        transparency_level: Transparency level for overlay (0.0 = fully transparent, 1.0 = fully opaque original image)
        save_classes: Optional list of class keys to save (if None, all classes are saved)
        
    Returns:
        Dictionary with "output_path" key
    """
    assert 0.0 <= transparency_level <= 1.0, "transparency_level must be between 0 and 1"
    
    output_path.mkdir(parents=True, exist_ok=True)
    
    keys = result["masks"].keys()
    if save_classes is not None:
        keys = [key for key in keys if key in save_classes]

    for key in keys:
        # Save colored mask
        colored_mask = result["lut"][result["masks"][key]]
        colored_image = Image.fromarray(colored_mask)
        colored_image.save(output_path / f"segmentation_mask_{key}.png")

        # Create overlay image
        roi_image = image_source.crop_rect(roi)

        colored_mask_rgb = colored_mask[..., :3]

        colored_mask_pil = Image.fromarray(colored_mask_rgb)
        mask_resized = colored_mask_pil.resize((roi_image.shape[1], roi_image.shape[0]), Image.Resampling.NEAREST)
        colored_mask_resized = np.array(mask_resized)

        segmentation_pil = Image.fromarray(result["masks"][key].astype(np.uint8))
        segmentation_resized = segmentation_pil.resize(
            (roi_image.shape[1], roi_image.shape[0]), Image.Resampling.NEAREST
        )
        segmented_areas = np.array(segmentation_resized) > 0

        roi_image[segmented_areas] = (
            roi_image[segmented_areas] * transparency_level
            + colored_mask_resized[segmented_areas] * (1 - transparency_level)
        ).astype(np.uint8)

        Image.fromarray(roi_image).save(output_path / f"segmentation_overlay_{key}.png")

    return {"output_path": str(output_path)}
