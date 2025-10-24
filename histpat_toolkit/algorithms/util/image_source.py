from typing import Protocol

import numpy as np

from ...geom import Rectangle, Shape


class ImageSource(Protocol):
    """
    Protocol (interface) for image sources that can provide image data at different scales.
    This allows the segmentation pipeline to work with different image sources
    (pyramids, local files, in-memory arrays, etc.)
    
    Any class that implements these methods can be used as an ImageSource.
    """

    @property
    def size(self) -> tuple[int, int]:
        """Returns the (width, height) of the full image at scale 1.0"""
        ...

    @property
    def tile_size(self) -> int:
        """Returns the tile size used by this image source"""
        ...

    def get_scale_for(self, mpp: float | None = None, magnification: float | None = None) -> float:
        """
        Calculate the scale factor needed to achieve the desired MPP or magnification.
        At least one of mpp or magnification must be provided.
        """
        ...

    def full_image(self, scale: float = 1.0, use_cache: bool = False) -> np.ndarray:
        """
        Returns the full image at the specified scale.
        """
        ...

    def crop_rect(self, rect: Rectangle, scale: float = 1.0, allow_out_of_bounds: bool = False) -> np.ndarray:
        """
        Crop a rectangular region from the image at the specified scale.
        """
        ...

    def crop_shape(
        self, shape: Shape, scale: float = 1.0, allow_out_of_bounds: bool = False, fill_value: int = 255
    ) -> np.ndarray:
        """
        Crop an arbitrary shape from the image at the specified scale.
        """
        ...

