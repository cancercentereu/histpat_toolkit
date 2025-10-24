"""
OpenSlide-based image pyramid implementation.

Provides an ImageSource implementation using OpenSlide for reading whole slide images.
"""


import cv2
import numpy as np

try:
    import openslide
    OPENSLIDE_AVAILABLE = True
except ImportError:
    OPENSLIDE_AVAILABLE = False

from ..geom import Rectangle, Shape, crop_shape_from_img


class OpenSlideImagePyramid:
    """
    Image pyramid implementation using OpenSlide for whole slide images.
    
    This class implements the ImageSource protocol and can be used with the
    segmentation pipeline and other algorithms.
    """

    def __init__(self, slide_path: str, tile_size: int = 256):
        """
        Initialize OpenSlide pyramid.
        
        Args:
            slide_path: Path to the slide file (e.g., .svs, .tiff, .ndpi)
            tile_size: Tile size for image operations (default: 256)
        """
        if not OPENSLIDE_AVAILABLE:
            raise ImportError(
                "openslide-python is required for OpenSlideImagePyramid. "
                "Install it with: pip install openslide-python"
            )
        
        self.slide = openslide.OpenSlide(slide_path)
        self._tile_size = tile_size
        
        # Cache properties
        self._size = (int(self.slide.dimensions[0]), int(self.slide.dimensions[1]))
        self._num_levels = self.slide.level_count
        
        # Try to extract MPP (microns per pixel) from metadata
        self._mpp = self._extract_mpp()
        
        # Try to extract magnification from metadata
        self._magnification = self._extract_magnification()

    def _extract_mpp(self) -> float | None:
        """Extract microns per pixel from slide metadata."""
        try:
            # Try OpenSlide MPP properties
            if openslide.PROPERTY_NAME_MPP_X in self.slide.properties:
                return float(self.slide.properties[openslide.PROPERTY_NAME_MPP_X])
            
            # Try TIFF resolution tags (used by some scanners)
            if 'tiff.XResolution' in self.slide.properties and 'tiff.ResolutionUnit' in self.slide.properties:
                resolution = float(self.slide.properties['tiff.XResolution'])
                unit = self.slide.properties['tiff.ResolutionUnit']
                
                # Convert to microns per pixel
                if unit == 'centimeter':
                    # pixels per cm -> microns per pixel
                    return 10000.0 / resolution
                elif unit == 'inch':
                    # pixels per inch -> microns per pixel
                    return 25400.0 / resolution
        except (KeyError, ValueError, AttributeError):
            pass
        
        return None

    def _extract_magnification(self) -> float | None:
        """Extract objective power (magnification) from slide metadata."""
        try:
            if openslide.PROPERTY_NAME_OBJECTIVE_POWER in self.slide.properties:
                return float(self.slide.properties[openslide.PROPERTY_NAME_OBJECTIVE_POWER])
        except (KeyError, ValueError, AttributeError):
            pass
        
        return None

    @property
    def size(self) -> tuple[int, int]:
        """Returns (width, height) of the full image at level 0."""
        return self._size

    @property
    def tile_size(self) -> int:
        """Returns the tile size used for operations."""
        return self._tile_size

    @property
    def num_levels(self) -> int:
        """Returns the number of pyramid levels."""
        return self._num_levels

    @property
    def mpp(self) -> float | None:
        """Returns microns per pixel at level 0."""
        return self._mpp

    @property
    def magnification(self) -> float | None:
        """Returns the objective magnification."""
        return self._magnification

    def get_scale_for(self, mpp: float | None = None, magnification: float | None = None) -> float:
        """
        Calculate the scale factor needed to achieve the desired MPP or magnification.
        
        Args:
            mpp: Target microns per pixel
            magnification: Target magnification
            
        Returns:
            Scale factor (e.g., 0.5 means half resolution)
        """
        if mpp is None and magnification is None:
            raise ValueError("At least one of mpp and magnification should be provided")

        if mpp is not None and self.mpp is not None:
            return self.mpp / mpp

        if magnification is not None and self.magnification is not None:
            return magnification / self.magnification

        raise ValueError("Cannot get scale for this pyramid: insufficient metadata")

    def get_level_for_scale(self, scale: float) -> tuple[int, float]:
        """
        Get the best pyramid level for a given scale.
        
        Args:
            scale: Desired scale factor
            
        Returns:
            Tuple of (level, adjusted_scale) where adjusted_scale accounts for the level downsample
        """
        if scale > 1:
            return 0, scale
        
        # Find the best level
        best_level = 0
        for level in range(self.num_levels):
            level_downsample = self.slide.level_downsamples[level]
            if level_downsample <= 1.0 / scale:
                best_level = level
            else:
                break
        
        level_downsample = self.slide.level_downsamples[best_level]
        adjusted_scale = scale * level_downsample
        
        return best_level, adjusted_scale

    def full_image(self, scale: float = 1.0, use_cache: bool = False) -> np.ndarray:
        """
        Returns the full image at the specified scale.
        
        Args:
            scale: Scale factor (1.0 = full resolution)
            use_cache: Ignored (kept for interface compatibility)
            
        Returns:
            Full image as RGB numpy array
        """
        level, adjusted_scale = self.get_level_for_scale(scale)
        
        # Get image at the selected level
        level_dimensions = self.slide.level_dimensions[level]
        pil_image = self.slide.read_region((0, 0), level, level_dimensions)
        
        # Convert RGBA to RGB
        img = np.array(pil_image.convert('RGB'))
        
        # Resize if needed
        if adjusted_scale != 1.0:
            target_size = (
                int(level_dimensions[0] * adjusted_scale),
                int(level_dimensions[1] * adjusted_scale)
            )
            img = cv2.resize(img, target_size, interpolation=cv2.INTER_LINEAR)
        
        return img

    def crop_rect(self, rect: Rectangle, scale: float = 1.0, allow_out_of_bounds: bool = False) -> np.ndarray:
        """
        Crop a rectangular region from the image at the specified scale.
        
        Args:
            rect: Rectangle to crop (at scale 1.0 coordinates)
            scale: Scale factor for the output
            allow_out_of_bounds: If True, pad out-of-bounds regions with white
            
        Returns:
            Cropped image as RGB numpy array
        """
        if rect is None or rect.area() == 0:
            raise ValueError("Rectangle is degenerated")
        
        level, adjusted_scale = self.get_level_for_scale(scale)
        level_downsample = self.slide.level_downsamples[level]
        
        # Convert rectangle to level coordinates
        level_rect = Rectangle(
            x=rect.x / level_downsample,
            y=rect.y / level_downsample,
            w=rect.w / level_downsample,
            h=rect.h / level_downsample,
            rot=rect.rot
        )
        
        # Get bounding box at level coordinates
        boundary = level_rect.integer_boundary()
        
        # Check bounds
        level_width, level_height = self.slide.level_dimensions[level]
        
        if not allow_out_of_bounds:
            if (boundary.x < 0 or boundary.y < 0 or 
                boundary.x + boundary.w > level_width or 
                boundary.y + boundary.h > level_height):
                raise ValueError("Rectangle is out of bounds")
        
        # Read the region
        # OpenSlide read_region uses level 0 coordinates
        location_level0 = (int(boundary.x * level_downsample), int(boundary.y * level_downsample))
        size = (int(boundary.w), int(boundary.h))
        
        pil_image = self.slide.read_region(location_level0, level, size)
        img = np.array(pil_image.convert('RGB'))
        
        # Handle out of bounds if needed
        if allow_out_of_bounds:
            # Pad image if it's smaller than expected (out of bounds)
            if img.shape[0] < boundary.h or img.shape[1] < boundary.w:
                padded = np.full((int(boundary.h), int(boundary.w), 3), 255, dtype=np.uint8)
                padded[:img.shape[0], :img.shape[1]] = img
                img = padded
        
        # Apply rotation if needed
        if rect.rot != 0:
            # Rotate around the rectangle center
            center = (boundary.w / 2, boundary.h / 2)
            rotation_matrix = cv2.getRotationMatrix2D(center, rect.rot, 1.0)
            img = cv2.warpAffine(img, rotation_matrix, (int(boundary.w), int(boundary.h)), 
                                borderValue=(255, 255, 255))
        
        # Resize if needed
        if adjusted_scale != 1.0:
            target_size = (
                max(1, int(boundary.w * adjusted_scale)),
                max(1, int(boundary.h * adjusted_scale))
            )
            img = cv2.resize(img, target_size, interpolation=cv2.INTER_LINEAR)
        
        return img

    def crop_shape(
        self, shape: Shape, scale: float = 1.0, allow_out_of_bounds: bool = False, fill_value: int = 255
    ) -> np.ndarray:
        """
        Crop an arbitrary shape from the image at the specified scale.
        
        Args:
            shape: Shape to crop (Circle, Ellipse, Polygon, etc.)
            scale: Scale factor for the output
            allow_out_of_bounds: If True, pad out-of-bounds regions
            fill_value: Value to fill outside the shape
            
        Returns:
            Cropped image as RGB numpy array with shape mask applied
        """
        bounding_rect = shape.bounding_box()
        cropped_rect = self.crop_rect(bounding_rect, scale, allow_out_of_bounds)
        
        # Adjust shape to cropped coordinates
        translated = shape.translate(-bounding_rect.x, -bounding_rect.y)
        if scale != 1.0:
            translated = translated.scale(scale)
        
        # Apply shape mask
        cropped = crop_shape_from_img(cropped_rect, translated, fill_value)
        
        return cropped

    def close(self):
        """Close the OpenSlide object and free resources."""
        if hasattr(self, 'slide') and self.slide is not None:
            self.slide.close()
            self.slide = None

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()

    def __del__(self):
        """Destructor to ensure slide is closed."""
        self.close()

