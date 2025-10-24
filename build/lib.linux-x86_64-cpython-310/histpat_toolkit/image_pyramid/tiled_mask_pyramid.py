from .base_image_pyramid import BaseImagePyramid
from ..dzi_file import DZIFile
from ..types import TiledMaskPyramidInfo
from math import floor
from functools import cached_property
import cv2

class TiledMaskPyramid(BaseImagePyramid):
    def __init__(self, dzi_file: DZIFile, tile_mask_info: TiledMaskPyramidInfo) -> None:
        super().__init__()

        self.dzi_file = dzi_file
        self.tile_mask_info = tile_mask_info

    @property
    def num_channels(self) -> int:
        return 4
    
    @cached_property
    def num_levels(self) -> int:
        if len(self.tile_mask_info.tiles) == 0:
            return self.dzi_file.levels
        return max([tile.level for tile in self.tile_mask_info.tiles]) + 1
    
    @property
    def size(self) -> tuple[int, int]:
        return (floor(self.dzi_file.width / self.scale), floor(self.dzi_file.height / self.scale))
    
    @property
    def tile_size(self) -> int:
        return self.dzi_file.tile_size
    
    @property
    def scale(self) -> float:
        return self.tile_mask_info.scale
    
    @property
    def magnification(self) -> float | None:
        mag = self.dzi_file.properties.get('magnification', None)
        if mag is not None:
            return mag / self.scale
        return None
    
    @property
    def mpp(self) -> float | None:
        mpp = self.dzi_file.properties.get('mpp', None)
        if mpp is not None:
            return mpp * self.scale
        return None
    
    @property
    def interpolation(self) -> int:
        return cv2.INTER_NEAREST

    def get_tile_url(self, level: int, x: int, y: int) -> str | None:
        """ Returns tile url. It reverses level naming
        so that 0 is the level with original resolution
        while level num_levels - 1 is the level with the single pixel"""

        tile_url = self.tile_mask_info.tiles_url + \
            ('' if self.tile_mask_info.tiles_url.endswith('/') else '/') + \
            f'{level}_{x}_{y}.png'

        return tile_url
