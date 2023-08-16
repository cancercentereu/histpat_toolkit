from .base_image_pyramid import BaseImagePyramid
from ..dzi_file import DZIFile


class DZIPyramid(BaseImagePyramid):
    def __init__(self, dzi_file: DZIFile) -> None:
        self.dzi_file = dzi_file

    @property
    def num_channels(self) -> int:
        if self.dzi_file.format == 'png':
            return 4
        return 3

    @property
    def num_levels(self) -> int:
        return self.dzi_file.levels

    @property
    def size(self) -> tuple[int, int]:
        return (self.dzi_file.width, self.dzi_file.height)

    @property
    def tile_size(self) -> int:
        return self.dzi_file.tile_size

    def get_tile_url(self, level: int, x: int, y: int) -> str | None:
        """ Returns tile url. It reverses level naming
        so that 0 is the level with original resolution
        while level num_levels - 1 is the level with the single pixel"""

        tile_url = self.dzi_file.files_url + \
            ('' if self.dzi_file.files_url.endswith('/') else '/') + \
            f'{self.num_levels - 1 - level}/{x}_{y}.{self.dzi_file.format}'

        return tile_url
