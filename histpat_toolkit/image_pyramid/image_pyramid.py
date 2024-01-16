from .base_image_pyramid import BaseImagePyramid

class ImagePyramid(BaseImagePyramid):
    def __init__(self, levels: int, width: int, height: int, tile_size: int, tiles_url: str,
                 magnification: float | None = None, mpp: float | None = None) -> None:
        super().__init__()
        self.levels = levels
        self.width = width
        self.height = height
        self._tile_size = tile_size
        self.tiles_url = tiles_url
        self._magnification = magnification
        self._mpp = mpp

    @property
    def format(self) -> str:
        return self.tiles_url.split('.')[-1]
    
    @property
    def num_channels(self) -> int:
        if self.format == 'png':
            return 4
        return 3

    @property
    def num_levels(self) -> int:
        return self.levels

    @property
    def size(self) -> tuple[int, int]:
        return (self.width, self.height)

    @property
    def tile_size(self) -> int:
        return self._tile_size
    
    @property
    def magnification(self) -> float | None:
        return self._magnification
    
    @property
    def mpp(self) -> float | None:
        return self._mpp

    def get_tile_url(self, level: int, x: int, y: int) -> str | None:
        """ Returns tile url. It reverses level naming so that
        0 is the level with original resolution while the level
        number num_levels - 1 is the level with the single pixel
        """

        url_base = self.tiles_url[:self.tiles_url.find('{level}/{x}_{y}')]
        return url_base + f'{self.num_levels - 1 - level}/{x}_{y}.{self.format}'
