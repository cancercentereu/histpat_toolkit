import xml.etree.ElementTree as ET
import requests
from typing import cast


class DZIFile:
    def __init__(self, url, properties={}):
        r = requests.get(url)
        root = ET.fromstring(r.text)
        assert (root.tag.endswith('Image'))

        tile_size = root.get('TileSize')
        assert tile_size is not None, 'TileSize is required'
        self.tile_size = int(tile_size)

        format = root.get('Format')
        assert format is not None, 'Format is required'
        self.format = format

        overlap = root.get('Overlap')
        if overlap is not None:
            self.overlap = int(overlap)
        else:
            self.overlap = 0

        size = root.find('./{*}Size')
        assert size is not None, 'Size is required'

        width = size.get('Width')
        assert width is not None, 'Width is required'
        height = size.get('Height')
        assert height is not None, 'Height is required'

        self.width = int(width)
        self.height = int(height)

        self.files_url = url[:-4] + '_files'  # url - '.dzi' + '_files'
        self.levels = (max(self.width, self.height) - 1).bit_length() + 1
        self.properties = properties

    def _length_at_level(self, length, level):
        lower = self.levels - level - 1
        mask = (1 << lower) - 1
        return (length + mask) >> lower

    def size_at_level(self, level):
        return (self._length_at_level(self.width, level),
                self._length_at_level(self.height, level))

    def tiles_at_level(self, level):
        width, height = self.size_at_level(level)
        return div_up(width, self.tile_size), div_up(height, self.tile_size)

    def rounded_tiles_at_level(self, level):
        width, height = self.size_at_level(level)
        return (
            max(1, round(width / self.tile_size)),
            max(1, round(height / self.tile_size))
        )

    def tile_url(self, level, x, y):
        return f'{self.files_url}/{level}/{x}_{y}.{self.format}'
    
    def tile_url_template(self):
        return f"{self.files_url}/" + "{level}/{x}_{y}." + f"{self.format}"


def div_up(a: int, b: int) -> int:
    # division with rounding up
    return (a + b - 1) // b
