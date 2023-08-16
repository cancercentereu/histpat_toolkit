import xml.etree.ElementTree as ET
import requests
from typing import cast


class DZIFile:
    def __init__(self, url, properties=None):
        r = requests.get(url)
        root = ET.fromstring(r.text)
        assert (root.tag.endswith('Image'))

        self.tile_size = int(root.get('TileSize'))  # type: ignore
        self.format = root.get('Format')

        size = root.find('./{*}Size')
        self.width = int(size.get('Width'))  # type: ignore
        self.height = int(size.get('Height'))  # type: ignore

        self.files_url = url[:-4] + '_files'  # url - '.dzi' + '_files'
        self.levels = (max(self.width, self.height) - 1).bit_length() + 1
        self.properties = properties
        if self.properties is None or self.properties == {}:
            self.properties = get_properties_from_file(url)

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


def get_properties_from_file(dzi_url: str):
    files_url = dzi_url[:-4] + '_files'

    properties_urls = [files_url + '/properties.json',
                       dzi_url[:-4] + '_properties.json']
    for properties_url in properties_urls:
        r = requests.get(properties_url)
        if 200 <= r.status_code < 300:
            return r.json()

    # try to load vips-properties.xml (for old files)
    properties = {}
    properties_url = files_url + '/vips-properties.xml'
    r = requests.get(properties_url)
    if 200 <= r.status_code < 300:
        root = ET.fromstring(r.text)
        mpp = root.find(
            ".//{*}property/{*}name[.='openslide.mpp-x']/../{*}value")
        if mpp is not None:
            properties['mpp'] = float(mpp.text)  # type: ignore
        magnification = root.find(
            ".//{*}property/{*}name[.='openslide.objective-power']/../{*}value")
        if magnification is not None:
            properties['magnification'] = int(
                magnification.text)  # type: ignore
    return properties


def div_up(a: int, b: int) -> int:
    # division with rounding up
    return (a + b - 1) // b
