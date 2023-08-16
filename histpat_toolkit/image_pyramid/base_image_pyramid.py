import numpy as np
from math import log2, ceil
from abc import ABC, abstractmethod
import requests
import cv2

from concurrent.futures import ThreadPoolExecutor

from ..geom import Rectangle
from ..dzi_file import div_up


class BaseImagePyramid(ABC):

    @property
    @abstractmethod
    def num_channels(self) -> int:
        pass

    @property
    @abstractmethod
    def num_levels(self) -> int:
        pass

    @property
    @abstractmethod
    def size(self) -> tuple[int, int]:
        pass

    @property
    @abstractmethod
    def tile_size(self) -> int:
        pass

    @abstractmethod
    def get_tile_url(self, level: int, x: int, y: int) -> str | None:
        pass

    def get_level_for_scale(self, scale: float) -> tuple[int, float]:
        """
        This function return the level of the pyramid and the scale of the level
        for given scale. For example, if scale is 0.3, the function should return
        (1, 0.6) for level 1 and scale 0.6.

        Returns:
            tuple[int, float]: level and scale of the level (scale should always be in the range (0.5, 1]
            except for the cases when output level is higher than maximal available level)
        """

        assert scale > 0

        if scale > 1:
            return 0, scale

        level = min(-ceil(log2(scale)), self.num_levels - 1)
        return level, scale * 2 ** level

    @property
    def default_pixel_fill(self) -> np.uint8:
        if self.num_channels == 3:
            return 255
        return 0

    def get_default_tile(self, level: int, x: int, y: int) -> np.ndarray:
        """ This function returns a default, white tile for the image pyramid """

        width, height = self.size

        scale = 1 / (2 ** level)
        width, height = ceil(width * scale), ceil(height * scale)

        def adjust_tile_size(coordinate: int, dimension_size: int) -> int:
            tile_start = coordinate * self.tile_size
            if tile_start >= dimension_size:
                raise ValueError("Tile is out of bounds")
            return min(self.tile_size, dimension_size - tile_start)

        tile_width = adjust_tile_size(x, width)
        tile_height = adjust_tile_size(y, height)

        return np.full((tile_height, tile_width, self.num_channels), fill_value=self.default_pixel_fill, dtype=np.uint8)

    def get_tile(self, level: int, x: int, y: int, tries_left: int = 10) -> np.ndarray:
        """
        Fetches tile from url and returns as numpy array.
        If there requests fails with status code 401, 403 or 404, return default tile.
        If the requests fails with status code 50x, retry up to 10 times
        """

        tile_url = self.get_tile_url(level, x, y)
        response = requests.get(tile_url)

        if 200 <= response.status_code < 300:
            data = np.frombuffer(response.content, dtype=np.uint8)
            img = cv2.imdecode(data, cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return img

        if 500 <= response.status_code < 510:
            if tries_left <= 1:
                response.raise_for_status()
            return self.get_tile(level, x, y, tries_left - 1)

        if response.status_code in [401, 403, 404]:
            return self.get_default_tile(level, x, y)

        response.raise_for_status()

    def full_image(self, scale: float = 1.0) -> np.ndarray:
        level, scale = self.get_level_for_scale(scale)
        full_x, full_y = ceil(
            self.size[0] / 2 ** level), ceil(self.size[1] / 2 ** level)

        ret = np.zeros((full_y, full_x, self.num_channels), dtype=np.uint8)

        number_of_columns = div_up(full_x, self.tile_size)
        number_of_rows = div_up(full_y, self.tile_size)

        def handle_tile(col, row):
            col_begin = col * self.tile_size
            row_begin = row * self.tile_size

            tile = self.get_tile(level, col, row)
            tile = crop_tile(tile, tile.shape, full_x,
                             full_y, row_begin, col_begin)

            ret[row_begin:row_begin + tile.shape[0],
                col_begin:col_begin + tile.shape[1]] = tile
            return

        with ThreadPoolExecutor() as executor:
            results = [[executor.submit(handle_tile, col, row)
                        for col in range(number_of_columns)]
                       for row in range(number_of_rows)]
            for row_of_results in results:
                for result in row_of_results:
                    result.result()

        return cv2.resize(ret, (int(full_x * scale), int(full_y * scale)))

    def crop_rect(self, rect: Rectangle, scale: float = 1.0, allow_out_of_bounds: bool = False) -> np.ndarray:
        """
        Cropping rectanle with possible rotation
        We think of it as of first scaling the full image and then cropping the exact rectange
        """

        if rect is None or rect.area() == 0:
            raise ValueError("Rectangle is degenerated")

        level, scale = self.get_level_for_scale(scale)

        full_x, full_y = ceil(
            self.size[0] / 2 ** level), ceil(self.size[1] / 2 ** level)

        # we increase the size of the rectangle, but in the end we will rescale it to the original size
        rect = rect.scale(1 / scale)


        boundary = rect.integer_boundary()
        assert boundary is not None
        
        last_x, last_y = boundary.x + boundary.w - 1, boundary.y + boundary.h - 1

        def in_bounds(pos, dim):
            return 0 <= pos < dim

        if not allow_out_of_bounds and \
            not (in_bounds(boundary.x, full_x) and in_bounds(boundary.y, full_y) and
                 in_bounds(last_x, full_x) and in_bounds(last_y, full_y)):
            raise ValueError(
                f"Rectangle is out of bounds {boundary = } {full_x = } {full_y = }")

        boundary = boundary.intersection(Rectangle(0, 0, full_x, full_y))
        # Case when boundary does not touch even touch the image
        if (boundary is None) or (boundary.area() == 0):
            return np.full((int(rect.h), int(rect.w), self.num_channels), fill_value=self.default_pixel_fill).astype(np.uint8)
        
        last_x, last_y = boundary.x + boundary.w - 1, boundary.y + boundary.h - 1

        boundary_pixels = np.zeros(
            (boundary.h, boundary.w, self.num_channels), dtype=np.uint8)

        def tile_id(pos):
            return pos // self.tile_size

        def handle_tile(col, row):
            col_begin = col * self.tile_size
            row_begin = row * self.tile_size

            tile = self.get_tile(level, col, row)
            tile = crop_tile(tile, tile.shape, full_x,
                             full_y, row_begin, col_begin)

            intersection = boundary.intersection(
                Rectangle(col_begin, row_begin, self.tile_size, self.tile_size))
            assert intersection is not None and intersection.area() > 0

            y_range, x_range = \
                np.array([intersection.y, intersection.y + intersection.h]), \
                np.array([intersection.x, intersection.x + intersection.w])

            boundary_pixels[slice(*(y_range - boundary.y)),
                            slice(*(x_range - boundary.x))] = \
                tile[slice(*(y_range - row_begin)),
                     slice(*(x_range - col_begin))]
            return

        with ThreadPoolExecutor() as executor:
            results = [[executor.submit(handle_tile, col, row)
                        for col in range(tile_id(boundary.x), tile_id(last_x) + 1)]
                       for row in range(tile_id(boundary.y), tile_id(last_y) + 1)]

            for row_of_results in results:
                for result in row_of_results:
                    result.result()

        pts1 = np.float32([[point[0] - boundary.x, point[1] - boundary.y]
                          for point in rect.points()[:3]])

        # rescaling rect back to the desired output size
        rect = rect.scale(scale)
        pts2 = np.float32([[0, 0], [rect.w, 0], [rect.w, rect.h]])
        M = cv2.getAffineTransform(pts1, pts2)

        return cv2.warpAffine(boundary_pixels, M, (int(rect.w), int(rect.h)),
                              borderMode=cv2.BORDER_CONSTANT,
                              borderValue=(self.default_pixel_fill,)*self.num_channels).astype(np.uint8)


def crop_tile(tile: np.ndarray, tile_shape: tuple[int, int], dim_x: int, dim_y: int, row_begin: int, col_begin: int) -> np.ndarray:
    """
    This function crops the tile not to escape from the (dim_y, dim_x) rectangle
    It raises an error if the tile has any pixel with negative coordinates
    """

    if row_begin < 0 or col_begin < 0:
        raise ValueError("Tile starts with negative coordinates")

    tile_x = min(dim_x - col_begin, tile_shape[1])
    tile_y = min(dim_y - row_begin, tile_shape[0])

    if tile_x < tile_shape[1] or tile_y < tile_shape[0]:
        return tile[:tile_y, :tile_x]
    return tile
