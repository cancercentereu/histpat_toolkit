from pydantic.dataclasses import dataclass


@dataclass
class Tile:
    x: int
    y: int
    level: int


@dataclass
class TiledMaskPyramidInfo:
    tiles: list[Tile]
    scale: float
    tiles_url: str
    tile_size: int


@dataclass
class SlideProperties:
    mpp: float | None
    magnification: float | None
