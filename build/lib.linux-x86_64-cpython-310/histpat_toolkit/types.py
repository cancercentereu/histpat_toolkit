from pydantic.dataclasses import dataclass

@dataclass
class Tile:
  x: int
  y: int
  level: int

@dataclass
class ColorMap:
  id: str
  name: str
  # todo

  @staticmethod
  def from_graphql(data):
    return ColorMap(
      id=data['id'],
      name=data['name']
    )
  
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