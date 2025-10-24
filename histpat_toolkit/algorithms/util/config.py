from typing import Any, Dict, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict


class FlexibleInputSize(BaseModel):
    max_surface: int


class ExactInputSize(BaseModel):
    width: int
    height: int


class FlexibleSquareInputSize(BaseModel):
    max_square_size: int


MicroscopyInputSize = Union[FlexibleInputSize, ExactInputSize, FlexibleSquareInputSize]


class MicroscopyConfig(BaseModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    model_type: Literal["microscopy"] = "microscopy"

    input_description: str = ""
    output_description: str = ""

    slide_magnification: Optional[float] = None
    slide_mpp: Optional[float] = None

    # use None to allow sending full image
    input_size: Union[MicroscopyInputSize, None] = None

    input_metadata: Union[Dict[str, Any], None] = None
    output_metadata: Union[Dict[str, Any], None] = None

    model_config = ConfigDict(protected_namespaces=())
