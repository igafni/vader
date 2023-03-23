from pydantic import BaseModel
from pydantic.typing import Optional


class ImageData(BaseModel):
    model_name: str
    task: str
    images: list[str]
    feature: Optional[str]
