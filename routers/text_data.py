from pydantic import BaseModel
from pydantic.typing import Optional


class TextData(BaseModel):
    model_name: str
    task: str
    texts: list[str]
    feature: Optional[str]
