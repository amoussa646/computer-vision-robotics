from typing import Optional
from pydantic import BaseModel

class Image(BaseModel):
    data: str
    timestamp: Optional[str]

class ImageState:
    def __init__(self):
        self.image = Image(data="", timestamp="")

image_state = ImageState()
