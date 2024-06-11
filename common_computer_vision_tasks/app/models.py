from typing import Optional
from pydantic import BaseModel

class Image(BaseModel):
    data: str
    timestamp: Optional[str]
