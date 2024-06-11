from pydantic import BaseModel

class HumanDetectionResponse(BaseModel):
    status: str
    bounding_box: list = None

class ObjectClassificationResponse(BaseModel):
    label: str
    confidence: float
