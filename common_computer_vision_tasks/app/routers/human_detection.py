from fastapi import APIRouter, Depends, HTTPException
from ..state import Image, image_state
from ..utils import decode_image, human_detection
import torch

router = APIRouter()

@router.post("/human_detection")
def detect_human(image: Image):
    
        data = image.data if isinstance(image.data, str) else str(image.data)
        result = human_detection(data)
        if result:
            image_state.image = Image(data=data, timestamp=image.timestamp)
            return {"status": result[0], "image_data": data, "bounding_box": result[1]}
        else:
            return {"status": "not_found", "image_data": data, "bounding_box": None}
    