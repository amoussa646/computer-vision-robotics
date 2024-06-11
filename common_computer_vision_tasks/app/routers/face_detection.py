from fastapi import APIRouter, Depends, HTTPException
from ..utils import face_detection, face_smile_eye_detection
from ..state import Image, image_state

router = APIRouter()

@router.post("/face_detection")
def detect_face(data: dict):
    try:
        result = face_detection(data["data"])
        image_state.image = result["image_data"]
        return {"status": "face detected","image_data":result["image_data"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/face_smile_eye_detection")
def detect_face_smile_eye(data: dict):
    try:
        result = face_smile_eye_detection(data["data"])
        image_state.image = result["image_data"]
        return {"status": "face, smile, and eye detected","image_data":result["image_data"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
