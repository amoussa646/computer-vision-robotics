from fastapi import FastAPI
from app.routers import human_detection, object_classification, face_detection,color_detection
import torch

app = FastAPI()

app.include_router(human_detection.router, prefix="/human_detection", tags=["Human Detection"])
app.include_router(object_classification.router, prefix="/object_classification", tags=["Object Classification"])
app.include_router(face_detection.router, prefix="/face_detection", tags=["face_detection"])
app.include_router(color_detection.router, prefix="/color_detection", tags=["color_detection"])

