import cv2
from fastapi import APIRouter, HTTPException
import numpy as np
from pydantic import BaseModel
from typing import Tuple
from ..utils import  decode_image, encode_image
from ..state import Image, image_state
from typing import Tuple
router = APIRouter()

class ColorDetectionRequest(BaseModel):
    data: str
    color_lower: Tuple[int, int, int]
    color_upper: Tuple[int, int, int]

@router.post("/color_detection")
def detect_color(request: ColorDetectionRequest):
    try:
        result = color_detection(request.data, request.color_lower, request.color_upper)
        return {"status": "color detected", "image_data": result["image_data"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Color detection function
def color_detection(data: str, color_lower: Tuple[int, int, int], color_upper: Tuple[int, int, int]) -> dict:
    img = decode_image(data)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array(color_lower), np.array(color_upper))
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)
    
    cnts, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Debugging output
    print(f"Color lower bound: {color_lower}")
    print(f"Color upper bound: {color_upper}")
    print(f"Number of contours found: {len(cnts)}")
    
    if len(cnts) > 0:
      for cnt in  cnts:
        #c = max(cnts, key=cv2.contourArea)
        c = cnt
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        M = cv2.moments(c)
        if M["m00"] > 0:  # Avoid division by zero
            center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
            if radius > 10:
                cv2.circle(img, (int(x), int(y)), int(radius), (0, 255, 255), 2)
                cv2.circle(img, center, 5, (0, 0, 255), -1)
    
    return encode_image(img)
# # Color detection function
# def color_detection(data: str, color_lower: Tuple[int, int, int], color_upper: Tuple[int, int, int]) -> dict:
#     img = decode_image(data)
#     hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
#     mask = cv2.inRange(hsv, np.array(color_lower), np.array(color_upper))
#     mask = cv2.erode(mask, None, iterations=2)
#     mask = cv2.dilate(mask, None, iterations=2)
    
#     cnts, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     if len(cnts) > 0:
#         c = max(cnts, key=cv2.contourArea)
#         ((x, y), radius) = cv2.minEnclosingCircle(c)
#         M = cv2.moments(c)
#         if M["m00"] > 0:  # Avoid division by zero
#             center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
#             if radius > 10:
#                 cv2.circle(img, (int(x), int(y)), int(radius), (0, 255, 255), 2)
#                 cv2.circle(img, center, 5, (0, 0, 255), -1)
    
#     return encode_image(img)