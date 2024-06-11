


from fastapi import APIRouter, Depends, HTTPException
from ..state import image_state, Image
from ..utils import decode_image, object_classification
import torch

router = APIRouter()

@router.post("/object_classification")
def classify_object(image: Image):
        data = image.data if isinstance(image.data, str) else str(image.data)
        results,classes = object_classification(data)
        json_results = results_to_json(results,classes)
        return json_results


def results_to_json(results, classes):
    ''' Helper function for process_home_form()'''
    return [
        [
          {
          "class": int(pred[5]),
          "class_name": classes[int(pred[5])],
          "bbox": [int(x) for x in pred[:4].tolist()], #convert bbox results to int from float
          "confidence": float(pred[4]),
          }
        for pred in result
        ]
      for result in results.xyxy
  ]
   