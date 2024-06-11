import base64
import os
import sys
import PIL
import cv2
import numpy as np
import torch

model = None
human_model = None
object_model = None
PERSON_CLASS_LABEL = 0

base_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, './yolov5')

face_cascade_path = os.path.join(base_dir, 'Cascades/haarcascade_frontalface_default.xml')
eye_cascade_path = os.path.join(base_dir, 'Cascades/haarcascade_eye.xml')
smile_cascade_path = os.path.join(base_dir, 'Cascades/haarcascade_smile.xml')
print(f"Face cascade path: {face_cascade_path}")
print(f"Eye cascade path: {eye_cascade_path}")
print(f"Smile cascade path: {smile_cascade_path}")

face_cascade = cv2.CascadeClassifier(face_cascade_path)
eye_cascade = cv2.CascadeClassifier(eye_cascade_path)
smile_cascade = cv2.CascadeClassifier(smile_cascade_path)

def load_models():
    
    global human_model, object_model
    
    try:
        print(base_dir)
        human_model = torch.load(os.path.join(base_dir, 'yolov5s_model.pth'), 'yolov5s', pretrained=True, force_reload = False)
    except :
         human_model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True, force_reload = False)
         torch.save(human_model, 'yolov5s_model.pth')
        

    
load_models()
def decode_image(data: str):
    nparr = np.frombuffer(base64.b64decode(data), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Image decoding failed")
    return img
def image_dimensions(data: str):
    nparr = np.frombuffer(base64.b64decode(data), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Image decoding failed")
    return  img.shape[1], img.shape[0]
def encode_image(img: np.ndarray) -> dict:
    _, jpg_converted_img = cv2.imencode(".jpg", img)
    base64_encoded_img = base64.b64encode(jpg_converted_img.tobytes()).decode("utf-8")
    return {"image_data": base64_encoded_img}


def human_detection(image: np.ndarray):
    image = decode_image(image)  # Decode the base64 image data
    results = human_model(image)
     
    person_detected = False
    bounding_box = None

    for *xyxy, conf, cls in results.xyxy[0]:
        if int(cls) == PERSON_CLASS_LABEL:
            person_detected = True
            xyxy = [int(x) for x in xyxy]
            bounding_box = xyxy
            cv2.rectangle(image, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), (255, 0, 0), 2)

    return person_detected, bounding_box, image

def object_classification(image: np.ndarray):
    image = decode_image(image)  # Decode the base64 image data
    results = human_model(PIL.Image.fromarray(image))
    try:
        x = results.xyxy[0]
        print(f"x type: {type(x)}, x dtype: {x.dtype}")
        # No class filtering, so just pass through the results
    except Exception as e:
        print(f"Error during non-max suppression: {e}")
        raise
    return results, human_model.names
   


def face_detection(data: str) -> dict:
    img = decode_image(data)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(20, 20))

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

    return encode_image(img)

def face_smile_eye_detection(data: str) -> dict:
    img = decode_image(data)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = img[y:y + h, x:x + w]

        eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.5, minNeighbors=5, minSize=(5, 5))
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

        smile = smile_cascade.detectMultiScale(roi_gray, scaleFactor=1.5, minNeighbors=15, minSize=(25, 25))
        for (sx, sy, sw, sh) in smile:
            cv2.rectangle(roi_color, (sx, sy), (sx + sw, sy + sh), (0, 255, 0), 2)

    return encode_image(img)


