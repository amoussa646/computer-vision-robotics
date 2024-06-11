
import cv2
import requests
import base64
from PIL import Image
import numpy as np
import streamlit as st

# Define the server endpoints
SERVER_URL = "http://localhost:9000"
ENDPOINTS = {
    "Human Detection": "/human_detection/human_detection",
    "Object Classification": "/object_classification/object_classification",
    "Face Detection": "/face_detection/face_detection",
    "Face, Smile, and Eye Detection": "/face_detection/face_smile_eye_detection",
    "Color Detection": "/color_detection/color_detection"
}
def display_frame(frame, bounding_box=None,detections=None):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    print("bounding box")
    print(bounding_box)
    if bounding_box:
        x1, y1, x2, y2 = bounding_box
        cv2.rectangle(rgb_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    if detections:
        for detection in detections[0]:
            x1, y1, x2, y2 = detection["bbox"]
            class_name = detection["class_name"]
            confidence = detection["confidence"]
            label = f"{class_name} ({confidence:.2f})"
            cv2.rectangle(rgb_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(rgb_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    pil_image = Image.fromarray(rgb_frame)
    return pil_image
def display_frame_objects(frame, bounding_box=None,label="",score=""):
    
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    if bounding_box:
        height, width, _ = rgb_frame.shape
        x1, y1, x2, y2 = bounding_box
        x1, y1, x2, y2 = int(x1 * width), int(y1 * height), int(x2 * width), int(y2 * height)
        cv2.rectangle(rgb_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        print("label")
        print(label)
        cv2.putText(rgb_frame, f"{label} ({score:.2f})", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    pil_image = Image.fromarray(rgb_frame)
    return pil_image

def process_frame(frame, endpoint, color_lower=None, color_upper=None):
    _, buffer = cv2.imencode('.jpg', frame)
    img_str = base64.b64encode(buffer).decode('utf-8')
    payload = {"data": img_str, "timestamp": "timestamp"}
    
    if endpoint == "/color_detection/color_detection":
        payload.update({
            "color_lower": color_lower,
            "color_upper": color_upper
        })
    
    response = requests.post(SERVER_URL + endpoint, json=payload)
    if response.status_code == 200:
      result = response.json()
      print(len(result))
      if(endpoint=="/object_classification/object_classification" ):
            detections = result  # Assuming result is already in the desired format
            return display_frame(frame,detections= detections)
      
      else:     
       
       status = result['status']
    #    if(result['labels']):
    #        print(result['labels'])
       print(status)
       if(status=='error'):
           print(result['message'])
       if(result['image_data'] is not None): 
        img_data = base64.b64decode(result['image_data'])
        img_np = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
        if 'bounding_box' in result:
            bounding_box = result['bounding_box']
            return display_frame(img,bounding_box= bounding_box)
        else:
             return display_frame(img)
        
        
       else:
        return frame
           
    else:
        st.error("Error in processing frame")
        return frame


def main():
    st.title("Computer Vision Client")
    task = st.selectbox("Select a task", list(ENDPOINTS.keys()))
    start_button = st.button("Start Processing")

    color_lower = (24, 100, 100)
    color_upper = (44, 255, 255)
    
    if task == "Color Detection":
        st.subheader("Specify color range for detection (HSV values)")
        lower_h = st.slider("Lower Hue", 0, 179, 24)
        lower_s = st.slider("Lower Saturation", 0, 255, 100)
        lower_v = st.slider("Lower Value", 0, 255, 100)
        upper_h = st.slider("Upper Hue", 0, 179, 44)
        upper_s = st.slider("Upper Saturation", 0, 255, 255)
        upper_v = st.slider("Upper Value", 0, 255, 255)
        color_lower = (lower_h, lower_s, lower_v)
        color_upper = (upper_h, upper_s, upper_v)

    if start_button:
        cap = cv2.VideoCapture(-1)
        stframe = st.empty()
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if task == "Color Detection":
                processed_frame = process_frame(frame, ENDPOINTS[task], color_lower=color_lower, color_upper=color_upper)
            else:
                processed_frame = process_frame(frame, ENDPOINTS[task])
            
            stframe.image(processed_frame, channels="BGR")

        cap.release()

if __name__ == "__main__":
    main()
