import cv2
import requests
from tkinter import *
from PIL import Image, ImageTk
def adjust_brightness(image, brightness=30):
    """Adjust the brightness of the image."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    v = cv2.add(v, brightness)
    v[v > 255] = 255
    v[v < 0] = 0
    final_hsv = cv2.merge((h, s, v))
    image = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return image
def send_image():
    # Capture an image
    cap = cv2.VideoCapture(0)
    frames = []
    for _ in range(5):
        ret, frame = cap.read()
        frames.append(frame)
    cap.release()

    # Select the middle frame
    middle_frame = frames[2]

    # Adjust the brightness of the middle frame
    middle_frame = adjust_brightness(middle_frame)

    # Encode the image as JPEG
    _, img_encoded = cv2.imencode('.jpg', middle_frame)
    _, img_encoded = cv2.imencode('.jpg', frame)
    response = requests.post('http://localhost:8000/process_image', files={'image': img_encoded.tostring()})
    text = response.json()['text']
    result_label.config(text=f'I see {text}')

# Create a Tkinter window
window = Tk()
window.title("Object Recognition")
window.geometry('400x200')

# Create a button
button = Button(window, text="What do you see?", command=send_image)
button.pack()

# Create a label for the result
result_label = Label(window, text="")
result_label.pack()

window.mainloop()
