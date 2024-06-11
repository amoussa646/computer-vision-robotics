import cv2
import os

base_dir = os.path.dirname(os.path.abspath(__file__))

face_cascade_path = os.path.join(base_dir, 'app/Cascades/haarcascade_frontalface_default.xml')
eye_cascade_path = os.path.join(base_dir, 'app/Cascades/haarcascade_eye.xml')
smile_cascade_path = os.path.join(base_dir, 'app/Cascades/haarcascade_smile.xml')

print(f"Face cascade path: {face_cascade_path}")
print(f"Eye cascade path: {eye_cascade_path}")
print(f"Smile cascade path: {smile_cascade_path}")

face_cascade = cv2.CascadeClassifier(face_cascade_path)
eye_cascade = cv2.CascadeClassifier(eye_cascade_path)
smile_cascade = cv2.CascadeClassifier(smile_cascade_path)

if face_cascade.empty():
    print(f"Failed to load face cascade from {face_cascade_path}")
else:
    print("Face cascade loaded successfully")

if eye_cascade.empty():
    print(f"Failed to load eye cascade from {eye_cascade_path}")
else:
    print("Eye cascade loaded successfully")

if smile_cascade.empty():
    print(f"Failed to load smile cascade from {smile_cascade_path}")
else:
    print("Smile cascade loaded successfully")
