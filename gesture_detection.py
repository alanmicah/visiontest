# Detect hand gestures using openCV
import cv2 as cv
from outcome import capture
import cvzone
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

BaseOptions = mp.tasks.BaseOptions
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
GestureRecognizerResult = mp.tasks.vision.GestureRecognizerResult
VisionRunningMode = mp.tasks.vision.RunningMode

cap = cv.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)
detector = cvzone.HandDetector(detectionCon=0.5, maxHands=1)

model_path = 'data/gesture_recognizer.task'
base_options = python.BaseOptions(model_asset_path=model_path)


# Create a gesture recognizer instance with the live stream mode:
def print_result(result: GestureRecognizerResult, output_image: mp.Image, timestamp_ms: int):
    print('gesture recognition result: {}'.format(result))

options = GestureRecognizerOptions(
    base_options=BaseOptions(model_asset_path='/path/to/model.task'),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=print_result)
with GestureRecognizer.create_from_options(options) as recognizer:
  # The detector is initialized. Use it here.

while True:
    # Get image frame
    success, img = cap.read()

    # Find the hand and its landmarks
    img = detector.findHands(img)
    lmList, bbox = detector.findPosition(img)
    
    cv.waitKey(1)
    # Display
    cv.imshow("Image", img)