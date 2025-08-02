import os
import cv2
import torch
import time
import pyttsx3
import threading
import requests
import winsound
import json
from twilio.rest import Client
from ultralytics import YOLO
from datetime import datetime
import cloudinary
import cloudinary.uploader
from deep_sort_realtime.deepsort_tracker import DeepSort

# Twilio Credentials
TWILIO_SID = ""
TWILIO_AUTH_TOKEN = ""
TWILIO_PHONE = "+"
RECEIVER_PHONE = "+"

# Initialize Twilio Client
client = Client(TWILIO_SID, TWILIO_AUTH_TOKEN)

# Cloudinary Configuration
cloudinary.config(
    cloud_name="",
    api_key="",
    api_secret=""
)

# Initialize Text-to-Speech
engine = pyttsx3.init()
engine.setProperty("rate", 150)

# Load YOLOv11
device = "cuda" if torch.cuda.is_available() else "cpu"
model = YOLO("yolo11n.pt").to(device)

# Object Tracker
tracker = DeepSort(max_age=30)

# Get COCO classes
coco_classes = model.names

print("Available classes:")
for idx, class_name in coco_classes.items():
    print(f"{idx}: {class_name}")

# Object selection
while True:
    try:
        selected_index = int(input("Enter the number corresponding to the object you want to detect: "))
        if selected_index in coco_classes:
            selected_object = coco_classes[selected_index]
            print(f"Detecting: {selected_object}")
            break
        else:
            print("Invalid selection. Please enter a number from the list.")
    except ValueError:
        print("Invalid input. Please enter a valid number.")

# Open webcam
cap = cv2.VideoCapture(0)

last_alert_sent = 0
alert_interval = 30

def check_internet():
    try:
        requests.get("https://www.google.com", timeout=5)
        return True
    except requests.ConnectionError:
        return False

def upload_to_cloudinary(file_path):
    """Uploads an image to Cloudinary and returns the URL."""
    if not os.path.exists(file_path):
        print(f"ERROR: File not found - {file_path}")
        return None
    try:
        print(f" Uploading {file_path} to Cloudinary...")
        response = cloudinary.uploader.upload(file_path)
        image_url = response.get("secure_url")
        print(f"Uploaded successfully: {image_url}")
        return image_url
    except Exception as e:
        print(f"Cloudinary Upload Failed: {str(e)}")
        return None

def send_sms_alert(image_url, detected_object):
    """Sends an SMS alert with the detected image."""
    if not image_url:
        print(" ERROR: Image URL is None, skipping SMS.")
        return
    try:
        print("Sending SMS alert...")
        message = client.messages.create(
            body=f"ALERT: A {detected_object} has been detected!",
            from_=TWILIO_PHONE,
            to=RECEIVER_PHONE,
            media_url=[image_url]
        )
        print(f"SMS sent! SID: {message.sid}")
    except Exception as e:
        print(f"Twilio SMS Failed: {str(e)}")

def alert_system(image_path):
    """Handles the alert system - Upload & send SMS."""
    global last_alert_sent
    current_time = time.time()
    if current_time - last_alert_sent < alert_interval:
        print("Alert already sent recently, skipping...")
        return

    if check_internet():
        image_url = upload_to_cloudinary(image_path)
        send_sms_alert(image_url, selected_object)
    else:
        print("No internet, alert will be sent later.")
    last_alert_sent = current_time

# Main loop
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model.predict(frame)
    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            if coco_classes[cls_id] == selected_object and conf > 0.5:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{selected_object} ({conf:.2f})", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                image_path = f"detection_{time.time()}.jpg"
                cv2.imwrite(image_path, frame)
                print(f"📸 Image saved: {image_path}")

                alert_system(image_path)

    cv2.imshow("YOLOv11 Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release everything properly
cap.release()
cv2.destroyAllWindows()
engine.stop()
del engine  # Properly release TTS engine to avoid warnings
