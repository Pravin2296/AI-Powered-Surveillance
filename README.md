-- Real-Time Surveillance and Alert System using YOLOv11 and DeepSORT --
  - This project implements an AI-powered real-time object detection and alerting system using YOLOv11, DeepSORT, and a multi-channel notification architecture (SMS, calls, buzzer).
  - It is designed for smart surveillance, especially useful in security, intrusion detection, and automation contexts.

-- Features --
 - Real-time object detection using YOLOv11
 - Live webcam feed monitoring
 - Smart object tracking with DeepSORT (avoids duplicate alerts)
 - Automated alert system:
 - SMS with attached detected image
 - Voice call using Twilio
 - Local buzzer sound trigger
 - Cloud image storage using Cloudinary
 - Offline alert queueing and retry mechanism
 - Supports custom object filtering (detect people, vehicles, etc.)

-- Technologies Used --
Component	                           Technology
Object Detection	                   YOLOv11 (via ultralytics)
Object Tracking                      DeepSORT
Alerts	                             Twilio API
Image Upload	                       Cloudinary API
Voice Feedback	                     pyttsx3 (Text-to-Speech)
Camera Interface	                   OpenCV
Local Storage	                       SQLite(optional for queueing)
Programming Lang	                   Python

-- OUTPUT SCREENSHOTS -- 
     link : https://output-screenshot-of-this-project.netlify.app/
