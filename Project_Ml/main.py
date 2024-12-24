# Import required libraries
from sklearn.neighbors import KNeighborsClassifier
import cv2
import pickle
import numpy as np
import os
import csv
import time
from datetime import datetime
from win32com.client import Dispatch

def speak(text):
    speaker = Dispatch("SAPI.SpVoice")
    speaker.Speak(text)

# Initialize video capture
video = cv2.VideoCapture(0)

# Load face detection model
facedetect = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')

# Load names and faces data
with open('data/names.pkl', 'rb') as f:
    LABELS = pickle.load(f)

with open('data/faces_data.pkl', 'rb') as f:
    FACES = pickle.load(f)

# Ensure FACES and LABELS consistency
if len(FACES) != len(LABELS):
    min_length = min(len(FACES), len(LABELS))
    FACES, LABELS = FACES[:min_length], LABELS[:min_length]

# Initialize the KNN model
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(FACES, LABELS)

# Background image
imgBackground = cv2.imread('Smart_Attendance_System.png')
imgBackground = cv2.resize(imgBackground, (1280, 720))

COL_NAMES = ['NAME', 'TIME', 'DATE']

# Set to track recorded names
recorded_names = set()

while True:
    ret, frame = video.read()
    if not ret:
        print("Failed to grab frame. Exiting...")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facedetect.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        crop_img = gray[y:y + h, x:x + w]
        resized_img = cv2.resize(crop_img, (50, 50)).flatten().reshape(1, -1)

        try:
            output = knn.predict(resized_img)
        except Exception as e:
            print(f"Error in prediction: {e}")
            continue

        name = str(output[0])

        # Draw bounding box
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.rectangle(frame, (x, y - 40), (x + w, y), (50, 50, 255), -1)
        cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # Save attendance only if the 'o' key is pressed and the name is not already recorded
        k = cv2.waitKey(1)
        if k == ord('o') and name not in recorded_names:
            # Mark attendance
            ts = time.time()
            date = datetime.fromtimestamp(ts).strftime("%d-%m-%Y")
            timestamp = datetime.fromtimestamp(ts).strftime("%H:%M:%S")
            attendance = [name, str(timestamp), str(date)]

            # Save attendance
            if not os.path.exists('Attendance'):
                os.makedirs('Attendance')

            attendance_file = f'Attendance/Attendance_{date}.csv'
            file_exists = os.path.isfile(attendance_file)

            with open(attendance_file, 'a' if file_exists else 'w', newline='') as f:
                writer = csv.writer(f)
                if not file_exists:
                    writer.writerow(COL_NAMES)
                writer.writerow(attendance)

            recorded_names.add(name)  # Add name to the recorded list
            speak(f"Attendance for {name} is marked. Thank you.")
            time.sleep(2)

    # Embed video frame into background
    imgBackground[162:162 + 480, 55:55 + 640] = cv2.resize(frame, (640, 480))
    cv2.imshow("Smart Attendance System", imgBackground)

    k = cv2.waitKey(1)
    if k == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
