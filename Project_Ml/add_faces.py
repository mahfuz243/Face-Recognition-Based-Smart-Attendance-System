import cv2
import pickle
import numpy as np
import os

# Initialize video capture and face detection
video = cv2.VideoCapture(0)
facedetect = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')

faces_data = []
i = 0

# Ensure 'data' folder exists
if not os.path.exists('data'):
    os.makedirs('data')

name = input("Enter Your Name: ")

while True:
    ret, frame = video.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facedetect.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        crop_img = gray[y:y + h, x:x + w]  # Use grayscale for consistency
        resized_img = cv2.resize(crop_img, (50, 50)).flatten()

        if len(faces_data) < 100 and i % 10 == 0:
            faces_data.append(resized_img)
        i += 1

        cv2.putText(frame, str(len(faces_data)), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (50, 50, 255), 1)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (50, 50, 255), 1)

    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) == ord('q') or len(faces_data) == 100:
        break

video.release()
cv2.destroyAllWindows()

faces_data = np.asarray(faces_data)

# Save names
if os.path.exists('data/names.pkl'):
    with open('data/names.pkl', 'rb') as f:
        names = pickle.load(f)
else:
    names = []

names += [name] * len(faces_data)

with open('data/names.pkl', 'wb') as f:
    pickle.dump(names, f)

# Save face data
if os.path.exists('data/faces_data.pkl'):
    with open('data/faces_data.pkl', 'rb') as f:
        faces = pickle.load(f)
    faces = np.vstack((faces, faces_data))
else:
    faces = faces_data

with open('data/faces_data.pkl', 'wb') as f:
    pickle.dump(faces, f)

print("Face data and names saved successfully!")
