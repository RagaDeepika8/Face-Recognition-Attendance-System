import numpy as np
import imutils
import pickle
import time
import cv2
import datetime
import csv
import os

import mysql.connector

# Establish a connection to the MySQL database
conn = mysql.connector.connect(
     host='DESKTOP-9EE6QDP',  # Your MySQL server host
    user='root',             # Your MySQL username
    password='password',     # Your MySQL password
    database='facerecog' 
)

# Create a cursor object
cursor = conn.cursor()

# Get the absolute paths
BASE_DIR = os.getcwd()
MODEL_DIR = os.path.join(BASE_DIR, "model")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

# Define file paths
YOLO_WEIGHTS = os.path.join(MODEL_DIR, "yolov3.weights")
YOLO_CFG = os.path.join(MODEL_DIR, "yolov3.cfg")
EMBEDDING_MODEL = os.path.join(MODEL_DIR, "nn4.small2.v1.t7")
RECOGNIZER_FILE = os.path.join(OUTPUT_DIR, "recognizer.pickle")
LABEL_ENCODER_FILE = os.path.join(OUTPUT_DIR, "le.pickle")
STUDENT_CSV = os.path.join(BASE_DIR, "student.csv")
ATTENDANCE_LOG = os.path.join(BASE_DIR, "attendance.csv")

# Check if files exist
for file_path in [YOLO_WEIGHTS, YOLO_CFG, EMBEDDING_MODEL, RECOGNIZER_FILE, LABEL_ENCODER_FILE, STUDENT_CSV]:
    if not os.path.exists(file_path):
        print(f"[ERROR] Missing file: {file_path}")
        exit()

# Load YOLO model for face detection
net = cv2.dnn.readNet(YOLO_WEIGHTS, YOLO_CFG)
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]

# Load embedding model
embedder = cv2.dnn.readNetFromTorch(EMBEDDING_MODEL)

# Load recognizer and label encoder
recognizer = pickle.loads(open(RECOGNIZER_FILE, "rb").read())
le = pickle.loads(open(LABEL_ENCODER_FILE, "rb").read())

# Load student data
# Create table if it doesn't exist
cursor.execute('''
    CREATE TABLE IF NOT EXISTS attendance (
        id INT AUTO_INCREMENT PRIMARY KEY,
        name VARCHAR(255),
        roll_number VARCHAR(255),
        timestamp DATETIME
    )
''')

# Inside your face recognition loop, after identifying the person:
now = datetime.datetime.now()
timestamp = now.strftime('%Y-%m-%d %H:%M:%S')

# Insert attendance record into MySQL
cursor.execute('''
    INSERT INTO attendance (name, roll_number, timestamp)
    VALUES (%s, %s, %s)
''', (name, roll_number, timestamp))

# Commit the transaction
conn.commit()


# Confidence threshold for face detection
conf_threshold = 0.5

print("[INFO] Starting video stream...")
cam = cv2.VideoCapture(0)
time.sleep(1.0)

while True:
    _, frame = cam.read()
    frame = imutils.resize(frame, width=600)
    height, width, channels = frame.shape

    # Detecting objects using YOLO
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    # Information to draw bounding boxes
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > conf_threshold and class_id == 0:  # Class 0 is person in COCO dataset
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            face = frame[y:y+h, x:x+w]
            (fH, fW) = face.shape[:2]

            if fW < 20 or fH < 20:
                continue

            # Get face embeddings
            faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255, (96, 96), (0, 0, 0), swapRB=True, crop=False)
            embedder.setInput(faceBlob)
            vec = embedder.forward()

            # Recognize face
            preds = recognizer.predict_proba(vec)[0]
            j = np.argmax(preds)
            proba = preds[j]
            name = le.classes_[j]

            # Fetch roll number from student data
            roll_number = student_data.get(name, "Unknown")

            # Record attendance with time and date
            now = datetime.datetime.now()
            timestamp = now.strftime("%Y-%m-%d %H:%M:%S")
            with open(ATTENDANCE_LOG, 'a') as logFile:
                logFile.write(f"{name},{roll_number},{timestamp}\n")

            # Display results
            text = f"{name} : {roll_number} : {proba * 100:.2f}%"
            y = y - 10 if y - 10 > 10 else y + 10
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # Press 'ESC' to exit
        break

cam.release()
cv2.destroyAllWindows()
