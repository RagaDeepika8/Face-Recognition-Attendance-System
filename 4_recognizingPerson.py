import numpy as np
import imutils
import pickle
import time
import cv2
import datetime
import csv
import os

# Debugging: Print current working directory
print("Current Working Directory:", os.getcwd())

# Define paths to model files (relative to the script's location)

cfg_path = "model/yolov3.cfg"
weights_path = "model/yolov3.weights"
embedding_model_path = "openface_nn4.small2.v1.t7"
recognizer_file = "output/recognizer.pickle"
label_enc_file = "output/le.pickle"
student_data_file = "student.csv"
attendance_log = "attendance.csv"

# Verify YOLO files exist

if not os.path.exists(cfg_path):
    print(f"Error: {cfg_path} does not exist! Please ensure yolov3.cfg is in the 'models' folder.")
    exit()
if not os.path.exists(weights_path):
    print(f"Error: {weights_path} does not exist! Please ensure yolov3.weights is in the 'models' folder.")
    exit()

# Load YOLO model for face detection
net = cv2.dnn.readNet(weights_path, cfg_path)
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]  # Updated for modern OpenCV

# Verify and load embedding model
if not os.path.exists(embedding_model_path):
    print(f"Error: {embedding_model_path} does not exist!")
    exit()
embedder = cv2.dnn.readNetFromTorch(embedding_model_path)

# Verify and load recognizer and label encoder
if not os.path.exists(recognizer_file):
    print(f"Error: {recognizer_file} does not exist!")
    exit()
if not os.path.exists(label_enc_file):
    print(f"Error: {label_enc_file} does not exist!")
    exit()
recognizer = pickle.loads(open(recognizer_file, "rb").read())
le = pickle.loads(open(label_enc_file, "rb").read())

# Load student data
if not os.path.exists(student_data_file):
    print(f"Error: {student_data_file} does not exist!")
    exit()
student_data = {}
with open(student_data_file, 'r') as csvFile:
    reader = csv.reader(csvFile)
    for row in reader:
        if len(row) >= 2:
            student_data[row[0]] = row[1]

print("[INFO] Starting video stream...")
cam = cv2.VideoCapture(0)
if not cam.isOpened():
    print("Error: Could not open webcam!")
    exit()
time.sleep(1.0)

while True:
    ret, frame = cam.read()
    if not ret:
        print("Error: Failed to capture frame from webcam!")
        break
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
            if confidence > 0.5 and class_id == 0:  # Class 0 is person in COCO dataset
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

            faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255, (96, 96), (0, 0, 0), swapRB=True, crop=False)
            embedder.setInput(faceBlob)
            vec = embedder.forward()

            preds = recognizer.predict_proba(vec)[0]
            j = np.argmax(preds)
            proba = preds[j]
            name = le.classes_[j]
            roll_number = student_data.get(name, "Unknown")

            # Record attendance with time and date
            now = datetime.datetime.now()
            timestamp = now.strftime("%Y-%m-%d %H:%M:%S")
            with open(attendance_log, 'a', newline='') as logFile:
                writer = csv.writer(logFile)
                writer.writerow([name, roll_number, timestamp])

            text = f"{name} : {roll_number} : {proba * 100:.2f}%"
            y_text = y - 10 if y - 10 > 10 else y + 10
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(frame, text, (x, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # Press 'ESC' to exit
        break

cam.release()
cv2.destroyAllWindows()