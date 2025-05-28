from imutils import paths
import numpy as np
import imutils
import pickle
import cv2
import os

dataset = "dataset"

embeddingFile = "output/embeddings.pickle"  # Initial name for embedding file
embeddingModel = "model/nn4.small2.v1.t7"  # Path to OpenFace model

# Load YOLO model for face detection
net = cv2.dnn.readNet("model/yolov3.weights", "model/yolov3.cfg")  # Updated paths
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Load embedding model
embedder = cv2.dnn.readNetFromTorch(embeddingModel)

# Get image paths
imagePaths = list(paths.list_images(dataset))

# Initialization
knownEmbeddings = []
knownNames = []
total = 0
conf = 0.5

# Process images one by one
for (i, imagePath) in enumerate(imagePaths):
    print("Processing image {}/{}".format(i + 1, len(imagePaths)))
    name = imagePath.split(os.path.sep)[-2]
    image = cv2.imread(imagePath)
    image = imutils.resize(image, width=600)
    (h, w) = image.shape[:2]

    # Detecting objects using YOLO
    blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
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
            if confidence > conf and class_id == 0:  # Class 0 is person in COCO dataset
                center_x = int(detection[0] * w)
                center_y = int(detection[1] * h)
                w_box = int(detection[2] * w)
                h_box = int(detection[3] * h)
                x = int(center_x - w_box / 2)
                y = int(center_y - h_box / 2)
                boxes.append([x, y, w_box, h_box])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    for i in range(len(boxes)):
        if i in indexes:
            x, y, w_box, h_box = boxes[i]

            # Ensure the bounding box is within the frame dimensions
            if x < 0 or y < 0 or w_box <= 0 or h_box <= 0 or x + w_box > w or y + h_box > h:
                print("Invalid bounding box. Skipping...")
                continue

            # Extract the face region
            face = image[y:y + h_box, x:x + w_box]
            (fH, fW) = face.shape[:2]

            # Ensure the face region is not too small
            if fW < 20 or fH < 20:
                print("Face too small. Skipping...")
                continue

            # Ensure the face region is not empty
            if face.size == 0:
                print("Empty face region. Skipping...")
                continue

            # Get face embeddings
            faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255, (96, 96), (0, 0, 0), swapRB=True, crop=False)
            embedder.setInput(faceBlob)
            vec = embedder.forward()

            # Append embeddings and names
            knownNames.append(name)
            knownEmbeddings.append(vec.flatten())
            total += 1

print("Embedding:{0} ".format(total))
data = {"embeddings": knownEmbeddings, "names": knownNames}
f = open(embeddingFile, "wb")
f.write(pickle.dumps(data))
f.close()
print("Process Completed")