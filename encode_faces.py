import os
import pickle
import face_recognition
import numpy as np

dataset_path = "dataset"
encodings_file = "output/encodings.pickle"

known_face_encodings = []
known_face_names = []

print("🔄 Generating face encodings...")

for name in os.listdir(dataset_path):
    person_path = os.path.join(dataset_path, name)
    if os.path.isdir(person_path):
        for filename in os.listdir(person_path):
            file_path = os.path.join(person_path, filename)
            try:
                image = face_recognition.load_image_file(file_path)
                encoding = face_recognition.face_encodings(image)
                if encoding:
                    known_face_encodings.append(encoding[0])
                    known_face_names.append(name)
                    print(f"✔ Encoded: {name}")
                else:
                    print(f"❌ No face found in {file_path}")
            except Exception as e:
                print(f"❌ Error processing {file_path}: {e}")

# Save encodings
data = {"encodings": known_face_encodings, "names": known_face_names}
with open(encodings_file, "wb") as f:
    pickle.dump(data, f)

print("✅ Face encodings saved successfully!")
