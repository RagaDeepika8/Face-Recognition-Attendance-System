
# 🎓 Real-Time Face Recognition Attendance System

A robust real-time face recognition attendance system using **YOLO**, **OpenFace embeddings**, and **SVM classifier**, integrated with a **MySQL database** and **Flask-SocketIO** for dynamic updates.


## 🚀 Features

🔍 **Real-time face detection** using YOLOv3
🧠 **Face recognition** via OpenFace + SVM classifier
📸 **Student registration** with automatic dataset creation (81 images)
📦 **Model retraining** pipeline triggered after every registration
💾 **MySQL Database** integration for student records and attendance
📈 **Attendance summary updates** with automatic percentage calculation
🔌 **SocketIO connection** to update web dashboard in real time
🧪 Fail-safes for:

  * Camera access issues
  * Bounding box validation
  * Small/empty face detection



## 🛠️ Tech Stack

| Component        | Technology              |
| ---------------- | ----------------------- |
| Face Detection   | YOLOv3                  |
| Face Embeddings  | OpenFace (Torch model)  |
| Face Classifier  | Scikit-learn SVM        |
| UI & Dashboard   | Flask + SocketIO        |
| Backend Language | Python                  |
| Database         | MySQL                   |
| Miscellaneous    | OpenCV, imutils, pickle |



## 📝 How It Works

### ➕ Registering a Student

1. Run the script and provide name & roll number.
2. Captures **81 face images** using webcam.
3. Stores data in `/dataset/<student_name>`.
4. Inserts student details into the MySQL database.
5. Triggers retraining of face recognition model.

### 🎯 Attendance Recognition

1. Loads YOLOv3 model and OpenFace embedder.
2. Uses webcam to capture faces in real-time.
3. Matches embeddings against trained SVM.
4. Logs student attendance in MySQL DB.
5. Updates `attendance_summary` table with:

   * Total days present
   * Percentage
   * Last updated timestamp

---

## 🖥️ Setup Instructions

### 🔧 Prerequisites

* Python ≥ 3.8
* MySQL server installed and running
* Webcam
* YOLOv3 weights and config (`model/yolov3.weights`, `yolov3.cfg`)
* OpenFace Torch model (`openface_nn4.small2.v1.t7`)

### 📦 Install Dependencies

pip install -r requirements.txt


### ⚙️ Setup MySQL

Create a database and run the following schema:


CREATE DATABASE facerecog;

USE facerecog;

CREATE TABLE students (
    id INT AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(100),
    roll_number VARCHAR(50)
);

CREATE TABLE attendance (
    id INT AUTO_INCREMENT PRIMARY KEY,
    student_id INT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE attendance_summary (
    student_id INT PRIMARY KEY,
    total_present INT,
    total_days INT,
    attendance_percentage FLOAT,
    last_updated DATETIME
);



## 💡 Future Enhancements

* Add GUI for student registration and dashboard
* Export attendance reports as CSV/PDF
* Add login authentication for admins
* Integrate with cloud storage for datasets
* Deploy as full-stack web app (React + Flask)

