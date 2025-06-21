from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO
import mysql.connector
import time
import threading
from datetime import datetime
import subprocess

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

# Database Connection Function
def get_db_connection():
    try:
        conn = mysql.connector.connect(
            host='DESKTOP-9EE6QDP', 
            user='root',
            password='password',
            database='facerecog'
        )
        return conn
    except mysql.connector.Error as err:
        print(f"Error connecting to database: {err}")
        return None

# Fetch Data from All 3 Tables
def fetch_all_data():
    conn = get_db_connection()
    if conn is None:
        return {}
    cursor = conn.cursor(dictionary=True)
    tables = ["students", "attendance", "attendance_summary"]
    data = {}
    for table in tables:
        try:
            cursor.execute(f"SELECT * FROM {table}")
            data[table] = cursor.fetchall()
        except mysql.connector.Error as err:
            print(f"Error fetching data from {table}: {err}")
            data[table] = []
    cursor.close()
    conn.close()
    return data

# Serialize Data
def serialize_data(data):
    serialized = {}
    for table, records in data.items():
        serialized[table] = []
        for record in records:
            if isinstance(record, dict):
                new_record = {}
                for key, value in record.items():
                    if isinstance(value, datetime):
                        new_record[key] = value.isoformat()
                    else:
                        new_record[key] = value
                serialized[table].append(new_record)
            else:
                print(f"Unexpected data format in {table}: {record}")
    return serialized

@app.route('/')
def index1():
    return render_template('index1.html')

@app.route('/dashboard')
def index():
    return render_template('index.html')

@app.route('/take_attendance', methods=['POST'])
def take_attendance():
    subprocess.Popen(["python", "../1_datasetCreation.py", "login"])
    return jsonify({'status': 'Attendance process started. Camera window should appear.'})

@app.route('/add_student', methods=['POST'])
def add_student():
    data = request.get_json()
    name = data.get('name')
    roll = data.get('roll_number')

    if not name or not roll:
        return jsonify({'error': 'Missing name or roll number'}), 400
    subprocess.Popen(["python", "../1_datasetCreation.py", "register", name, roll])
    return jsonify({'status': f'Registration started for {name}. Camera window should appear.'})

# Background DB monitor
def monitor_database():
    last_data = None
    while True:
        data = fetch_all_data()
        serialized_data_for_emit = serialize_data(data)
        if serialized_data_for_emit != last_data:
            print("Database Updated:", serialized_data_for_emit)
            socketio.emit('update_table', serialized_data_for_emit)
            last_data = serialized_data_for_emit
        time.sleep(3)

@socketio.on('connect')
def handle_connect():
    socketio.emit('update_table', serialize_data(fetch_all_data()))

if __name__ == '__main__':
    threading.Thread(target=monitor_database, daemon=True).start()
    socketio.run(app, debug=True)
    