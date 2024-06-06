from flask import Flask, Response, render_template, send_file, send_from_directory, jsonify
import cv2
import datetime
import time
import threading
import subprocess
import os
import glob

app = Flask(__name__)

os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;udp'

def run_other_script(script_name):
  subprocess.run(["python", script_name])

IMAGE_FOLDER = 'F:\Anomaly Detection\static/results'
GIF_FOLDER = 'F:\Anomaly Detection\static/results'

def generate_frames():
    # Initialize camera
    # cap = cv2.VideoCapture(0)
    
    # Replace 'ZCRIXS' with camera IP password, '192.168.1.8' with external IP address of router
    url = 'rtsp://admin:ZCRIXS@192.168.1.8:554/H.264'
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)

    fps = cap.get(cv2.CAP_PROP_FPS)
    new_fps = fps
    # Define video writer properties
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Adjust codec if needed (e.g., 'MP4V')
    frame_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"videos/webcam_{timestamp}.mp4"
    writer = cv2.VideoWriter(filename, fourcc, new_fps, frame_size)

    start_time = time.time()
    frame_count=0
    video_num=1
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            yield 'error'
            print("Failed to grab frame!")
            break

        writer.write(frame)

        # Save video
        if time.time() - start_time > 8:
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            filename = f"videos/webcam_{timestamp}.mp4"

            writer = cv2.VideoWriter(filename, fourcc, new_fps, frame_size)  # Adjust FPS if needed
            start_time = time.time()

        # Convert frame to bytes for streaming
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    # Release resources
    cap.release()
    if writer is not None:
        writer.release()

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def home():
    other_script_thread = threading.Thread(target=run_other_script, args=("detection.py",))
    other_script_thread.start()
    files = os.listdir(GIF_FOLDER)  # List all files in your GIF folder
    return render_template('index.html', files=files)

@app.route('/latest-image')
def latest_image():
    list_of_files = glob.glob(os.path.join(IMAGE_FOLDER, '*.gif'))
    latest_file = max(list_of_files, key=os.path.getmtime)
    return send_file(latest_file)

@app.route('/gifs/<path:filename>')
def gifs(filename):
    """Route to serve the GIF files."""
    return send_from_directory(GIF_FOLDER, filename)

@app.route('/refresh_gifs')
def refresh_gifs():
    """Route to get the list of GIF files."""
    files = os.listdir(GIF_FOLDER)
    return jsonify({'files': files})

if __name__ == '__main__':
    app.run(debug=True)
