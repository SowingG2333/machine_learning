import cv2
import numpy as np
from flask import Flask, render_template, Response, request

app = Flask(__name__)

# 假设你已经定义了 face_cascade
face_cascade = cv2.CascadeClassifier('path/to/your/cascade.xml')

def generate_frames(source):
    if source == 'camera':
        cap = cv2.VideoCapture(0)
    elif source.endswith(('.jpg', '.jpeg', '.png')):
        frame = cv2.imread(source)
        if frame is None:
            raise ValueError("Image not found or invalid image format")
        frames = [frame]
    else:
        cap = cv2.VideoCapture(source)

    while True:
        if source == 'camera' or not source.endswith(('.jpg', '.jpeg', '.png')):
            success, frame = cap.read()
            if not success:
                break
        else:
            frame = frames.pop(0)
        
        # 转为灰度图像
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 检测人脸
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # 在检测到的人脸上应用模糊处理
        for (x, y, w, h) in faces:
            face_region = frame[y:y+h, x:x+w]
            face_region = cv2.GaussianBlur(face_region, (99, 99), 30)
            frame[y:y+h, x:x+w] = face_region
        
        # 编码帧为 JPEG 格式
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        
        # 生成视频流
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

        if source.endswith(('.jpg', '.jpeg', '.png')):
            break

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    source = request.args.get('source', default='camera')
    return Response(generate_frames(source), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/upload_image', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return "No file part", 400
    file = request.files['file']
    if file.filename == '':
        return "No selected file", 400
    if file and file.filename.endswith(('.jpg', '.jpeg', '.png')):
        file_path = f"./{file.filename}"
        file.save(file_path)
        return Response(generate_frames(file_path), mimetype='multipart/x-mixed-replace; boundary=frame')
    else:
        return "Invalid file format", 400

if __name__ == '__main__':
    app.run(debug=True)