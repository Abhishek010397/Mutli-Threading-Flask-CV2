from flask import Flask, render_template, Response
import cv2
import threading
import face_recognition
import os
import numpy as np
from vidgear.gears import VideoGear

Names = []
Images = []
ActiveThread = []

path = './Images'

dir = os.listdir(path)

for img in dir:
    getImage = cv2.imread(f'{path}/{img}')
    Images.append(getImage)
    Names.append(os.path.splitext(img)[0])

def invoke_Thread(list_id,purpose):
    print('Thread Invoked = ',purpose+list_id)
    thread = threading.Thread(target=camera_analysis, args=(list_id, purpose,), name=purpose)
    thread.start()
    current_thread_name = thread.name+list_id

    return current_thread_name

def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

encodedListKnown = findEncodings(Images)

cameras = [
    "http://admin:admin@172.24.144.24:8081/",
    "http://admin:admin@172.24.144.24:8082/"
]
app = Flask(__name__)

def find_camera(list_id):
    return cameras[int(list_id)]


def camera_analysis(camera_id,camera_purpose):
    cam = find_camera(camera_id)
    cap = VideoGear(source=cam, logging=True).start()
    if camera_purpose == 'face_recognition':
        while True:
            frame = cap.read()
            if frame is not None:
                imgS = cv2.rotate(frame, cv2.cv2.ROTATE_90_COUNTERCLOCKWISE)
                imgS = cv2.resize(imgS, (0, 0), None, 0.25, 0.25)
                imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
                facesCurFrame = face_recognition.face_locations(imgS)
                encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)
                for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
                    matches = face_recognition.compare_faces(encodedListKnown, encodeFace)
                    faceDis = face_recognition.face_distance(encodedListKnown, encodeFace)
                    matchIndex = np.argmin(faceDis)
                    if matches[matchIndex]:
                        name = Names[matchIndex].upper()
                        print(name)
            else:
                print("Not Matched")
    if camera_purpose == 'vehicle_detection':
        pass



def gen_frames(camera_id):
    cam = find_camera(camera_id)
    cap = VideoGear(source=cam, logging=True).start()
    while True:
        frame = cap.read()
        frame = cv2.rotate(frame, cv2.cv2.ROTATE_90_COUNTERCLOCKWISE)
        frame = cv2.resize(frame,(420,236))
        if frame is None:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/video_feed/<string:purpose>/<string:list_id>/', methods=["GET"])
def video_feed(list_id,purpose):
    incoming_thread = purpose+list_id
    if len(ActiveThread) == 0:
        ActiveThread.append(invoke_Thread(list_id,purpose))
        return Response(gen_frames(list_id),
                         mimetype='multipart/x-mixed-replace; boundary=frame')
    else:
        if len(ActiveThread) == 1:
            if [incoming_thread] != ActiveThread:
                ActiveThread.append(invoke_Thread(list_id,purpose))
            else:
                print('Thread is Active')
        if len(ActiveThread) > 1:
            res = [thread_name for thread_name in ActiveThread if (thread_name in incoming_thread)]
            if str(bool(res)) == 'False':
                ActiveThread.append(invoke_Thread(list_id, purpose))
            else:
                print('Thread is Active')
        return Response(gen_frames(list_id),
                            mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/', methods=["GET"])
def index():
    return render_template('index.html', camera_list=len(cameras), camera=cameras)

@app.route('/view_all',methods=["GET"])
def view_all():
    return render_template('view.html')

@app.route('/stop_thread/<string:list_id>')
def stop_analytics_thread(camera_list_id):
    pass

if __name__ == '__main__':
    app.run()