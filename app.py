

from email.mime import image
from flask import Flask, render_template, Response
import cv2
import face_recognition
import numpy as np
import time
import random
import string

from paho.mqtt import client as mqtt_client

app=Flask(__name__)
camera = cv2.VideoCapture("http://192.168.1.10:6677/videofeed?username=&password=")


vit_image = face_recognition.load_image_file('Vit/vit.jpg')

# Create arrays of known face encodings and their names
known_face_encodings = []
known_face_names = []
known_face_access= []
# Initialize some variables
face_locations = []
face_encodings = []
process_this_frame = True
face_names = []
face_detected_time = []
flg = [0]


broker = '192.168.1.7'
port = 1883
topic = "python/mqtt"

# generate client ID with pub prefix randomly
client_id = f'python-mqtt-{random.randint(0, 1000)}'
# username = 'emqx'
# password = 'public'

def connect_mqtt():
    def on_connect(client, userdata, flags, rc):
        if rc == 0:
            print("Connected to MQTT Broker!")
        else:
            print("Failed to connect, return code %d\n", rc)

    client = mqtt_client.Client(client_id)
    # client.username_pw_set(username, password)
    client.on_connect = on_connect
    client.connect(broker, port)
    return client

client = connect_mqtt()


def getData():
    with open('static/name.txt') as f:
        for line in f:
            item = [i for i in line.split()]
            image = face_recognition.load_image_file(item[1])
            face_encoding = face_recognition.face_encodings(image)[0]
            known_face_encodings.append(face_encoding)
            known_face_names.append(item[0])
            known_face_access.append(item[2])

def publish(client, message: string):
    msg_count = 0
    while True:
        time.sleep(1)
        result = client.publish(topic, message)
        # result: [0, 1]
        status = result[0]
        if status == 0:
            print(f"Send `{message}` to topic `{topic}`")
        else:
            print(f"Failed to send message to topic {topic}")
        msg_count += 1

def increase_brightness(img, value=30):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    lim = 255 - value
    v[v > lim] = 255
    v[v <= lim] += value

    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img


def gen_frames():  
    ptime = 0
    while True:
        success, frame = camera.read()  # read the camera
        # frame = cv2.detailEnhance(frame, 100, 0.85)
        # frame = increase_brightness(frame, 50)
        ctime = time.time()
        fps = 1/(ctime - ptime)
        ptime = ctime
        
        # cv2.imshow("Frame", frame)
        cv2.waitKey(1)
        if not success:
            break
        else:
            # Resize frame of video to 1/4 size for faster face recognition processing
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
            rgb_small_frame = small_frame[:, :, ::-1]

            # Only process every other frame of video to save time
           
            # Find all the faces and face encodings in the current frame of video
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
            # cur_name = "Unknown"
            if( flg[0]<100 ):
                for face_encoding in face_encodings:
                    name = "Unknown"
                    # See if the face is a match for the known face(s)
                    matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                    # Or instead, use the known face with the smallest distance to the new face
                    face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                    best_match_index = np.argmin(face_distances)
                    if matches[best_match_index]:
                        name = known_face_names[best_match_index]
                    # if name != cur_name:
                    #     publish(client, name)
                    #     cur_name = name
                    for i in range(len(face_names)):
                        if(name == face_names[i]):
                            face_detected_time[i] = face_detected_time[i] + 1
                            break
                    face_names.append(name)
                    flg[0] = flg[0] + 1
            else:
                name_detected = ""
                if (max(face_detected_time)/flg[0]) > 0.65:
                    name_detected = face_names[face_detected_time.index(max(face_detected_time))]
                else:
                    name_detected = "Unknown"
                flg[0] = 0
                face_names.clear()
                for i in face_detected_time:
                    i = 1
                publish(client, name_detected)
                
            font = cv2.FONT_HERSHEY_DUPLEX
            
            # cv2.putText(frame, flg[0], (220, 70), font, 1.0, (255, 255, 255), 1)
            cv2.putText(frame, f'FPS: {int(fps)}', (20,70), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,0), 2)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

def run():
    client.loop_start()


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/index.html')
def main_page():
    return render_template('index.html')
    
@app.route('/storage.html')
def storage():
    return render_template('storage.html')
@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
if __name__=='__main__':
    getData()
    face_detected_time = [1]*len(known_face_names)
    app.run(debug=True)