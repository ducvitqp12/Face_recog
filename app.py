

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
# Load a sample picture and learn how to recognize it.
krish_image = face_recognition.load_image_file("Krish/krish.jpg")
krish_face_encoding = face_recognition.face_encodings(krish_image)[0]

# Load a second sample picture and learn how to recognize it.
bradley_image = face_recognition.load_image_file("Vit/vit.jpg")
bradley_face_encoding = face_recognition.face_encodings(bradley_image)[0]

# vit_image = face_recognition.load_image_file("Vit/vit.jpg")
# vit_face_encoding = face_recognition.face_encodings(vit_image)[0]

# Create arrays of known face encodings and their names
known_face_encodings = [
    krish_face_encoding,
    bradley_face_encoding
    # vit_face_encoding
]
known_face_names = [
    "Krish",
    "Bradly"
    # "Vit
]
# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True


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

def gen_frames():  
    ptime = 0
    while True:
        success, frame = camera.read()  # read the camera 
        ctime = time.time()
        fps = 1/(ctime - ptime)
        ptime = ctime
        cv2.putText(frame, f'FPS: {int(fps)}', (20,70), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,0), 2)
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
            face_names = []
            name = "Unknown"
            cur_name = "Unknown"
            for face_encoding in face_encodings:
                # See if the face is a match for the known face(s)
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                # Or instead, use the known face with the smallest distance to the new face
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_face_names[best_match_index]
                if name != cur_name:
                    publish(client, name)
                    cur_name = name
                face_names.append(name)
                
            font = cv2.FONT_HERSHEY_DUPLEX
            
            cv2.putText(frame, name, (220, 70), font, 1.0, (255, 255, 255), 1)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

def run():
    client.loop_start()


@app.route('/')
def index():
    return render_template('index.html')
@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
if __name__=='__main__':
    app.run(debug=True)