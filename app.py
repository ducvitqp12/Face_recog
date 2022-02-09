

from email.mime import image
from flask import Flask, render_template, Response
import cv2
import face_recognition
import numpy as np
import time
import random
import string

from paho.mqtt import client as mqtt_client

from os import listdir, remove
from os.path import isfile, join


app = Flask(__name__)
camera = cv2.VideoCapture(
    "http://192.168.1.10:6677/videofeed?username=&password=")


vit_image = face_recognition.load_image_file('Vit/vit.jpg')

# Create arrays of known face encodings and their names
known_face_encodings = []
known_face_names = []
known_face_path = []
known_face_access = []
# Initialize some variables
face_locations = []
face_encodings = []
process_this_frame = True
face_names = []
face_detected_time = []
flg = [0, True, False, False, ""]
data = []
ctime = [0, 0]


broker = '192.168.1.5'
port = 1883
topic1 = "doorlock/open"
topic2 = "doorlock/face_infor"
subTopic = "doorlock/capture"


# generate client ID with pub prefix randomly
sub_client_id = f'python-mqtt-{random.randint(0, 1000)}'
pub_client_id = f'python-mqtt-{random.randint(0, 1000)}'
# username = 'emqx'
# password = 'public'


def connect_mqtt(id: string):
    def on_connect(client, userdata, flags, rc):
        if rc == 0:
            print("Connected to MQTT Broker!")
        else:
            print("Failed to connect, return code %d\n", rc)

    client = mqtt_client.Client(id)
    # client.username_pw_set(username, password)
    client.on_connect = on_connect
    client.connect(broker, port)
    return client


pubclient = connect_mqtt(pub_client_id)


def current_milli_time():
    return round(time.time() * 1000)


def getData():
    with open('static/name.txt') as f:
        for line in f:
            item = [i for i in line.split()]
            data.append(item)
            image = face_recognition.load_image_file(item[1])
            face_encoding = face_recognition.face_encodings(image)[0]
            known_face_encodings.append(face_encoding)
            known_face_names.append(item[0])
            known_face_path.append(item[1])
            known_face_access.append(item[2])


def publish(client, message: string, topic: string):
    msg_count = 0
    # while True:
    #     time.sleep(1)
    result = client.publish(topic, message)
    # result: [0, 1]
    status = result[0]
    if status == 0:
        print(f"Send `{message}` to topic `{topic}`")
    else:
        print(f"Failed to send message to topic {topic}")
    msg_count += 1

def subscribe(client: mqtt_client):
    def on_message(client, userdata, msg):
        data = msg.payload
        message = data.decode("utf-8")
        # if(msg.topic == topic2):
        if(msg.topic == subTopic):
            if message == "capture":
                flg[3] = True
                if flg[1]:
                    while flg[3]:
                        success, frame = camera.read()  
                        if not success:
                            break
                        else:
                            if flg[2]:
                                recog(frame)
                            ret, buffer = cv2.imencode('.jpg', frame)
                            frame = buffer.tobytes()
                            if flg[3] and ctime[1] < 3:
                                if round(time.time()) > (ctime[0] + 1):
                                    ctime[0] = round(time.time())
                                    ctime[1] += 1
                                    i = round(time.time() * 1000)
                                    # print(i)
                                    f = open(f"static/temp_image/{str(i)}.jpg", "wb")
                                    f.write(frame)
                                    f.close()
                            elif ctime[1] == 3:
                                flg[3] = False
                                ctime[1] = 0
            if message == "start":
                flg[2] = True
        print(f"From topic {msg.topic} got message {message}")
        # f = open('output.jpg', "wb")
        # f.write(msg.payload)
        # print("Image Received")
        # f.close()

    # client.subscribe(topic2)
    client.subscribe(subTopic)
    client.on_message = on_message


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
        crtime = time.time()
        fps = 1/(crtime - ptime)
        ptime = crtime

        # cv2.imshow("Frame", frame)
        cv2.waitKey(1)
        if not success:
            break
        else:

            if flg[2]:
                recog(frame)

            font = cv2.FONT_HERSHEY_DUPLEX
            if not(flg[3]):
                cv2.putText(frame, str(flg[0]), (220, 70),
                            font, 1.0, (255, 255, 255), 1)
                cv2.putText(
                    frame, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 2)
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            if flg[3] and ctime[1] < 3:
                if round(time.time()) > (ctime[0] + 1):
                    ctime[0] = round(time.time())
                    ctime[1] += 1
                    i = round(time.time() * 1000)
                    # print(i)
                    f = open(f"static/temp_image/{str(i)}.jpg", "wb")
                    f.write(frame)
                    f.close()
            elif ctime[1] == 3:
                flg[3] = False
                ctime[1] = 0
                # flg[3] = False
            # if(flg[1] == 0):
            #     f = open('output.jpg', "wb")
            #     f.write(frame)
            #     f.close()
            #     flg[1] += 1
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


def recog(frame):
    # Resize frame of video to 1/4 size for faster face recognition processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_small_frame = small_frame[:, :, ::-1]

         # Only process every other frame of video to save time

         # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(
               rgb_small_frame, face_locations)
           # cur_name = "Unknown"
        if(flg[0] < 10):
                for face_encoding in face_encodings:
                    name = "Unknown"
                    # See if the face is a match for the known face(s)
                    matches = face_recognition.compare_faces(
                        known_face_encodings, face_encoding)
                    # Or instead, use the known face with the smallest distance to the new face
                    face_distances = face_recognition.face_distance(
                        known_face_encodings, face_encoding)
                    best_match_index = np.argmin(face_distances)
                    if matches[best_match_index]:
                        name = known_face_names[best_match_index]
                    for i in range(len(face_names)):
                        if(name == face_names[i]):
                            # print(face_names)
                            face_detected_time[i] = face_detected_time[i] + 1
                            break
                        if(name != face_names[i] and i == len(face_names)-1):
                            face_names.append(name)
                    flg[0] = flg[0] + 1

                    for (top, right, bottom, left), name in zip(face_locations, face_names):
                        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
                        top *= 4
                        right *= 4
                        bottom *= 4
                        left *= 4

                        # Draw a box around the face
                        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

                        # Draw a label with a name below the face
                        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
                        font = cv2.FONT_HERSHEY_DUPLEX
                        # cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
        else:
                name_detected = ""
                if (max(face_detected_time)/flg[0]) > 0.65:
                    name_detected = face_names[face_detected_time.index(
                        max(face_detected_time))]
                else:
                    name_detected = "Unknown"
                # publish(client, max(face_detected_time))
                flg[0] = 0
                print(face_names)
                face_names.clear()
                face_names.append("Unknown")
                print(face_names)
                for i in range(len(face_detected_time)):
                    face_detected_time[i] = 1
                print(face_detected_time)
                i = round(time.time() * 1000)
                if not(flg[4] == name_detected):
                    flg[4] = name_detected
                    state = False
                    if not(flg[4] == "Unknown"):
                        if known_face_access[known_face_names.index(flg[4])]:
                            state = True
                    message = '{"time": ' + str(i) + ',"name": "' + \
                        flg[4] + '","state": ' + str(state).lower() + '}'
                    if (flg[4] == "Unknown"):
                        flg[3] = True
                    publish(pubclient, message, topic2)
                    if state:
                        publish(pubclient, "open", topic1)
        


@app.route('/start')
def start():
    flg[2] = False
    flg[4] = ""
    publish(pubclient, "open", topic1)
    return "Nothing"


def run():
    subclient = connect_mqtt(sub_client_id)
    subscribe(subclient)
    subclient.loop_start()
    pubclient.loop_start()
    # client.loop_forever()


def myFunc(e):
    return e[2]


@app.route('/')
@app.route('/index.html')
def main_page():
    flg[1] = False
    return render_template('index.html')


@app.route('/storage.html')
def storage():
    gen_frames()
    data.sort(key=myFunc)
    return render_template('storage.html', data=data)


@app.route('/add_new.html')
def addNew():
    gen_frames()
    files = [f for f in listdir("static/temp_image")
             if isfile(join("static/temp_image", f))]
    onlyfiles = [""]*len(files)
    for i in range(len(files)):
        onlyfiles[i] = "static/temp_image/" + files[i]
    return render_template('add_new.html', onlyfiles=onlyfiles)

@app.route('/delete')
def delete():
    files = [f for f in listdir("static/temp_image")
             if isfile(join("static/temp_image", f))]
    onlyfiles = [""]*len(files)
    for i in range(len(files)):
        onlyfiles[i] = "static/temp_image/" + files[i]
        remove(onlyfiles[i])
    return render_template('add_new.html', onlyfiles=onlyfiles)

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    getData()
    face_detected_time = [1]*(len(known_face_names)+1)
    face_names.append("Unknown")
    run()
    app.run(debug=True)
