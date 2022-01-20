# python 3.6

import random
import time

from os import listdir
from os.path import isfile, join
onlyfiles = [f for f in listdir("static/temp_image") if isfile(join("static/temp_image", f))]
for item in onlyfiles:
    tmp = "static/temp_image/" + item
    item = tmp
    print(tmp)
# from paho.mqtt import client as mqtt_client


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


def publish(client, message):
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


list = [10, 20, 70, 5, 15, 20]
data = []

def current_milli_time():
    return round(time.time() * 1000)

def getData():
    # with open('static/name.txt') as f:
    #     for line in f:
    #         item = [i for i in line.split()]
    #         data.append(item)
    #         print(item[1])
    #         print(data)
    #         # print(list)
    #     print(list)
    #     # list.clear()
    #     print(max(list))
    #     # print(list.index(1))
    i = str(current_milli_time())
    print(onlyfiles)

if __name__ == '__main__':
    getData()