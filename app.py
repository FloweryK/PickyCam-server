import time
import base64
import argparse
import cv2
import json
import numpy as np
from flask import Flask, request
from flask_socketio import SocketIO
from serve_model import ServeModel


def base64_to_img(string):
    buffer = base64.b64decode(string)
    img_np = np.frombuffer(buffer, dtype=np.uint8)
    img = cv2.imdecode(img_np, flags=1)
    return img


def img_to_base64(img):
    _, buffer = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 45])
    string = str(base64.b64encode(buffer))
    return string


# define server
app = Flask(__name__)
app.config["SECRET_KEY"] = "test"
socketio = SocketIO(app)

# serve model
serve_model = ServeModel()


@app.route("/")
def home():
    return "hello world!"


@socketio.on("connect")
def on_connect():
    print("client connected, waiting for frames")


@socketio.on("disconnect")
def on_disconnect():
    print("client disconnected.")


@socketio.on("request")
def process(data):
    # check request arrival time
    # date_req_arrive = time.time()

    # read data
    string = data["frame"]
    mode = data["mode"]
    cameraType = data["cameraType"]
    # date_req_depart = data["date_req_depart"]

    # make text
    text = f"got request from client: {string[-10:]}"
    print(text)

    # convert from base64 to cv2 format
    # start = time.time()
    img = base64_to_img(string)
    # interval_base2img = (time.time() - start) * 1000

    # inference
    # start = time.time()
    img_processed = serve_model.inference(img, mode, cameraType)
    # interval_inference = (time.time() - start) * 1000

    # convert from cv2 format to base64
    # start = time.time()
    string_processed = img_to_base64(img_processed)
    # interval_img2base = (time.time() - start) * 1000

    # check response departure time
    # date_res_depart = time.time()

    # make json
    data = {
        "frame": string_processed,
        # "date_req_depart": date_req_depart,
        # "date_req_arrive": date_req_arrive,
        # "date_res_depart": date_res_depart,
        # "interval_base2img": interval_base2img,
        # "interval_inference": interval_inference,
        # "interval_img2base": interval_img2base,
    }

    data = json.dumps(data)

    # response
    socketio.emit("response", data, room=request.sid)


if __name__ == "__main__":
    # argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, help="host address", default="localhost")
    parser.add_argument("--port", type=int, help="port number", required=True)

    args = parser.parse_args()

    # server config
    HOST = args.host
    PORT = args.port

    socketio.run(app, host=HOST, port=PORT, debug=True)
