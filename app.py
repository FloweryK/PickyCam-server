import time
import datetime
import argparse
import cv2
import json
from flask import Flask, request
from flask_socketio import SocketIO
from serve_model import ServeModel
from utils.images import base64_to_img, img_to_base64

# global? should i change this as a session variable?
is_processing = False

# define server
app = Flask(__name__)
app.config["SECRET_KEY"] = "test"
socketio = SocketIO(app)

# serve model
serve_model = ServeModel()
img = cv2.imread('dummy.jpg')
serve_model.inference(img)


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
    print("request")

    global is_processing

    if not is_processing:
        is_processing = True
        
        # read data
        string = data["frame"]
        settings = data["settings"]

        print(datetime.datetime.now())
        print(settings)

        # convert from base64 to cv2 format
        img = base64_to_img(string)

        # inference
        img_processed = serve_model.inference(img, settings)

        # convert from cv2 format to base64
        string_processed = img_to_base64(img_processed)

        # make json
        res = json.dumps({
            "frame": string_processed,
        })

        # response
        socketio.emit("response", res, room=request.sid)

        is_processing = False
    else:
        print("infenrence is ongoing... skip this request")


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
