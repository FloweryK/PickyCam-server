import base64
import argparse
import cv2
import numpy as np
from flask import Flask
from flask_socketio import SocketIO
from serve_model import ServeModel


def base64_to_img(string):
    buffer = base64.b64decode(string)
    img_np = np.frombuffer(buffer, dtype=np.uint8)
    img = cv2.imdecode(img_np, flags=1)
    return img


def img_to_base64(img):
    _, buffer = cv2.imencode(".jpg", img)
    string = base64.b64encode(buffer)
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
def request(data):
    print("got frame from client: ", data["frame"][-10:])

    # read data
    string = data["frame"]

    # convert from base64 to cv2 format
    img = base64_to_img(string)

    # process image
    img_processed = serve_model.inference(img)

    # convert from cv2 format to base64
    string_processed = img_to_base64(img_processed)

    # response
    socketio.emit("response", {"processed": string_processed})


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
