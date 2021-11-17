from flask import Flask, request, jsonify
from server.inference import Inferencer
from PIL import Image
import numpy as np
import cv2
import io
import base64
import time
import ssl

# server config
HOST = "localhost"
PORT = 4321

# inference model
model = Inferencer()

# define server
app = Flask(__name__)


@app.route("/")
def home():
    return "hello world!"


@app.route("/upload", methods=["POST"])
def upload():
    # decode base64 image
    img = request.json["data"]
    print("got data from client!: ", len(img))
    img = base64.b64decode(img)
    img = np.frombuffer(img, dtype=np.uint8)
    img = cv2.imdecode(img, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # inference
    img = model.inference(img)
    img = Image.fromarray(img.astype("uint8"))

    # encode image to base64
    rawBytes = io.BytesIO()
    img.save(rawBytes, "JPEG")
    rawBytes.seek(0)
    img = base64.b64encode(rawBytes.read())
    print("sent: ", len(img))

    return jsonify({"image": str(img)})


if __name__ == "__main__":
    app.run(debug=True, host=HOST, port=PORT)
